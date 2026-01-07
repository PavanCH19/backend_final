const path = require("path");
const fs = require("fs");
const User = require("../modules/userSchema");
const { executePythonModel } = require("../utils/pythonConnector");

const firstSessionPath = path.join(__dirname, "../python_models/question_recomendation/first_session.py");

// ---------------- MODEL CONFIG ----------------
const MODEL_CONFIGS = {
    first_session: {
        scriptPath: firstSessionPath,
        pythonPath: "python",
        envVars: {}
    },
    subjective_eval: {
        scriptPath: path.join(__dirname, "../python_models/evaluations/subjective_eval.py"),
        pythonPath: "python",
        envVars: {}
    },
    audio_eval: {
        scriptPath: path.join(__dirname, "../python_models/evaluations/audio_eval.py"),
        pythonPath: "python",
        envVars: {}
    },
    voice_analysis: {
        scriptPath: path.join(__dirname, "../python_models/evaluation/voice_analysis.py"),
        pythonPath: "python",
        envVars: {}
    },
    text_eval: {
        scriptPath: path.join(__dirname, "../python_models/evaluation/evaluate_text.py"),
        pythonPath: "python",
        envVars: {}
    },
    code_evaluation: {
        scriptPath: path.join(__dirname, "../python_models/evaluation/evaluate_code.py"),
        pythonPath: "python",
        envVars: {}
    },
    build_summary: {
        scriptPath: path.join(__dirname, "../python_models/question_recomendation/build_user_summary_from_answers.py"),
        pythonPath: "python",
        envVars: {}
    },
    adaptive_recommendation: {
        scriptPath: path.join(__dirname, "../python_models/question_recomendation/recommend_adaptive_questions.py"),
        pythonPath: "python",
        envVars: {}
    }
};

const startAdaptiveInterviewSession = async (user, domain, lastSession) => {
    console.log(`[Adapter] Starting adaptive session for user ${user._id} in ${domain}`);

    // 1. Gather all previously asked question IDs across all sessions for this domain
    const domainSessions = user.interview_sessions.filter(s => s.domain === domain);
    let allAskedIds = [];
    domainSessions.forEach(s => {
        if (s.questions && Array.isArray(s.questions)) {
            allAskedIds = [...allAskedIds, ...s.questions];
        }
    });

    // 2. Prepare Payload
    // We attach specific fields that the Python script expects from "session_data"
    // Since Schema structure might differ slightly from what script expects (e.g. 'answers' list),
    // we structure it to be robust. 
    // The script uses: 'skill_analysis', 'session_stats' (inside detailed_summary), 'answers' (optional for type check but stats are better)

    const lastSessionObj = lastSession.toObject ? lastSession.toObject({ flattenMaps: true }) : lastSession;

    // Flatten session_stats if nested in detailed_summary for easier access if script expects it at top level?
    // The script checks: session_data.get("session_stats", {})
    // In our schema: session.detailed_summary.session_stats
    // So we should map it.

    // Prepare proper skill_analysis for Python
    // Python expects: { "SkillName": { "average_score": 50, ... }, ... }
    // Schema provides: { skill_averages: { "SkillName": 50 }, ... }
    let rawSkillAnalysis = lastSessionObj.detailed_summary?.skill_analysis || lastSessionObj.skill_analysis || {};

    // Normalize: if it's the schema format (has skill_averages), convert to rich format
    let normalizedSkillAnalysis = {};

    // If we have the rich detailed analysis, use it (it usually doesn't have skill_averages key at top level)
    // But check if it has keys that resemble skills
    if (rawSkillAnalysis.skill_averages) {
        // Log for debugging
        console.log("[Adapter] Normalizing Schema-style skill_analysis for Python");

        const avgs = rawSkillAnalysis.skill_averages || {};
        // Iterate skills and create synthetic rich objects
        for (const [skill, score] of Object.entries(avgs)) {
            normalizedSkillAnalysis[skill] = {
                average_score: score,
                // We fake these since we don't have history in the simple schema
                questions_tested: 0,
                questions: [],
                status: score < 40 ? "struggling" : (score < 70 ? "developing" : "mastered"),
                priority: score < 40 ? "high" : "low"
            };
        }

        // Also copy stronger/weaker lists if useful, though Python re-derives
        normalizedSkillAnalysis.stronger_skills = rawSkillAnalysis.stronger_skills || [];
        normalizedSkillAnalysis.weaker_skills = rawSkillAnalysis.weaker_skills || [];

    } else {
        // Assume it's already in the rich format (or empty)
        normalizedSkillAnalysis = rawSkillAnalysis;
    }

    const preparedSessionData = {
        ...lastSessionObj,
        session_stats: lastSessionObj.detailed_summary?.session_stats || {},
        recommendations: lastSessionObj.detailed_summary?.recommendations || {},
        skill_analysis: normalizedSkillAnalysis,
        // We don't store raw answers in DB session usually, but we have stats.
        answers: []
    };

    const payload = {
        user_id: user._id.toString(),
        domain: domain,
        last_session: preparedSessionData,
        questions_dir: domainsDir,
        already_asked_ids: allAskedIds,
        k: 5
    };
    console.log("payload =", JSON.stringify(payload, null, 2));

    // 3. Call Python
    const wrapper = await executePythonModel(
        MODEL_CONFIGS.adaptive_recommendation,
        "generate_adaptive_recommendations",
        payload,
        45000
    );

    // Handle error
    if (wrapper.error) {
        throw new Error(wrapper.error);
    }

    // Extract actual result content from the bridge wrapper
    const result = wrapper.result;

    // 4. Return formatted result
    // result.questions_recommended is list of { question: {...}, ... }
    return {
        message: result.message || "Adaptive session ready",
        skill_analysis: result.skill_analysis || {},
        questions: result.questions_recommended ? result.questions_recommended.map(item => item.question || item) : []
    };
};

const startInterview = async (req, res) => {
    try {
        const userId = req.user.id;
        const domain = req.query.domain;
        if (!domain) throw new Error("Query parameter 'domain' required")
        // -----------------------------------------
        // Fetch user
        // -----------------------------------------
        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({
                success: false,
                message: "User not found"
            });
        }

        // -----------------------------------------
        // Find domain sessions
        // -----------------------------------------
        const domainSessions = user.interview_sessions.filter(
            (s) => s.domain === domain
        );

        const isFirstSession = domainSessions.length === 0;
        let response = null;

        // ============================================================
        // FIRST SESSION (Call Python)
        // ============================================================
        if (isFirstSession) {
            const payload = {
                _id: user._id.toString(),
                skills: user.skills || [],
                target_domain: domain
            };
            console.log("payload", payload)
            // PYTHON: first_session_recommendations()
            response = await executePythonModel(
                MODEL_CONFIGS.first_session,
                "first_session_recommendations",
                payload,
                60000
            );
        }

        // ============================================================
        // 🔵 ADAPTIVE SESSION
        // ============================================================
        else {
            const lastSession =
                domainSessions[domainSessions.length - 1];

            response = await startAdaptiveInterviewSession(
                user,
                domain,
                lastSession
            );

            // user.interview_sessions.push({
            //     session_id: `sess_${Date.now()}`,
            //     domain,
            //     session_type: "adaptive_recommendation",
            //     session_number: domainSessions.length + 1,
            //     message: response.message,
            //     skill_analysis: response.skill_analysis || {},
            //     questions: response.questions.map((q) => q._id),
            //     startedAt: new Date(),
            //     status: "ongoing"
            // });

            // await user.save();
        }

        return res.json({
            success: true,
            message: "Interview session initialized.",
            data: response
        });



    } catch (error) {
        // console.error("Interview start error:", error);
        return res.status(500).json({
            success: false,
            message: "Server error initializing interview session",
            error: error.message
        });
    }
};

// ----------------------------------------------------------------
// QUESTION & EVALUATION HELPERS
// ----------------------------------------------------------------
const domainsDir = path.join(__dirname, "../python_models/question_recomendation/domains");
const DOMAIN_CACHE = {};

const getQuestionsForDomain = (domain) => {
    // Return cached if available
    if (DOMAIN_CACHE[domain]) return DOMAIN_CACHE[domain];

    const questions = {};
    // Sanitize domain to prevent path traversal
    const safeDomain = path.basename(domain);
    const filePath = path.join(domainsDir, `${safeDomain}.json`);

    if (fs.existsSync(filePath)) {
        try {
            const content = fs.readFileSync(filePath, 'utf-8');
            const data = JSON.parse(content);
            if (Array.isArray(data)) {
                data.forEach(q => {
                    questions[q._id] = q;
                });
            }
        } catch (e) {
            console.error(`Error reading domain file ${safeDomain}:`, e);
        }
    } else {
        console.warn(`Domain file not found: ${filePath}`);
    }

    DOMAIN_CACHE[domain] = questions;
    return questions;
};

const evaluateMCQ = (question, userAnswer) => {
    if (!question || !question.answer) {
        return { correct: false, score: 0, reason: "Question data invalid" };
    }
    // Simple string equality check
    const isCorrect = userAnswer === question.answer;
    return {
        correct: isCorrect,
        score: isCorrect ? 10 : 0,
        expected_answer: question.answer,
        user_answer: userAnswer
    };
};

const normalizeQid = (key) => {
    if (typeof key !== "string") return key;
    if (key.startsWith("answers[")) {
        return key.slice(8, -1); // remove answers[  and  ]
    }
    return key;
};

// --- Utility: Pretty console logger
const logFormKeys = (body, files) => {
    console.log("\n========== FORM DATA RECEIVED ==========");
    console.log("Body Keys:", Object.keys(body));
    console.log("Files Keys:", files ? Object.keys(files) : "No Files");
    console.log("========================================\n");
};

// ----------------- Performance helpers -----------------
// Limit concurrent python model calls to avoid spawning many heavy processes
const CONCURRENCY_LIMIT = 3;
const _semaphore = { count: 0, queue: [] };
const acquire = () => new Promise((res) => {
    if (_semaphore.count < CONCURRENCY_LIMIT) {
        _semaphore.count++;
        return res();
    }
    _semaphore.queue.push(res);
});
const release = () => {
    _semaphore.count--;
    if (_semaphore.queue.length) {
        _semaphore.count++;
        const next = _semaphore.queue.shift();
        next();
    }
};
const limitedExecutePythonModel = async (config, funcName, payload, timeout) => {
    await acquire();
    try {
        return await executePythonModel(config, funcName, payload, timeout);
    } finally {
        release();
    }
};

// Simple in-memory cache for text evaluations to avoid duplicate expensive runs
const textEvalCache = new Map();
const cachedTextEval = async (question, user_answer, timeout) => {
    const answerStr = typeof user_answer === 'string' ? user_answer.trim() : String(user_answer || '');
    if (!answerStr) return null;
    const qid = question && (question._id || question.id || question.qid) ? (question._id || question.id || question.qid) : JSON.stringify(question).slice(0, 50);
    const key = `${qid}::${answerStr.slice(0, 200)}`;
    if (textEvalCache.has(key)) {
        return textEvalCache.get(key);
    }
    const p = limitedExecutePythonModel(MODEL_CONFIGS.text_eval, "evaluate_text", { question, user_answer: user_answer }, timeout || 90000)
        .then(res => res?.result || res)
        .catch(err => { textEvalCache.delete(key); throw err; });
    textEvalCache.set(key, p);
    return p;
};

const updateUserInterviewSession = async (userId, domain, summaryResult, responseData) => {
    const user = await User.findById(userId);
    if (!user) {
        console.error(`[Background] User ${userId} not found`);
        return;
    }

    const summary = summaryResult?.result || summaryResult;

    const sessionStats = summary.session_stats || {};
    const recommendations = summary.recommendations || {};

    const buildSkillAverages = () => {
        const skillAverages = {};
        const skills = summary.skill_analysis || {};
        for (const [name, data] of Object.entries(skills)) {
            if (typeof data.average_score === "number") {
                // Mongoose Maps do not support keys with ".", so we replace with "_"
                const safeName = name.replace(/\./g, "_");
                skillAverages[safeName] = data.average_score;
            }
        }
        return skillAverages;
    };

    const calculateAccuracy = () => {
        const mcqResults = responseData.results.filter(r =>
            r.question_type === 'mcq' || r.question_type === 'multiple-choice'
        );
        const correct = mcqResults.filter(r => r.evaluation?.correct).length;
        return mcqResults.length ? (correct / mcqResults.length) * 100 : null;
    };

    // ALWAYS CREATE A NEW SESSION – DO NOT UPDATE EXISTING ONES
    user.interview_sessions.push({
        session_id: `sess_${domain}_${Date.now()}`,
        domain,
        status: "completed",
        session_type: "adaptive_recommendation",   // or derive logically; previously "first_time"
        questions: responseData.results.map(r => r.qid),
        score: sessionStats.overall_average || 0,
        accuracy: calculateAccuracy(),
        skill_analysis: {
            stronger_skills: summary.top_skills || [],
            weaker_skills: summary.weak_skills || [],
            skill_averages: buildSkillAverages()
        },
        message: recommendations.next_steps
            ? `Analysis complete. Suggested: ${recommendations.suggested_difficulty}. Next: ${recommendations.next_steps.slice(0, 2).join(". ")}`
            : `Session completed. Score: ${(sessionStats.overall_average || 0).toFixed(1)}/100`,
        startedAt: new Date(),
        completedAt: new Date(),
        detailed_summary: summary
    });
    console.log("")

    await user.save();
    console.log(`✅ Added new session for user ${userId}`);
};

const submitTestController = async (req, res) => {
    try {
        console.log("\n================= submitTestController START =================");

        const files = req.files || {};
        console.log("[STEP-1] Raw form keys received:");
        logFormKeys(req.body, files);

        let submissionData;

        // ----------------------------
        // STEP 2: Extract submissionData
        // ----------------------------
        console.log("[STEP-2] Checking if req.body.data exists (FormData JSON)...");
        if (req.body.data) {
            console.log("[STEP-2] Body contains 'data' → parsing JSON");
            submissionData =
                typeof req.body.data === "string"
                    ? JSON.parse(req.body.data)
                    : req.body.data;
        } else {
            console.log("[STEP-2] No 'data' field → using raw body");
            submissionData = req.body;
        }

        console.log("[STEP-2] submissionData =", submissionData);

        const {
            domain,
            answers = {},
            totalQuestions,
            answeredCount,
            hintsUsed,
            completedAt,
            timeRemaining
        } = submissionData;

        // ----------------------------
        // STEP 3: Merge answers from req.body
        // ----------------------------
        console.log("[STEP-3] Merging answers from direct body keys...");
        Object.keys(req.body).forEach(key => {
            if (key.startsWith("answers[")) {
                const qid = normalizeQid(key);
                if (!answers[qid]) {
                    answers[qid] = req.body[key];
                }
            }
        });

        console.log("[STEP-3] Final merged answers =", answers);

        if (!domain) {
            return res.status(400).json({
                type: "error",
                message: "Domain is required"
            });
        }

        // ----------------------------
        // STEP 4: Load domain questions
        // ----------------------------
        console.log(`[STEP-4] Loading questions for domain: ${domain}`);
        const domainQuestions = getQuestionsForDomain(domain);
        console.log(`[STEP-4] Loaded ${Object.keys(domainQuestions).length} questions`);

        // ----------------------------
        // STEP 5: Merge QIDs
        // ----------------------------
        console.log("[STEP-5] Merging QIDs from answers & files...");
        const allQids = new Set([
            ...Object.keys(answers).map(normalizeQid),
            ...Object.keys(files).map(normalizeQid)
        ]);

        console.log("[STEP-5] Final QIDs:", [...allQids]);

        // ----------------------------
        // FILE FINDER
        // ----------------------------
        const findFile = (qid) => {
            const directKey = `answers[${qid}]`;
            if (files[directKey]) return files[directKey];
            if (files[qid]) return files[qid];

            const foundKey = Object.keys(files).find(k => normalizeQid(k) === qid);
            return foundKey ? files[foundKey] : null;
        };

        // ============================================
        // STEP 6: PROCESS QUESTIONS IN PARALLEL
        // ============================================
        console.log("\n================= PROCESSING QUESTIONS (PARALLEL) =================");

        const evaluationPromises = [...allQids].map(async (rawQid) => {
            const qid = normalizeQid(rawQid);
            const question = domainQuestions[qid];

            if (!question) {
                console.log(`⚠ Skipping unknown QID: ${qid}`);
                return null;
            }

            const qType = question.question_type;
            const userAnswer = answers[qid] || "";
            let evaluationResult;

            try {
                // ---------- MCQ ----------
                if (qType === "mcq" || qType === "multiple-choice") {
                    evaluationResult = evaluateMCQ(question, userAnswer);
                }

                // ---------- SUBJECTIVE / CODING ----------
                // else if (qType === "subjective" || qType === "coding") {
                //     // Skip empty answers to save time
                //     if (!userAnswer || String(userAnswer).trim().length === 0) {
                //         evaluationResult = { skipped: true, reason: "no answer provided" };
                //     } else {
                //         const textEval = await cachedTextEval(question, userAnswer, 1000000);
                //         evaluationResult = textEval;
                //     }
                // }
                // ---------- SUBJECTIVE ----------
                else if (qType === "subjective") {
                    if (!userAnswer || String(userAnswer).trim().length === 0) {
                        evaluationResult = { skipped: true, reason: "no answer provided" };
                    } else {
                        evaluationResult = await cachedTextEval(
                            question,
                            userAnswer,
                            1000000
                        );
                    }
                }

                // ---------- CODING ----------
                else if (qType === "coding") {
                    if (!userAnswer || String(userAnswer).trim().length === 0) {
                        evaluationResult = { skipped: true, reason: "no code submitted" };
                    } else {
                        evaluationResult = await limitedExecutePythonModel(
                            MODEL_CONFIGS.code_evaluation,
                            "evaluate_coding",
                            {
                                question,
                                user_code: userAnswer
                            },
                            120000
                        );
                    }
                }

                // ---------- VOICE ----------
                else if (qType === "voice") {
                    const fileObj = findFile(qid);
                    if (!fileObj?.tempFilePath) {
                        throw new Error("Audio file missing");
                    }

                    const absoluteAudioPath = path.resolve(fileObj.tempFilePath);

                    const voiceEval = await limitedExecutePythonModel(
                        MODEL_CONFIGS.voice_analysis,
                        "evaluate_voice",
                        { file_path: absoluteAudioPath },
                        120000
                    );

                    const transcript =
                        voiceEval?.result?.transcription?.transcript ||
                        voiceEval?.transcription?.transcript;

                    let textEval = null;
                    if (transcript) {
                        textEval = await cachedTextEval(question, transcript, 90000);
                    }

                    evaluationResult = {
                        voice_analysis: voiceEval?.result || voiceEval,
                        text_evaluation: textEval
                    };
                }

                if (qType === "coding") {
                    console.log(`[QID ${qid}] Coding Result:`, JSON.stringify(evaluationResult, null, 2));
                }

                // Extract nested result if present (common pattern from pythonConnector)
                const actualResult = evaluationResult?.result || evaluationResult;

                return {
                    qid,
                    question_type: qType,
                    status: (actualResult?.error || actualResult?.status === "error")
                        ? "error"
                        : "evaluated",
                    score: actualResult?.score || 0,
                    grade: actualResult?.grade || "F",
                    evaluation: evaluationResult, // Keep full original raw data
                    feedback: actualResult?.detailed_feedback || actualResult?.feedback || null,
                    tests: actualResult?.test_results || [],
                    error: actualResult?.error || actualResult?.message || actualResult?.reason || null
                };

            } catch (err) {
                console.error(`[QID ${qid}] Evaluation error:`, err.message);
                return {
                    qid,
                    question_type: qType,
                    status: "error",
                    reason: err.message
                };
            }
        });

        const results = (await Promise.all(evaluationPromises)).filter(Boolean);

        // ----------------------------
        // FINAL RESPONSE
        // ----------------------------
        const responseData = {
            type: "success",
            message: "Test submitted successfully",
            domain,
            results,
            metadata: {
                totalQuestions,
                answeredCount,
                hintsUsed,
                completedAt,
                timeRemaining
            }
        };

        res.status(200).json(responseData);

        // ----------------------------
        // BACKGROUND SUMMARY
        // ----------------------------
        setImmediate(async () => {
            try {
                const userId = req.user?.id;
                if (!userId) return;

                const payload = {
                    user_id: userId,
                    domain,
                    evaluation_json: responseData,
                    questions_dir: domainsDir
                };

                const summaryResult = await executePythonModel(
                    MODEL_CONFIGS.build_summary,
                    "process_user_summary",
                    payload,
                    30000
                );

                await updateUserInterviewSession(userId, domain, summaryResult, responseData);
            } catch (err) {
                console.error("[BG] Summary error:", err);
            }
        });

    } catch (err) {
        console.error("Submit error:", err);
        if (!res.headersSent) {
            res.status(500).json({
                type: "error",
                message: "Server failed to submit test",
                error: err.message
            });
        }
    }
};



const getSessiondata = async (req, res) => {
    try {
        const userId = req.user.id;
        if (!userId) throw new Error('Authentication middleware error | user id not found in the request')

        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({
                success: false,
                message: "User not found"
            });
        }

        const sessions = user.interview_sessions
            .sort((a, b) => new Date(b.startedAt) - new Date(a.startedAt)); // Sort newest first

        return res.json({
            success: true,
            count: sessions.length,
            sessions: sessions
        });

    } catch (err) {
        // console.error("Get session data error:", err);
        return res.status(500).json({
            success: false,
            message: "Failed to get sessions",
            error: err.message,
        });
    }
}



module.exports = {
    startInterview,
    submitTestController,
    getSessiondata
}
