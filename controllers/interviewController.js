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
        const domain = req.params.domain;
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
        console.error("Interview start error:", error);
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



// const submitTestController = async (req, res) => {
//     try {
//         console.log("SubmitTest hit. Body keys:", Object.keys(req.body));
//         console.log("SubmitTest hit. Files keys:", req.files ? Object.keys(req.files) : "No files");

//         let submissionData;
//         // Handle different request formats (multipart with 'data' string, or direct JSON)
//         if (req.body.data) {
//             submissionData = typeof req.body.data === 'string' ? JSON.parse(req.body.data) : req.body.data;
//         } else if (req.body.answers) {
//             submissionData = req.body;
//         } else {
//             throw new Error("Missing submission data (body.data or direct body)");
//         }
//         const { domain, totalQuestions, answeredCount, hintsUsed, completedAt, timeRemaining } = submissionData;

//         if (!domain) {
//             return res.status(400).json({
//                 type: "error",
//                 message: "Domain is required in submission data"
//             });
//         }
//         // Initialize answers from submissionData or empty object
//         const answers = submissionData.answers || {};

//         // 1. Merge "answers[qid]" fields from req.body into answers object
//         // (This happens when FormData appends individual fields)
//         Object.keys(req.body).forEach(key => {
//             const match = key.match(/^answers\[(.+)\]$/);
//             if (match) {
//                 answers[match[1]] = req.body[key];
//             }
//         });

//         // 2. Helper to extract pure QID from file keys like "answers[qid]"
//         const extractQid = (key) => {
//             const match = key.match(/^answers\[(.+)\]$/);
//             return match ? match[1] : key;
//         };

//         const domainQuestions = getQuestionsForDomain(domain);
//         const files = req.files || {};
//         const results = [];

//         // 3. Merge keys from answers and uploaded files using clean QIDs
//         const fileQids = Object.keys(files).map(key => extractQid(key));
//         const allQids = new Set([...Object.keys(answers), ...fileQids]);

//         for (const qid of allQids) {
//             const userAnswer = answers[qid] || ""; // Valid answer or empty string if only file

//             // Use helper to look up question from loaded domain questions
//             const question = domainQuestions[qid];

//             if (!question) {
//                 // If it's a file for a non-existent question or irrelevant key, we can skip or log
//                 // results.push({ qid, status: "error", reason: "Question not found in domain" });
//                 continue;
//             }

//             let evaluationResult;
//             const qType = question.question_type;

//             // -----------------------------------------
//             // MCQ – normal JS evaluation
//             // -----------------------------------------
//             if (qType === "mcq" || qType === "multiple-choice") {
//                 evaluationResult = evaluateMCQ(question, userAnswer);
//             }

//             // -----------------------------------------
//             // SUBJECTIVE & CODING – call Python text evaluator
//             // -----------------------------------------
//             else if (qType === "subjective" || qType === "coding") {
//                 const payload = {
//                     question: question,
//                     user_answer: userAnswer
//                 };

//                 evaluationResult = await executePythonModel(
//                     MODEL_CONFIGS.text_eval,
//                     "evaluate_text",
//                     payload,
//                     45000
//                 );
//             }

//             // -----------------------------------------
//             // VOICE – Comprehensive Voice Analysis + Text Evaluation
//             // -----------------------------------------
//             else if (qType === "voice") {
//                 // Check uploaded files first, then fallback to local path in answer (for testing)
//                 let audioFilePath = null;
//                 let tempFilePath = null;

//                 // Helper to find file by trying common FormData key variations
//                 const findFile = (targetId) => {
//                     if (!files) return null;
//                     // 1. Direct match
//                     if (files[targetId]) return files[targetId];
//                     // 2. Array notation match (common in FormData appending)
//                     if (files[`answers[${targetId}]`]) return files[`answers[${targetId}]`];
//                     // 3. Fallback: Search for key containing the ID (if unique enough)
//                     const key = Object.keys(files).find(k => k.includes(targetId));
//                     if (key) return files[key];
//                     return null;
//                 };

//                 const fileObj = findFile(qid);

//                 if (fileObj) {
//                     audioFilePath = fileObj.tempFilePath;
//                 } else if (typeof userAnswer === 'string' && (userAnswer.includes('/') || userAnswer.includes('\\')) && userAnswer.endsWith('.wav')) {
//                     // Fallback to local path if provided (testing only)
//                     audioFilePath = userAnswer;
//                 }

//                 if (!audioFilePath) {
//                     console.error(`[SubmitTest] Audio file missing for QID: ${qid}`);
//                     console.error(`[SubmitTest] Received file keys: ${Object.keys(files).join(', ')}`);
//                     results.push({
//                         qid,
//                         status: "error",
//                         reason: `Audio file missing. Received keys: [${Object.keys(files).join(', ')}]`
//                     });
//                     continue;
//                 }

//                 const voicePayload = {
//                     file_path: audioFilePath
//                 };

//                 // 1. Voice Analysis
//                 const voiceResult = await executePythonModel(
//                     MODEL_CONFIGS.voice_analysis,
//                     "evaluate_voice",
//                     voicePayload,
//                     120000
//                 );

//                 // 2. Text Evaluation (if transcript exists)
//                 let textResult = null;
//                 const transcript = voiceResult.result?.transcription?.transcript || voiceResult.transcription?.transcript;

//                 if (transcript && transcript.length > 0 && voiceResult.result?.transcription?.confidence !== "none") {
//                     const textPayload = {
//                         question: question,
//                         user_answer: transcript
//                     };
//                     textResult = await executePythonModel(
//                         MODEL_CONFIGS.text_eval,
//                         "evaluate_text",
//                         textPayload,
//                         45000
//                     );
//                 }

//                 // 3. Merge Results (handle wrapper if needed)
//                 evaluationResult = {
//                     voice_analysis: voiceResult.result || voiceResult,
//                     text_evaluation: textResult ? (textResult.result || textResult) : null
//                 };
//             }

//             // Final result aggregation
//             results.push({
//                 qid,
//                 question_type: qType,
//                 evaluation: evaluationResult,
//             });
//         }

//         const responseData = {
//             type: "success",
//             domain: domain,
//             message: "Test submitted successfully",
//             results,
//             metadata: {
//                 totalQuestions,
//                 answeredCount,
//                 hintsUsed,
//                 completedAt,
//                 timeRemaining,
//             },
//         };

//         // 1. Send response to frontend immediately
//         res.status(200).json(responseData);

//         // 2. Background Process: Generate Summary and Store in DB
//         // Using setImmediate to ensure it runs after the response tick
//         // Background Process: Generate Summary and Store in DB
//         setImmediate(async () => {
//             try {
//                 // Ensure we have user ID (middleware should provide this)
//                 const userId = req.user ? req.user.id : null;
//                 if (!userId) {
//                     console.error("User ID missing for background summary generation");
//                     return;
//                 }

//                 console.log(`[Background] Generating summary for user ${userId}, domain ${domain}`);

//                 const payload = {
//                     user_id: userId,
//                     domain: domain,
//                     evaluation_json: responseData,
//                     questions_dir: domainsDir
//                 };

//                 const summaryResult = await executePythonModel(
//                     MODEL_CONFIGS.build_summary,
//                     "process_user_summary",
//                     payload,
//                     30000 // 30s timeout
//                 );

//                 if (summaryResult.error) {
//                     console.error("Summary script returned error:", summaryResult.error);
//                     return;
//                 }

//                 console.log("✅ Summary result:", JSON.stringify(summaryResult, null, 2));

//                 // Extract the actual result from the wrapper
//                 const summary = summaryResult.success ? summaryResult.result : summaryResult;

//                 // 3. Update User in DB
//                 const user = await User.findById(userId);
//                 if (!user) {
//                     console.error(`[Background] User ${userId} not found`);
//                     return;
//                 }

//                 // Find the active/latest session for this domain
//                 const sessionIndex = user.interview_sessions.map(s => s.domain).lastIndexOf(domain);

//                 if (sessionIndex !== -1) {
//                     const session = user.interview_sessions[sessionIndex];

//                     // Update session status and completion
//                     session.status = "completed";
//                     session.completedAt = new Date();

//                     // Update score - use overall_average from session_stats
//                     session.score = summary.session_stats?.overall_average || 0;

//                     // Calculate accuracy if MCQ data available
//                     if (summary.session_stats?.mcq_attempted > 0) {
//                         const mcqResults = responseData.results.filter(r =>
//                             r.question_type === 'mcq' || r.question_type === 'multiple-choice'
//                         );
//                         const correctCount = mcqResults.filter(r =>
//                             r.evaluation?.correct === true
//                         ).length;
//                         session.accuracy = mcqResults.length > 0
//                             ? (correctCount / mcqResults.length) * 100
//                             : null;
//                     }

//                     // Build skill_averages Map from Python skill_analysis
//                     const skillAverages = {};
//                     const pySkillDetails = summary.skill_analysis || {};

//                     for (const [skillName, details] of Object.entries(pySkillDetails)) {
//                         if (details && typeof details.average_score === 'number') {
//                             skillAverages[skillName] = details.average_score;
//                         }
//                     }

//                     // Update skill_analysis following the schema
//                     session.skill_analysis = {
//                         stronger_skills: summary.top_skills || [],
//                         weaker_skills: summary.weak_skills || [],
//                         skill_averages: skillAverages
//                     };

//                     // Update message with recommendations
//                     if (summary.recommendations?.next_steps?.length > 0) {
//                         const nextSteps = summary.recommendations.next_steps.slice(0, 2).join('. ');
//                         session.message = `Analysis complete. Suggested: ${summary.recommendations.suggested_difficulty}. Next: ${nextSteps}`;
//                     } else {
//                         session.message = `Session completed. Score: ${session.score.toFixed(1)}/100`;
//                     }

//                     // Store additional metadata in a new field (optional enhancement)
//                     // You can add this to your schema if needed:
//                     // detailed_summary: { type: Schema.Types.Mixed }
//                     // Store additional metadata
//                     session.detailed_summary = {
//                         session_stats: summary.session_stats,
//                         recommendations: summary.recommendations,
//                         voice_insights: summary.voice_insights,
//                         detailed_feedback: summary.detailed_feedback,
//                         skill_analysis: summary.skill_analysis // Save rich analysis
//                     };

//                     await user.save();
//                     console.log(`✅ [Background] User ${userId} session updated successfully`);
//                     console.log(`   - Score: ${session.score.toFixed(1)}/100`);
//                     console.log(`   - Stronger Skills: ${session.skill_analysis.stronger_skills.join(', ')}`);
//                     console.log(`   - Weaker Skills: ${session.skill_analysis.weaker_skills.join(', ')}`);

//                 } else {
//                     console.warn(`[Background] No active session found for user ${userId} in domain ${domain}`);
//                     console.log(`[Background] Creating new completed session...`);

//                     // Build skill_averages Map
//                     const skillAverages = {};
//                     const pySkillDetails = summary.skill_analysis || {};

//                     for (const [skillName, details] of Object.entries(pySkillDetails)) {
//                         if (details && typeof details.average_score === 'number') {
//                             skillAverages[skillName] = details.average_score;
//                         }
//                     }

//                     // Calculate accuracy for MCQ questions
//                     let accuracy = null;
//                     if (summary.session_stats?.mcq_attempted > 0) {
//                         const mcqResults = responseData.results.filter(r =>
//                             r.question_type === 'mcq' || r.question_type === 'multiple-choice'
//                         );
//                         const correctCount = mcqResults.filter(r =>
//                             r.evaluation?.correct === true
//                         ).length;
//                         accuracy = mcqResults.length > 0
//                             ? (correctCount / mcqResults.length) * 100
//                             : null;
//                     }

//                     // Create new session
//                     const newSession = {
//                         session_id: `sess_${domain}_${Date.now()}`,
//                         domain: domain,
//                         session_type: "first_time", // or determine based on previous sessions
//                         session_number: user.interview_sessions.filter(s => s.domain === domain).length + 1,
//                         message: summary.recommendations?.next_steps?.length > 0
//                             ? `Analysis complete. Suggested: ${summary.recommendations.suggested_difficulty}. Next: ${summary.recommendations.next_steps.slice(0, 2).join('. ')}`
//                             : `Session completed. Score: ${(summary.session_stats?.overall_average || 0).toFixed(1)}/100`,
//                         skill_analysis: {
//                             stronger_skills: summary.top_skills || [],
//                             weaker_skills: summary.weak_skills || [],
//                             skill_averages: skillAverages
//                         },
//                         questions: responseData.results ? responseData.results.map(r => r.qid) : [],
//                         score: summary.session_stats?.overall_average || 0,
//                         accuracy: accuracy,
//                         status: "completed",
//                         startedAt: responseData.metadata?.completedAt
//                             ? new Date(responseData.metadata.completedAt)
//                             : new Date(),
//                         completedAt: new Date()
//                     };

//                     // Optional: Add detailed_summary
//                     newSession.detailed_summary = {
//                         session_stats: summary.session_stats,
//                         recommendations: summary.recommendations,
//                         voice_insights: summary.voice_insights,
//                         detailed_feedback: summary.detailed_feedback,
//                         skill_analysis: summary.skill_analysis // Save rich analysis
//                     };

//                     user.interview_sessions.push(newSession);
//                     await user.save();

//                     console.log(`✅ [Background] New session created for user ${userId}`);
//                     console.log(`   - Score: ${newSession.score.toFixed(1)}/100`);
//                     console.log(`   - Stronger Skills: ${newSession.skill_analysis.stronger_skills.join(', ')}`);
//                     console.log(`   - Weaker Skills: ${newSession.skill_analysis.weaker_skills.join(', ')}`);
//                 }

//             } catch (err) {
//                 console.error("❌ [Background] Summary generation/DB update failed:", err);
//                 console.error(err.stack);
//             }
//         });

//     } catch (err) {
//         console.error(err);
//         // If header already sent, don't send again
//         if (!res.headersSent) {
//             return res.status(500).json({
//                 type: "error",
//                 message: "Failed to submit test",
//                 error: err.message,
//             });
//         }
//     }
// };

// --- Utility: Normalize keys like "answers[q_ai_0034]" → "q_ai_0034"


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

const updateUserInterviewSession = async (userId, domain, summaryResult, responseData) => {
    const user = await User.findById(userId);
    if (!user) {
        console.error(`[Background] User ${userId} not found`);
        return;
    }

    const summary = summaryResult?.result || summaryResult;

    // Update existing or create new session
    let sessionIndex = user.interview_sessions.map(s => s.domain).lastIndexOf(domain);

    const sessionStats = summary.session_stats || {};
    const recommendations = summary.recommendations || {};

    const buildSkillAverages = () => {
        const skillAverages = {};
        const skills = summary.skill_analysis || {};
        for (const [name, data] of Object.entries(skills)) {
            if (typeof data.average_score === "number") {
                skillAverages[name] = data.average_score;
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

    if (sessionIndex !== -1) {
        // Update existing session
        const session = user.interview_sessions[sessionIndex];
        session.status = "completed";
        session.completedAt = new Date();
        session.score = sessionStats.overall_average || 0;
        session.accuracy = calculateAccuracy();

        session.skill_analysis = {
            stronger_skills: summary.top_skills || [],
            weaker_skills: summary.weak_skills || [],
            skill_averages: buildSkillAverages()
        };

        session.message = recommendations.next_steps
            ? `Analysis complete. Suggested: ${recommendations.suggested_difficulty}. Next: ${recommendations.next_steps.slice(0, 2).join(". ")}`
            : `Session completed. Score: ${session.score.toFixed(1)}/100`;

        session.detailed_summary = summary;

    } else {
        // Create new session
        user.interview_sessions.push({
            session_id: `sess_${domain}_${Date.now()}`,
            domain,
            status: "completed",
            session_type: "first_time",
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
    }

    await user.save();
    console.log(`✅ Updated session for user ${userId}`);
};



const submitTestController = async (req, res) => {
    try {
        const files = req.files || {};
        logFormKeys(req.body, files);

        let submissionData;

        // Handle FormData (stringified JSON) OR raw JSON
        if (req.body.data) {
            submissionData =
                typeof req.body.data === "string"
                    ? JSON.parse(req.body.data)
                    : req.body.data;
        } else {
            submissionData = req.body;
        }

        const {
            domain,
            answers = {},
            totalQuestions,
            answeredCount,
            hintsUsed,
            completedAt,
            timeRemaining
        } = submissionData;

        // --- MERGE answers from req.body (for FormData) ---
        // If "answers[qid]" keys exist in body, add them to our answers object
        Object.keys(req.body).forEach(key => {
            if (key.startsWith("answers[")) {
                const qid = normalizeQid(key);
                // Don't overwrite if likely already there, unless needed
                if (!answers[qid]) {
                    answers[qid] = req.body[key];
                }
            }
        });

        if (!domain) {
            return res.status(400).json({
                type: "error",
                message: "Domain is required"
            });
        }

        const domainQuestions = getQuestionsForDomain(domain);

        // Merge answer keys + file keys
        const allQids = new Set([
            ...Object.keys(answers).map(normalizeQid),
            ...Object.keys(files).map(normalizeQid)
        ]);

        const results = [];

        // --- FIXED: File finder that works for all naming cases ---
        const findFile = (qid) => {
            if (!files) return null;

            const directKey = `answers[${qid}]`;
            if (files[directKey]) return files[directKey];

            if (files[qid]) return files[qid];

            const foundKey = Object.keys(files).find(
                (k) => normalizeQid(k) === qid
            );

            return foundKey ? files[foundKey] : null;
        };

        // ============================================
        // PROCESS EACH QUESTION
        // ============================================
        for (const rawQid of allQids) {
            const qid = normalizeQid(rawQid);

            const question = domainQuestions[qid];
            if (!question) {
                console.log(`⚠ Skipping unknown QID: ${qid}`);
                continue;
            }

            const userAnswer = answers[qid] || "";
            const qType = question.question_type;

            let evaluationResult = null;

            // -------------- MCQ ----------------
            if (qType === "mcq" || qType === "multiple-choice") {
                evaluationResult = evaluateMCQ(question, userAnswer);
            }

            // -------------- SUBJECTIVE / CODING ----------------
            else if (qType === "subjective" || qType === "coding") {
                evaluationResult = await executePythonModel(
                    MODEL_CONFIGS.text_eval,
                    "evaluate_text",
                    {
                        question,
                        user_answer: userAnswer
                    },
                    45000
                );
            }

            // -------------- VOICE ----------------
            else if (qType === "voice") {
                let audioFilePath = null;
                const fileObj = findFile(qid);

                if (fileObj) {
                    audioFilePath = fileObj.tempFilePath;
                }

                if (!audioFilePath) {
                    console.error(`[SubmitTest] Audio file missing for QID: ${qid}`);
                    console.error(`[SubmitTest] Received file keys: ${Object.keys(files).join(', ')}`);
                    results.push({
                        qid,
                        question_type: qType,
                        status: "error",
                        reason: `Audio file missing. Received keys: [${Object.keys(files).join(', ')}]`
                    });
                    continue;
                }

                // Resolve to absolute path to avoid Python CWD issues
                const absoluteAudioPath = path.resolve(audioFilePath);
                console.log(`[SubmitTest] Processing audio at: ${absoluteAudioPath}`);

                const voiceResult = await executePythonModel(
                    MODEL_CONFIGS.voice_analysis,
                    "evaluate_voice",
                    { file_path: absoluteAudioPath },
                    120000
                );

                // If transcript exists → run text evaluation
                let textResult = null;
                const transcript =
                    voiceResult?.result?.transcription?.transcript ||
                    voiceResult?.transcription?.transcript;

                if (transcript) {
                    textResult = await executePythonModel(
                        MODEL_CONFIGS.text_eval,
                        "evaluate_text",
                        {
                            question,
                            user_answer: transcript
                        },
                        45000
                    );
                }

                evaluationResult = {
                    voice_analysis: voiceResult.result || voiceResult,
                    text_evaluation: textResult ? textResult.result || textResult : null
                };
            }

            // Push evaluation result
            results.push({
                qid,
                question_type: qType,
                evaluation: evaluationResult
            });
        }

        // ============= FINAL RESPONSE TO FRONTEND ============
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

        // =======================================================
        // BACKGROUND SUMMARY GENERATION (unchanged)
        // =======================================================
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
                console.error("Background process error:", err);
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
        const { domain } = req.body;

        if (!domain) {
            return res.status(400).json({
                success: false,
                message: "Domain is required"
            });
        }

        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({
                success: false,
                message: "User not found"
            });
        }

        const sessions = user.interview_sessions
            .filter(s => s.domain === domain)
            .sort((a, b) => new Date(b.startedAt) - new Date(a.startedAt)); // Sort newest first

        return res.json({
            success: true,
            count: sessions.length,
            sessions: sessions
        });

    } catch (err) {
        console.error("Get session data error:", err);
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
