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
    }
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

            user.interview_sessions.push({
                session_id: `sess_${Date.now()}`,
                domain,
                session_type: "adaptive_recommendation",
                session_number: domainSessions.length + 1,
                message: response.message,
                skill_analysis: response.skill_analysis || {},
                questions: response.questions.map((q) => q._id),
                startedAt: new Date(),
                status: "ongoing"
            });

            await user.save();
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



const submitTestController = async (req, res) => {
    try {
        let submissionData;
        // Handle different request formats (multipart with 'data' string, or direct JSON)
        if (req.body.data) {
            submissionData = typeof req.body.data === 'string' ? JSON.parse(req.body.data) : req.body.data;
        } else if (req.body.answers) {
            submissionData = req.body;
        } else {
            throw new Error("Missing submission data (body.data or direct body)");
        }

        const { domain, answers, totalQuestions, answeredCount, hintsUsed, completedAt, timeRemaining } = submissionData;

        if (!domain) {
            return res.status(400).json({
                type: "error",
                message: "Domain is required in submission data"
            });
        }

        const domainQuestions = getQuestionsForDomain(domain);
        const files = req.files || {};
        const results = [];

        for (const [qid, userAnswer] of Object.entries(answers)) {

            // Use helper to look up question from loaded domain questions
            const question = domainQuestions[qid];

            if (!question) {
                results.push({ qid, status: "error", reason: "Question not found in domain" });
                continue;
            }

            let evaluationResult;

            // -----------------------------------------
            // MCQ – normal JS evaluation
            // -----------------------------------------
            // Handle both keys just in case
            if (question.question_type === "mcq" || question.question_type === "multiple-choice") {
                evaluationResult = evaluateMCQ(question, userAnswer);
            }

            // -----------------------------------------
            // SUBJECTIVE & CODING – call Python text evaluator
            // -----------------------------------------
            if (question.question_type === "subjective" || question.question_type === "coding") {
                const payload = {
                    question: question,
                    user_answer: userAnswer
                };

                evaluationResult = await executePythonModel(
                    MODEL_CONFIGS.text_eval,
                    "evaluate_text",
                    payload,
                    45000
                );
            }

            // -----------------------------------------
            // VOICE – Comprehensive Voice Analysis + Text Evaluation
            // -----------------------------------------
            if (question.question_type === "voice") {
                // Check uploaded files first, then fallback to local path in answer (for testing)
                let tempFilePath = null;

                if (files[qid]) {
                    tempFilePath = files[qid].tempFilePath;
                } else if (typeof userAnswer === 'string' && (userAnswer.includes('/') || userAnswer.includes('\\')) && userAnswer.endsWith('.wav')) {
                    tempFilePath = userAnswer;
                }

                if (!tempFilePath) {
                    results.push({ qid, status: "error", reason: "Audio file missing" });
                    continue;
                }

                const voicePayload = {
                    file_path: tempFilePath
                };

                // 1. Voice Analysis
                const voiceResult = await executePythonModel(
                    MODEL_CONFIGS.voice_analysis,
                    "evaluate_voice",
                    voicePayload,
                    120000
                );

                // 2. Text Evaluation (if transcript exists)
                let textResult = null;
                const transcript = voiceResult.transcription?.transcript;

                if (transcript && transcript.length > 0 && voiceResult.transcription.confidence !== "none") {
                    const textPayload = {
                        question: question,
                        user_answer: transcript
                    };
                    textResult = await executePythonModel(
                        MODEL_CONFIGS.text_eval,
                        "evaluate_text",
                        textPayload,
                        45000
                    );
                }

                // 3. Merge Results
                evaluationResult = {
                    voice_analysis: voiceResult,
                    text_evaluation: textResult
                };
            }

            // Final result aggregation
            results.push({
                qid,
                question_type: question.question_type,
                evaluation: evaluationResult,
            });
        }

        return res.status(200).json({
            type: "success",
            domain: domain,
            message: "Test submitted successfully",
            results,
            metadata: {
                totalQuestions,
                answeredCount,
                hintsUsed,
                completedAt,
                timeRemaining,
            },
        });

    } catch (err) {
        console.error(err);
        return res.status(500).json({
            type: "error",
            message: "Failed to submit test",
            error: err.message,
        });
    }
};



module.exports = {
    startInterview,
    submitTestController
}