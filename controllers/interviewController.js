const path = require("path");
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


const submitTestController = async (req, res) => {
    try {
        const { answers, totalQuestions, answeredCount, hintsUsed, completedAt, timeRemaining } =
            JSON.parse(req.body.data);

        const files = req.files || {};
        const results = [];

        for (const [qid, userAnswer] of Object.entries(answers)) {

            const question = await Question.findOne({ _id: qid });
            if (!question) {
                results.push({ qid, status: "error", reason: "Question not found" });
                continue;
            }

            let evaluationResult;

            // -----------------------------------------
            // MCQ – normal JS evaluation
            // -----------------------------------------
            if (question.question_type === "mcq") {
                evaluationResult = evaluateMCQ(question, userAnswer);
            }

            // -----------------------------------------
            // SUBJECTIVE – call Python model
            // -----------------------------------------
            if (question.question_type === "subjective") {
                const payload = {
                    question: question.text,
                    expected_answer: question.expected_answer,
                    user_answer: userAnswer
                };

                evaluationResult = await executePythonModel(
                    MODEL_CONFIGS.subjective_eval,
                    "evaluate_subjective",
                    payload,
                    40000
                );
            }

            // -----------------------------------------
            // CODING – JS evaluator or judge0
            // -----------------------------------------
            if (question.question_type === "coding") {
                evaluationResult = await evaluateCoding(question, userAnswer);
            }

            // -----------------------------------------
            // AUDIO (3 types) – uses Python
            // -----------------------------------------
            if (question.question_type === "audio") {
                const audioFile = files[qid];

                if (!audioFile) {
                    results.push({ qid, status: "error", reason: "Audio file missing" });
                    continue;
                }

                const basePayload = {
                    question: question.text,
                    expected_answer: question.expected_answer,
                    file_path: audioFile.tempFilePath
                };

                if (question.audio_type === "mcq_audio") {
                    evaluationResult = await executePythonModel(
                        MODEL_CONFIGS.audio_eval,
                        "evaluate_audio_mcq",
                        basePayload,
                        60000
                    );
                }

                if (question.audio_type === "subjective_audio") {
                    evaluationResult = await executePythonModel(
                        MODEL_CONFIGS.audio_eval,
                        "evaluate_audio_subjective",
                        basePayload,
                        60000
                    );
                }

                if (question.audio_type === "coding_audio") {
                    evaluationResult = await executePythonModel(
                        MODEL_CONFIGS.audio_eval,
                        "evaluate_audio_coding",
                        basePayload,
                        60000
                    );
                }
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