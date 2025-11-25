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


module.exports = {
    startInterview
}