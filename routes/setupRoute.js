const express = require("express");
const router = express.Router();
const fileUpload = require("express-fileupload");
const { processResume, classifyResume, mock_domain_questions } = require("../controllers/setupController");
const fetchUser = require("../middleware/fetchUser");
const User = require('../modules/userSchema')

// Middleware for file upload
router.use(fileUpload());

router.post('/upload-resume', fetchUser, async (req, res) => {
    try {
        const pdfFile = req.files?.resumeFile;

        // Process the resume (controller already handles user update)
        const result = await processResume(pdfFile, req.user.id);

        // Return the controller response
        return res.status(result.status).json(result);
    } catch (error) {
        return res.status(500).json({
            success: false,
            status: 500,
            message: "Internal server error.",
            error: error.message,
        });
    }
});



router.post('/mock_session_questions', fetchUser, async (req, res) => {
    try {
        let { skills, target_domains } = req.body;

        // Normalize target_domains (convert space → underscore)
        if (Array.isArray(target_domains)) {
            target_domains = target_domains.map(d =>
                d.trim().replace(/\s+/g, "_").toLowerCase()
            );
        }

        const payload = {
            _id: req.user.id,
            skills,
            target_domain: target_domains[0]
        };

        console.log("📤 Final Payload to Python:", payload);

        const result = await mock_domain_questions(payload);

        // ⭐ PRINT FULL NESTED RESULT
        console.log("🟢 Full Python Result:");
        console.dir(result, { depth: null, colors: true });

        return res.status(200).json({ success: true, data: result });

    } catch (error) {
        console.error("❌ MOCK SESSION API ERROR:", error);
        res.status(500).json({ success: false, error: error.message });
    }
});


const DOMAIN_ALIASES = {
    "ai ml": "ai_ml",
    "ai/ml": "ai_ml",
    "aiml": "ai_ml",
    "machine learning": "ai_ml",
    "ml ai": "ai_ml",

    "data science": "data_science",
    "datascience": "data_science",

    "web development": "web_development",
    "web dev": "web_development",
    "frontend": "web_development",
    "backend": "web_development",
    "fullstack": "web_development",

    "mobile development": "mobile_development",
    "mobile dev": "mobile_development",
    "android": "mobile_development",
    "ios": "mobile_development",

    "devops": "devops",
    "sre": "devops",

    "cyber security": "cybersecurity",
    "cybersecurity": "cybersecurity",
    "security": "cybersecurity",

    "cloud": "cloud_computing",
    "cloud computing": "cloud_computing",
    "aws": "cloud_computing",
    "azure": "cloud_computing",
    "gcp": "cloud_computing",

    "blockchain": "blockchain",
    "web3": "blockchain",

    "game development": "game_development",
    "game dev": "game_development",
    "gaming": "game_development",

    "embedded": "embedded_systems",
    "embedded systems": "embedded_systems",
    "iot": "embedded_systems",

    "ar vr": "ar_vr",
    "ar": "ar_vr",
    "vr": "ar_vr",

    "ui ux": "ui_ux_design",
    "ui/ux": "ui_ux_design",
    "design": "ui_ux_design"
};


function normalizeDomain(input) {
    if (!input) return null;

    const key = input.trim().toLowerCase().replace(/[^a-z0-9]/g, ' ');

    // direct match
    if (DOMAIN_ALIASES[key]) return DOMAIN_ALIASES[key];

    // snake_case fallback (ai ml → ai_ml)
    const fallback = key.replace(/\s+/g, '_');

    return fallback;
}




router.post('/setUp_result', fetchUser, async (req, res) => {
    try {
        console.log("========================", req.body);
        const cleanResume = {
            id: req.user.id,
            skills: req.body.skills || [],
            projects: req.body.projects || [],
            work_experience: req.body.work_experience || [],
            test_score: req.body.test_score || 0,
            preferred_domain: req.body.preferred_domain
            //preferred_domain: normalizeDomain(req.body.preferred_domain || "")
        };

        // Also set domain for Python
        cleanResume.domain = cleanResume.preferred_domain;

        console.log("CLEAN RESUME SENT TO PYTHON:", cleanResume);

        // sending the data to the model to classify user to 'fit' or 'not-fit'
        const result = await classifyResume(cleanResume);
        console.log("CLASSIFIER RESULT:", result);

        // update user as he completes the setup | update setupCompleted : true
        let updatedUser = await User.findByIdAndUpdate(req.user.id, {setupCompleted : true }, { new : true })
        console.log("updatedUser : ", updatedUser.setupCompleted)

        res.status(200).json({ success: true, data: result });

    } catch (error) {
        console.error("❌ Resume classification error:", error);
        res.status(500).json({ success: false, error: error.message });
    }
});




module.exports = router;
