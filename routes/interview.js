// const express = require("express");
// const router = express.Router();
// const fetchUser = require("../middleware/fetchUser");
// const { startInterview, submitTestController,
//     getSessiondata
// } = require("../controllers/interviewController");
// router.get('/:domain', fetchUser, startInterview);

// router.post("/submit-test", fetchUser, submitTestController);

// router.post('/recent-sessions', fetchUser, getSessiondata);

// module.exports = router;



const express = require("express");
const router = express.Router();
const fetchUser = require("../middleware/fetchUser");

const { startInterview, submitTestController, getSessiondata } =
    require("../controllers/interviewController");

const { processResume } = require("../controllers/setupController");

// ===============================
// 🚀 Route to Upload Resume (Fixes Busboy Issue)
// ===============================
router.post("/upload-resume", fetchUser, async (req, res) => {
    console.log("🔥 USER FROM TOKEN:", req.user.id);

    try {
        console.log("REQ FILES:", req.files);

        if (!req.files || !req.files.pdfFile) {
            return res.status(400).json({ error: "No PDF uploaded" });
        }

        const result = await processResume(req.files.pdfFile, req.user.id);
        res.status(result.status).json(result);

    } catch (err) {
        console.error("Resume Upload Error:", err);
        res.status(500).json({ error: err.message });
    }
});

// ===============================
// Other API Routes
// ===============================
router.get('/:domain', fetchUser, startInterview);
router.post("/submit-test", fetchUser, submitTestController);
router.post('/recent-sessions', fetchUser, getSessiondata);

module.exports = router;
