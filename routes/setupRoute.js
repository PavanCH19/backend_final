const express = require("express");
const router = express.Router();
const { processResume } = require("../controllers/setupController");
const fileUpload = require("express-fileupload");

// Middleware for file upload
router.use(fileUpload());

// POST /resume upload route
router.post("/upload-resume", async (req, res) => {
    const pdfFile = req.files?.resumeFile;
    const result = await processResume(pdfFile);
    res.status(result.status).json(result);
});

module.exports = router;
