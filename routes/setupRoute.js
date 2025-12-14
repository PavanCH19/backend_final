const express = require("express");
const router = express.Router();
const { processResume, classifyResume, mock_domain_questions } = require("../controllers/setupController");
const fetchUser = require("../middleware/fetchUser");
const User = require('../modules/userSchema')
const mongoose = require("mongoose");

// Note: fileUpload middleware is already configured globally in index.js

router.post('/upload-resume', fetchUser, async (req, res) => {
    try {
        // Check if file was uploaded
        if (!req.files || !req.files.resumeFile) {
            return res.status(400).json({
                success: false,
                status: 400,
                message: "No file uploaded. Please select a resume file.",
            });
        }

        const pdfFile = req.files.resumeFile;

        // Validate file type
        if (pdfFile.mimetype !== 'application/pdf') {
            return res.status(400).json({
                success: false,
                status: 400,
                message: "Invalid file type. Please upload a PDF file.",
            });
        }

        console.log("❤️❤️❤️❤️", req.user);

        // Process the resume (controller already handles user update)
        const result = await processResume(pdfFile, req.user.id);

        // Return the controller response
        return res.status(result.status).json(result);
    } catch (error) {
        console.error("Resume upload error:", error);

        // Handle busboy/multipart errors specifically
        if (error.message && error.message.includes("Unexpected end of form")) {
            return res.status(400).json({
                success: false,
                status: 400,
                message: "File upload incomplete. Please try again.",
                error: "The file upload was interrupted or incomplete.",
            });
        }

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


/**
 * Saves alternative suggested domains into the user's resume_session.
 * If no resume_session exists, a new one will be created.
 */
/**
 * Saves alternative suggested domains at USER ROOT LEVEL.
 * Also ensures a resume_session exists and marks it completed.
 */
const saveAlternativeDomains = async (userId, suggestions, domain) => {
    try {
        const user = await User.findById(userId);
        // console.log("😎User before updating the setup results",user )
        if (!user) {
            console.error("User not found while saving suggested domains.");
            return null;
        }

        // console.log("Saving suggestions:", suggestions);

        /* -----------------------------------------------------
           1. Save suggestions at ROOT LEVEL (REQUIRED)
        ----------------------------------------------------- */
        user.alternative_suggested_domains = suggestions || [];

        /* -----------------------------------------------------
           2. Maintain resume_session for tracking purposes
        ----------------------------------------------------- */
        let session = user.interview_sessions.find(
            (s) => s.session_type === "resume_session"
        );

        if (!session) {
            session = {
                session_id: new mongoose.Types.ObjectId().toString(),
                domain: domain || "General",
                session_type: "resume_session",
                session_number: (user.interview_sessions?.length || 0) + 1,
                status: "completed",
                startedAt: new Date(),
                completedAt: new Date()
            };

            user.interview_sessions.push(session);
        } else {
            session.completedAt = new Date();
            session.status = "completed";
        }

        /* -----------------------------------------------------
           3. Save user document
        ----------------------------------------------------- */
        await user.save();
        const saved = await User.findById(userId)
        // console.log("Saved user updated target domains?", saved);

        return session;

    } catch (err) {
        console.error("Error saving alternative domain suggestions:", err);
        return null;
    }
};



router.post('/setUp_result', fetchUser, async (req, res) => {
    try {
        let user = await User.findById(req.user.id)
        // console.log(" 😘 User when /setup-result hit : ", user)
        const cleanResume = {
            id: req.user.id,
            skills: req.body.skills || [],
            projects: req.body.projects || [],
            work_experience: req.body.work_experience || [],
            test_score: req.body.test_score || 0,
            preferred_domain: req.body.preferred_domain
        };

        cleanResume.domain = cleanResume.preferred_domain;

        // checking the user if target domain is updated?
        user = await User.findById(req.user.id)
        console.log(" 😘 User before sending the data to python : ", user)

        // console.log("CLEAN RESUME SENT TO PYTHON:", cleanResume);

        // Send to Python classifier
        const classification = await classifyResume(cleanResume);
        console.log("CLASSIFIER RESULT:", classification);

        const resultData = classification?.data?.result;

        if (!resultData) {
            return res.status(500).json({
                success: false,
                message: "Classifier returned no result"
            });
        }

        // Extract the suggestions from model output
        const suggestions = resultData.alternative_domain_suggestions || [];

        // Save to interview session
        await saveAlternativeDomains(
            req.user.id,
            suggestions,
            cleanResume.preferred_domain
        );

        // Mark setup as completed
        const updatedUser = await User.findByIdAndUpdate(
            req.user.id,
            { setupCompleted: true },
            { new: true }
        );

        res.status(200).json({
            success: true,
            data: classification,
            user : updatedUser
        });

    } catch (error) {
        console.error("❌ Resume classification error:", error);
        res.status(500).json({ success: false, error: error.message });
    }
});


// router.post('/setUp_result', fetchUser, async (req, res) => {
//     try {
//         console.log("========================", req.body);
//         const cleanResume = {
//             id: req.user.id,
//             skills: req.body.skills || [],
//             projects: req.body.projects || [],
//             work_experience: req.body.work_experience || [],
//             test_score: req.body.test_score || 0,
//             preferred_domain: req.body.preferred_domain
//             //preferred_domain: normalizeDomain(req.body.preferred_domain || "")
//         };

//         // Also set domain for Python
//         cleanResume.domain = cleanResume.preferred_domain;

//         console.log("CLEAN RESUME SENT TO PYTHON:", cleanResume);

//         // sending the data to the model to classify user to 'fit' or 'not-fit'
//         const result = await classifyResume(cleanResume);
//         console.log("CLASSIFIER RESULT:", result);

//         // update user as he completes the setup | update setupCompleted : true
//         let updatedUser = await User.findByIdAndUpdate(req.user.id, { setupCompleted: true }, { new: true })
//         console.log("updatedUser : ", updatedUser.setupCompleted)

//         res.status(200).json({ success: true, data: result });

//     } catch (error) {
//         console.error("❌ Resume classification error:", error);
//         res.status(500).json({ success: false, error: error.message });
//     }
// });


router.get('/alternative-domain-suggestions', fetchUser, async(req,res)=>{
    try {
        const user = await User.findById(req.user.id).select('alternative_suggested_domains');

        if (!user) {
            return res.status(404).json({
                success: false,
                message: "User not found."
            });
        }

        return res.status(200).json({
            success: true,
            alternative_suggested_domains: user.alternative_suggested_domains
        });

    } catch (error) {
        console.error("❌ Error fetching alternative domain suggestions:", error);
        return res.status(500).json({
            success: false,
            message: "Internal server error.",
            error: error.message
        });
    }
})

router.put('/add-target-domain', fetchUser, async(req, res)=>{
    try{
        const userId = req.user.id;
        const {domain} = req.body;        
        const user = await User.findById(userId)
        const found = user.target_domains.filter((dom)=>{
            return dom === domain;
        })
        if(found.length>0) {
            res.status(400).json({  
                message : "domain already exists. chooose another"
            })
        }
        const updated_user = await User.findByIdAndUpdate(
            userId,
            { $push: { target_domains: domain } },
            { new: true }
          )

        res.status(200).json(updated_user)
    }
    catch(err){
        res.status(500).json({
            message : err.message || "Internal server Error",
            error : err
        })
    }

})





module.exports = router;