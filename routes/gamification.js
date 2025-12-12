// routes/gamificationAdvanced.js
const express = require("express");
const router = express.Router();
const { getAdvancedGamification } = require("../controllers/gamificationController");
const fetchUser = require("../middleware/fetchUser");

// GET calculated gamification (reads DB, computes, returns — no DB writes)
router.get("/advanced", fetchUser, getAdvancedGamification);

module.exports = router;