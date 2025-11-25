const express = require("express");
const router = express.Router();
const fetchUser = require("../middleware/fetchUser");
const { startInterview } = require("../controllers/interviewController");

router.get('/:domain', fetchUser, startInterview);

module.exports = router;
