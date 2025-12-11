const express = require("express");
const router = express.Router();
const fetchUser = require("../middleware/fetchUser");
const { startInterview, submitTestController,
    getSessiondata
} = require("../controllers/interviewController");
router.get('/:domain', fetchUser, startInterview);

router.post("/submit-test", fetchUser, submitTestController);

router.post('/recent-sessions', fetchUser, getSessiondata);

module.exports = router;
