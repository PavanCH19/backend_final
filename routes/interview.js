const express = require("express");
const router = express.Router();
const fetchUser = require("../middleware/fetchUser");
const { startInterview, submitTestController,
    getSessiondata
} = require("../controllers/interviewController");
const fileUpload = require("express-fileupload");


router.get('/:domain', fetchUser, startInterview);

router.use(
    fileUpload({
        useTempFiles: true,
        tempFileDir: "/tmp/",
    })
);

router.post("/submit-test", fetchUser, submitTestController);

router.post('/recent-sessions', fetchUser, getSessiondata);

module.exports = router;
