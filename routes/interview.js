const express = require("express");
const router = express.Router();
const fetchUser = require("../middleware/fetchUser");
const { startInterview, submitTestController } = require("../controllers/interviewController");
const fileUpload = require("express-fileupload");


router.get('/:domain', fetchUser, startInterview);

router.use(
    fileUpload({
        useTempFiles: true,
        tempFileDir: "/tmp/",
    })
);

router.post("/submit-test", submitTestController);

module.exports = router;
