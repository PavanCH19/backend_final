const express = require('express')
const router = express.Router();

router.get('/', (req, res) => {
    res.send("dashboard router is running")
})

module.exports = router;