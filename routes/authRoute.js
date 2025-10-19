const express = require('express');
const router = express.Router();
const fetchUser = require('../middleware/fetchUser'); // Middleware to fetch user from token
const { createUser, loginUser, getUserDetails, updateUserDetails } = require('../controllers/authController')
const { requestPasswordReset, verifyAndReset } = require('../controllers/passResetController');

router.post('/createUser', async (req, res) => {
    const data = req.body;
    const result = await createUser(data);
    res.status(result.status).json(result);
});

router.post('/loginUser', async (req, res) => {
    const data = req.body;
    const result = await loginUser(data);
    res.status(result.status).json(result)
});

router.get('/getUserDetails', async (req, res) =>{
    const data = req.body;
    const result = await getUserDetails(data);
    res.status(result.status).json(result)
});

router.post('/updateUserDetails', fetchUser, async (req,res) => {
    const data = req.body;
    const result = await updateUserDetails(data);
    res.status(result.status).json(result)
});

router.post('/password-reset-request', async (req, res) => {
    const data = req.body;
    console.log(data);
    const result = await requestPasswordReset(data);
    res.status(result.status).json(result)
});

router.post("/verify-and-reset", verifyAndReset);

module.exports = router;