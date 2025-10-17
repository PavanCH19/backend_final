const express = require('express');
const router = express.Router();
const fetchUser = require('../middleware/fetchUser'); // Middleware to fetch user from token
const { createUser, loginUser, getUserDetails, updateUserDetails } = require('../controllers/authController')
const { requestPasswordReset, verifyOTP, resetPassword } = require('../controllers/passResetController');

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

router.post('/verify-otp', async (req, res) => {
    const data = req.body;
    const result = await verifyOTP(data);
    res.status(result.status).json(result);
});

router.post('/reset-password', async (req, res) => {
    const data = req.body;
    const result = await resetPassword(data);
    res.status(result.status).json(result);
});

module.exports = router;
