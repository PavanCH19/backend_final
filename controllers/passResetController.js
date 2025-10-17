const sendEmail = require('../mail/smtp_email');
const { generateOTP, generateJWTToken } = require('../otp_JWT/jwtOtpGeneration');
const jwt = require('jsonwebtoken');
const bcrypt = require("bcryptjs");
const env = require('dotenv');
env.config();
const User = require('../modules/user'); // Adjust the path to your User schema


// In-memory storage for OTP (use Redis or DB in production)
let otpStore = {};
let resetTokenStore = {};

// Password Reset Request Endpoint
const requestPasswordReset = async (data) => {
    let { email, resetLink } = data;

    // Generate OTP
    const otp = generateOTP();

    otpStore[email] = otp;
    console.log(otpStore[email])

    // Generate JWT Token for password reset link
    const resetToken = generateJWTToken(email);
    resetTokenStore[email] = resetToken;

    console.log('Reset Token:', resetToken);

    resetLink = resetLink + resetToken;
    // Send email with OTP and reset link

    const htmlContent = `
    <html>
        <head>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f4f7fc;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    width: 70%;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 40px;
                    background-color: #ffffff;
                    border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                h1 {
                    font-size: 28px;
                    color: #2c3e50;
                    margin-bottom: 25px;
                    font-weight: 600;
                }
                p {
                    font-size: 16px;
                    color: #555;
                    line-height: 1.8;
                    margin: 10px 0;
                    text-align: justify; /* Added justification */
                }
                a {
                    font-size: 18px;
                    color: #1a73e8;
                    text-decoration: none;
                    font-weight: bold;
                    padding: 12px 25px;
                    border-radius: 6px;
                    border: 2px solid #1a73e8;
                    transition: background-color 0.3s ease, color 0.3s ease;
                }
                a:hover {
                    background-color: #1a73e8;
                    color: #ffffff;
                }
                .otp {
                    font-size: 20px;
                    font-weight: bold;
                    color:rgb(32, 32, 32);
                    margin: 20px 0;
                    text-align: justify; /* Justifying OTP text */
                }
                .footer {
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-top: 30px;
                    text-align: center; /* Ensuring footer content remains centered */
                }
                .footer a {
                    color: #7f8c8d;
                    text-decoration: none;
                }
                .footer a:hover {
                    color:rgb(243, 239, 233);
                }
                .footer p {
                    margin-bottom: 8px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Reset Your Password</h1>
                <p>Hi,</p>
                <p>Welcome to MyCloudNoteBook, the most secure and flexible way to store your notes and files in the cloud. To ensure the safety of your account, please reset your password.</p>
                <p>Click the link below to reset your password:</p>
                <p><a href="${resetLink}">Reset Password</a></p>
                <p>Alternatively, use this OTP:</p>
                <p><span class="otp">${otp.otp}</span></p>
                <p>This OTP is valid for 10 minutes.</p>
                <div class="footer">
                    <p>If you didn't request a password reset, please ignore this email.</p>
                    <p>For further assistance, visit our <a href="#">Help Center</a>.</p>
                </div>
            </div>
        </body>
    </html>
`;

    try {
        await sendEmail(email, 'Password Reset Request', 'Password Reset Instructions', htmlContent);
        return { status: 200, msg: 'Password reset email sent.' };
    } catch (error) {
        return { status: 500, msg: 'Error sending email.' };
    }
};

// OTP Verification Endpoint
const verifyOTP = (data) => {
    let { email, otp } = data;
    otp = Number(otp);
    console.log(email, otp, otpStore[email])

    if (otpStore[email]) {
        const storedOtp = otpStore[email].otp;
        const timestamp = otpStore[email].timestamp;
        console.log(storedOtp === otp, storedOtp, otp)

        // Check if OTP matches
        if (storedOtp === otp) {
            const currentTime = Date.now();
            const timeDifference = currentTime - timestamp;  // Calculate time difference
            const otpValidityDuration = 10 * 60 * 1000; // 10 minutes in milliseconds

            // Check if OTP is within the 10-minute window
            if (timeDifference <= otpValidityDuration) {
                // OTP is valid
                delete otpStore[email];  // Remove OTP after verification
                return { status: 200, msg: 'OTP verified. Proceed to reset password.' };
            } else {
                // OTP expired
                return { status: 400, msg: 'OTP expired. Please request a new one.' };
            }
        } else {
            // Invalid OTP
            return { status: 400, msg: 'Invalid OTP.' };
        }
    } else {
        return { status: 400, msg: 'OTP not found. Please request a new one.' };
    }
};


// JWT Token Verification Endpoint (for Password Reset)
const resetPassword = async (req, res) => {
    const { token, newPassword } = req.body;

    jwt.verify(token, process.env.JWT_SECRET, async (err, decoded) => {
        if (err) {
            return res.status(400).send({ msg: 'Invalid or expired token.' });
        }

        const { email } = decoded;

        if (resetTokenStore[email] !== token) {
            return res.status(400).send({ msg: 'Invalid reset request.' });
        }

        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(newPassword, salt);

        try {
            // 🔐 Update the password in MongoDB (Hash it if needed)
            const result = await User.findOneAndUpdate(
                { email },
                { $set: { password: hashedPassword } }, // You should hash this!
                { new: true }
            );

            if (!result) {
                return res.status(404).send({ msg: 'User not found.' });
            }

            delete resetTokenStore[email];

            res.status(200).send({ msg: 'Password successfully reset.' });
        } catch (error) {
            console.error(error);
            res.status(500).send({ msg: 'Error updating password in database.' });
        }
    });
};

module.exports = {
    requestPasswordReset,
    verifyOTP,
    resetPassword
}