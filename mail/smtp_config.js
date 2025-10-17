// config/smtpConfig.js
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');
dotenv.config(); // Load environment variables from .env file

// Create secure transporter
const transporter = nodemailer.createTransport({
    host: 'smtp.gmail.com',
    port: 465,
    secure: true,
    auth: {
        user: process.env.SMTP_EMAIL,      // Your Gmail ID
        pass: process.env.SMTP_PASSWORD  // Your Gmail App Password
    }
});

// Optional: Verify connection
transporter.verify(function (error, success) {
    if (error) {
        console.error('❌ SMTP connection error:', error);
    } else {
        console.log('✅ SMTP server is ready to send emails');
    }
});

module.exports = transporter;
