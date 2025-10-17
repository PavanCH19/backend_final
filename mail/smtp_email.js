// routes/email.js
const express = require('express');
const router = express.Router();
const transporter = require('./smtp_config.js'); // Import the transporter from smtp_config.js

// Route to send email
const sendEmail = async (to, subject, text, html) => {

    if (!to || !subject || !text) {
        return res.status(400).json({ message: 'To, subject, and text are required.' });
    }

    const mailOptions = {
        from: '"Pavan" <pavandvh27@gmail.com>',
        to,
        subject,
        text,
        html
    };

    try {
        const info = await transporter.sendMail(mailOptions);
    } catch (error) {
        console.error('❌ Error sending email:', error);
    }
};

module.exports = sendEmail;
