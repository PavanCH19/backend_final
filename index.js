const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const app = express();
const dotenv = require('dotenv');
const dbConnect = require('./utils/dbConnect');
const authRoutes = require('./routes/authRoute');
const setupRoutes = require('./routes/setupRoute');
const interviewRoutes = require('./routes/interview');
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));
dotenv.config();
dbConnect();
//routes
app.use('/api/auth', authRoutes);
app.use('/api/setup', setupRoutes);
app.use('/api/interview', interviewRoutes);

// ============================================
// MODEL CONFIGURATIONS
// ============================================



const { set } = require('./mail/smtp_config');

// ============================================
// RESUME CLASSIFIER ENDPOINTS
// ============================================




// ============================================
// START SERVER
// ============================================

app.listen(PORT, () => {
    console.log(`\n=== Server Started ===`);
    console.log(`Server: http://localhost:${PORT}`);
    console.log(`=====================\n`);
});