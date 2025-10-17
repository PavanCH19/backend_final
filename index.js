const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const app = express();
const dotenv = require('dotenv');
const dbConnect = require('./utils/dbConnect');
const authRoutes = require('./routes/authRoute');
const setupRoutes = require('./routes/setupRoute');
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

// ============================================
// MODEL CONFIGURATIONS
// ============================================

const MODEL_CONFIGS = {
    resume_classifier: {
        scriptPath: './python_models/setup/src/prediction.py',
        pythonPath: 'python', // or specify full path like 'C:/Python39/python.exe'
        envVars: {} // optional environment variables
    },
    another_model: {
        scriptPath: './other_project/model.py',
        pythonPath: 'python',
        envVars: {}
    }
    // Add more models here
};

const { executePythonModel } = require('./utils/pythonConnector');
const { set } = require('./mail/smtp_config');

// ============================================
// RESUME CLASSIFIER ENDPOINTS
// ============================================

app.post('/api/resume/classify', async (req, res) => {
    try {
        const resumeData = req.body;

        if (!resumeData || Object.keys(resumeData).length === 0) {
            return res.status(400).json({
                success: false,
                error: 'Resume data is required'
            });
        }

        const result = await executePythonModel(
            MODEL_CONFIGS.resume_classifier,
            'main',
            resumeData,
            60000 // 60 second timeout for model loading
        );

        res.json(result);
    } catch (error) {
        console.error('Resume classification error:', error);
        res.status(500).json(error);
    }
});


// ============================================
// START SERVER
// ============================================

app.listen(PORT, () => {
    console.log(`\n=== Server Started ===`);
    console.log(`Server: http://localhost:${PORT}`);
    console.log(`=====================\n`);
});