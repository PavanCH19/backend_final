// const express = require('express');
// const bodyParser = require('body-parser');
// const cors = require('cors');
// const app = express();
// const dotenv = require('dotenv');
// const dbConnect = require('./utils/dbConnect');
// const authRoutes = require('./routes/authRoute');
// const setupRoutes = require('./routes/setupRoute');
// const interviewRoutes = require('./routes/interview');
// const PORT = process.env.PORT || 3000;

// app.use(cors());
// dotenv.config();
// dbConnect();

// // Parse JSON / urlencoded
// app.use(bodyParser.json({ limit: '10mb' }));
// app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));

// // --- File Upload Middleware ---
// const fileUpload = require('express-fileupload');
// app.use(fileUpload({
//     useTempFiles: true,
//     tempFileDir: './tmp/',
//     createParentPath: true,
//     limits: { fileSize: 10 * 1024 * 1024 }
// }));

// app.use(express.static('public'));

// // routes
// app.use('/api/auth', authRoutes);
// app.use('/api/setup', setupRoutes);
// app.use('/api/interview', interviewRoutes);

// app.listen(PORT, () => {
//     console.log(`Server is running on port ${PORT}`);
// });




const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const app = express();
const dotenv = require('dotenv');
const dbConnect = require('./utils/dbConnect');
const authRoutes = require('./routes/authRoute');
const setupRoutes = require('./routes/setupRoute');
const interviewRoutes = require('./routes/interview');
const dashBoardRoutes = require('./routes/dashboard');
const gamificationRoutes = require('./routes/gamification')
const PORT = process.env.PORT || 3000;

app.use(cors());
dotenv.config();
dbConnect();

// --- FIX: Move fileUpload BEFORE bodyParser ---
const fileUpload = require('express-fileupload');
app.use(fileUpload({
    useTempFiles: true,
    tempFileDir: './tmp/',
    createParentPath: true,
    limits: { fileSize: 10 * 1024 * 1024 }
}));

// Now parse JSON/urlencoded
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));

app.use(express.static('public'));

// routes
app.use('/api/auth', authRoutes);
app.use('/api/setup', setupRoutes);
app.use('/api/interview', interviewRoutes);
app.use('/api/dashboard', dashBoardRoutes);
app.use("/api/gamification", gamificationRoutes);

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
