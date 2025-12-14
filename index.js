// const express = require('express');
// const bodyParser = require('body-parser');
// const cors = require('cors');
// const app = express();
// const dotenv = require('dotenv');
// const dbConnect = require('./utils/dbConnect');
// const authRoutes = require('./routes/authRoute');
// const setupRoutes = require('./routes/setupRoute');
// const interviewRoutes = require('./routes/interview');
// const dashBoardRoutes = require('./routes/dashboard');
// const gamificationRoutes = require('./routes/gamification')
// const PORT = process.env.PORT || 3000;


// app.use(cors({
//     origin: "http://localhost:5173",
//     credentials: true,
//     methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
//     allowedHeaders: ["Content-Type", "Authorization"]
// }));
// dotenv.config();
// dbConnect();

// // --- FIX: Move fileUpload BEFORE bodyParser ---
// const fileUpload = require('express-fileupload');
// app.use(fileUpload({
//     useTempFiles: true,
//     tempFileDir: './tmp/',
//     createParentPath: true,
//     limits: { fileSize: 10 * 1024 * 1024 }
// }));

// // Now parse JSON/urlencoded
// app.use(bodyParser.json({ limit: '10mb' }));
// app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));

// app.use(express.static('public'));

// // routes
// app.use('/api/auth', authRoutes);
// app.use('/api/setup', setupRoutes);
// app.use('/api/interview', interviewRoutes);
// app.use('/api/dashboard', dashBoardRoutes);
// app.use("/api/gamification", gamificationRoutes);

// app.listen(PORT, () => {
//     console.log(`Server is running on port ${PORT}`);
// });

const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const dotenv = require('dotenv');
const fileUpload = require('express-fileupload');

const dbConnect = require('./utils/dbConnect');
const authRoutes = require('./routes/authRoute');
const setupRoutes = require('./routes/setupRoute');
const interviewRoutes = require('./routes/interview');
const dashBoardRoutes = require('./routes/dashboard');
const gamificationRoutes = require('./routes/gamification');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

/* =======================
   CORS (MUST BE FIRST)
======================= */
// const corsOptions = {
//    origin: "*",
//    credentials: true,
//    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
//    allowedHeaders: ["Content-Type", "Authorization"]
// };

app.use(cors());

/* =======================
   Database
======================= */
dbConnect();

/* =======================
   File Upload (before body parsing)
======================= */
app.use(fileUpload({
    useTempFiles: true,
    tempFileDir: './tmp/',
    createParentPath: true,
    limits: { fileSize: 10 * 1024 * 1024 }
}));

/* =======================
   Body Parsing
======================= */
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));

/* =======================
   Static Files
======================= */
app.use(express.static('public'));

/* =======================
   Routes
======================= */
app.use('/api/auth', authRoutes);
app.use('/api/setup', setupRoutes);
app.use('/api/interview', interviewRoutes);
app.use('/api/dashboard', dashBoardRoutes);
app.use('/api/gamification', gamificationRoutes);

/* =======================
   Server
======================= */
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

