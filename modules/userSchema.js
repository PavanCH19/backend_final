const mongoose = require('mongoose');
const { Schema } = mongoose;

const SkillAnalysisSchema = new Schema({
    stronger_skills: [{ type: String }],
    weaker_skills: [{ type: String }],
    skill_averages: {
        type: Map,        // { "Python": 95, "ML": 80 }
        of: Number
    }
}, { _id: false });

const InterviewSessionSchema = new Schema({
    session_id: { type: String },  // optional unique session identifier
    domain: { type: String, required: true },

    session_type: {
        type: String,
        enum: [
            "first_time",
            "adaptive_recommendation",
            "mixed_level",
            "resume_session"
        ],
        default: "first_time"
    },

    session_number: { type: Number, default: 1 }, // 1, 2, 3...

    // Session messages or contextual info
    message: { type: String },

    // Skill analysis object
    skill_analysis: SkillAnalysisSchema,

    // Question IDs asked during this session
    questions: [{ type: String }],

    // Session performance
    score: { type: Number, default: null },
    accuracy: { type: Number, default: null },

    // Session lifecycle
    startedAt: { type: Date, default: Date.now },
    completedAt: { type: Date, default: null },

    status: {
        type: String,
        enum: ["ongoing", "completed"],
        default: "ongoing"
    }
});

const UserSchema = new Schema({

    // Basic auth fields
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    date: { type: Date, default: Date.now },
    firstLogin : {
        type : Boolean,
        default : true
    },

    // Profile
    profile: {
        name: String,
        phone: String,
        location: String,
    },

    // Skills
    skills: [String],

    // Education
    education: [
        {
            id: Number,
            degree: String,
            college: String,
            startYear: String,
            endYear: String
        }
    ],

    // Experience
    experience: [
        {
            id: Number,
            role: String,
            company: String,
            startDate: String,
            endDate: String,
            current: Boolean
        }
    ],

    // Projects
    projects: [
        {
            id: Number,
            title: String,
            description: String,
            technologies: [String],
            startDate: String,
            endDate: String,
            link: String,
            github: String
        }
    ],

    // Targets
    target_domains: [String],
    target_companies: [String],

    // 🔥 INTERVIEW SESSIONS — FULL SESSION HISTORY PER DOMAIN
    interview_sessions: [InterviewSessionSchema]

});

module.exports = mongoose.model('User', UserSchema);
