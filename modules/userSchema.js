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

// NEW: Detailed Summary Schema for storing comprehensive analysis
const DetailedSummarySchema = new Schema({
    session_stats: {
        total_questions: Number,
        questions_answered: Number,
        mcq_attempted: Number,
        subjective_attempted: Number,
        voice_attempted: Number,
        coding_attempted: Number,
        overall_average: Number,
        grade_distribution: Schema.Types.Mixed
    },
    recommendations: {
        focus_skills: [String],
        strong_skills: [String],
        suggested_difficulty: String,
        areas_to_improve: [String],
        next_steps: [String]
    },
    voice_insights: {
        confidence_score: Number,
        clarity_score: Number,
        speech_ratio: Number,
        words_per_minute: Number,
        recommendations: [String]
    },
    detailed_feedback: {
        strengths: [String],
        areas_needing_work: [String]
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
            "resume_session",
            "completed_session"  // Added for fallback cases
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

    // 🔥 NEW: Store complete detailed summary from Python analysis
    detailed_summary: DetailedSummarySchema,

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
    setupCompleted: {
        type: Boolean,
        default: false
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