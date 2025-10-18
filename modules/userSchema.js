const mongoose = require('mongoose');
const { Schema } = mongoose;

const UserSchema = new Schema({
    email: {
        type: String,
        required: true,
        unique: true
    },
    password: {
        type: String,
        required: true
    },
    date: {
        type: Date,
        default: Date.now
    },
    profile: {
        name: { type: String },
        phone: { type: String },
        location: { type: String }
    },
    skills: [
        { type: String }
    ],
    education: [
        {
            id: { type: Number },
            degree: { type: String },
            college: { type: String },
            startYear: { type: String },
            endYear: { type: String }
        }
    ],
    experience: [
        {
            id: { type: Number },
            role: { type: String },
            company: { type: String },
            startDate: { type: String },
            endDate: { type: String },
            current: { type: Boolean }
        }
    ]
});

module.exports = mongoose.model('User', UserSchema);
