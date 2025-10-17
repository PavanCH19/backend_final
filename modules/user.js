const mongoose = require('mongoose');

const { Schema } = mongoose;
const UserSchema = new Schema({
    // id: {
    //     type: String,
    //     default: () => new mongoose.Types.ObjectId().toString(), // Generates a unique ID as a string
    //     unique: true,
    // },
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
    }
});

module.exports = mongoose.model("User", UserSchema);