const { body, validationResult } = require("express-validator");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../modules/userSchema");
const dotenv = require("dotenv");

dotenv.config(); // Load environment variables

// =======================================
// Controller: Create a New User
// =======================================
const createUser = async (user) => {
    const { email, password } = user;
    try {
        // Check if user already exists
        const existingUser = await User.findOne({ email: email });
        if (existingUser) {
            return { status: 400, msg: "User already exists" };
        }

        // Hash the password
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        // Create a new user
        const newUser = await User.create({
            email: email,
            password: hashedPassword,
        });

        // Generate JWT token
        //const token = jwt.sign({ id: newUser._id }, process.env.JWT_SECRET, { expiresIn: "1h" });

        return { status: 201, msg: "User created successfully", user: newUser };
    } catch (error) {
        console.error("Error creating user:", error);
        return { status: 500, msg: "Server error" };
    }
}


// =======================================
// Controller: User Login
// =======================================
const loginUser = async (data) => {
    const { email, password } = data;
    try {
        // Check if user exists
        const user = await User.findOne({ email: email });
        if (!user) {
            return { status: 400, msg: "User not found" };
        }

        // Compare passwords
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return { status: 400, msg: "Invalid password" };
        }

        // Generate JWT token
        const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: "1h" });

        return { status: 200, msg: "User logged in successfully", user, token };
    } catch (error) {
        console.error("Error logging in:", error);
        return { status: 500, msg: "Server error" };
    }
}

// =======================================
// Controller: Get User Details
// =======================================
const getUserDetails = async (data) => {
    const { email } = data;
    try {
        const user = await User.findOne({ email }).select("-password");

        if (!user) {
            return { status: 404, msg: "User not found" };
        }

        const userDetails = {
            profile: {
                name: user.name,
                email: user.email,
                phone: user.phone,
                location: user.location,
            },
            skills: user.skills,
            education: user.education,
            experience: user.experience,
        };

        return { status: 200, user: userDetails };
    } catch (error) {
        console.error("Error fetching user:", error);
        return { status: 500, msg: "Server error" };
    }
};


// =====================================
// Controller: Update User Details
// =====================================
const updateUserDetails = async (data) => {
    const { email, ...updateFields } = data;
    try {
        // Get user from DB
        const user = await User.findById(data.user.id);
        if (!user) {
            return { status: 404, msg: "User not found" };
        }

        // Compare password
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return { status: 401, msg: "Invalid password" };
        }

        // Update user fields (excluding password)
        const updatedUser = await User.findByIdAndUpdate(
            req.user.id,
            { $set: updateFields },
            { new: true, runValidators: true }
        ).select("-password");

        return { status: 200, msg: "User updated successfully", user: updatedUser };
    } catch (error) {
        console.error("Error updating user:", error);
        return { status: 500, msg: "Server error" };
    }
};

module.exports = {
    createUser,
    loginUser,
    getUserDetails,
    updateUserDetails
};