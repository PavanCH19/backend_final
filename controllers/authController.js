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
        // 1️⃣ Check if user exists
        const user = await User.findOne({ email });
        if (!user) {
            return { success: false, status: 400, msg: "User not found" };
        }

        // if(user.firstLogin){
        //     console.log("first login : ", user.firstLogin);
        //     let updatedUser = await User.findByIdAndUpdate(user._id,
        //         {
        //             $set : {firstLogin : false}
        //         },
        //         {
        //             new : true
        //         }
        //     )
        //     console.log('first login updated : ', updatedUser.firstLogin)
        // }

        // 2️⃣ Compare password
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return { success: false, status: 400, msg: "Invalid password" };
        }

        // 3️⃣ Prepare safe user payload for JWT
        const userPayload = {
            id: user._id,
            email: user.email,
            profile: user.profile || {},
            skills: user.skills || [],
            education: user.education || [],
            experience: user.experience || [],
            date: user.date,
        };

        // 4️⃣ Generate JWT token
        const token = jwt.sign(userPayload, process.env.JWT_SECRET, { expiresIn: "1h" });

        // 5️⃣ Return response (omit password)
        const userResponse = { ...userPayload };

        return {
            success: true,
            status: 200,
            msg: "User logged in successfully",
            token,
            setupCompleted: user.setupCompleted,
            user: userResponse,
        };
    } catch (error) {
        console.error("Error logging in:", error);
        return { success: false, status: 500, msg: "Server error", error: error.message };
    }
};

// =======================================
// Controller: Get User Details
// =======================================
const getUserDetails = async (email) => {
    try {
        console.log("Fetching details for:", email);

        const user = await User.findOne({ email }).select("-password");

        if (!user) {
            return {
                success: false,
                status: 404,
                message: "User not found",
                data: null,
            };
        }

        // Build clean structured response
        // Safely get skill_analysis from the last interview session
        let skillAnalysis = null;
        if (user.interview_sessions &&
            Array.isArray(user.interview_sessions) &&
            user.interview_sessions.length > 0) {
            const lastSession = user.interview_sessions[user.interview_sessions.length - 1];
            skillAnalysis = lastSession?.skill_analysis || null;
        }

        const userDetails = {
            profile: {
                name: user.profile?.name || "",
                email: user.email,
                phone: user.profile?.phone || "",
                location: user.profile?.location || "",
            },
            skills: user.skills || [],
            skill_analysis: skillAnalysis || {
                stronger_skills: [],
                weaker_skills: [],
                skill_averages: {}
            },
            education: user.education || [],
            experience: user.experience || [],
            projects: user.projects || [], // ✅ fixed typo
        };
        console.log(userDetails)
        return {
            success: true,
            status: 200,
            message: "User details fetched successfully",
            data: userDetails,
        };
        // return user;
    } catch (error) {
        console.error("Error fetching user:", error);
        return {
            success: false,
            status: 500,
            message: "Server error",
            error: error.message,
            data: null,
        };
    }
};


// =====================================
// Controller: Update User Details
// =====================================
const updateUserDetails = async (req) => {
    try {
        const userId = req.user.id;
        const updateData = req.body;
        console.log("Update data received:", updateData);
        const user = await User.findById(userId);
        if (!user) {
            return { status: 404, success: false, msg: "User not found" };
        }

        const updateFields = (target, source) => {
            for (const key in source) {
                const value = source[key];

                if (Array.isArray(value)) {
                    // Handle array updates
                    if (!Array.isArray(target[key])) target[key] = [];

                    // If array contains objects (like education, projects)
                    if (value.length > 0 && typeof value[0] === "object") {
                        const updatedArray = [];

                        value.forEach((item) => {
                            if (item.id) {
                                const existingIndex = target[key].findIndex(
                                    (t) => String(t.id) === String(item.id)
                                );
                                if (existingIndex !== -1) {
                                    // Update existing item
                                    updateFields(target[key][existingIndex], item);
                                    updatedArray.push(target[key][existingIndex]);
                                } else {
                                    // Add new item
                                    updatedArray.push(item);
                                }
                            } else {
                                updatedArray.push(item); // No id — just push
                            }
                        });

                        target[key] = updatedArray;
                    } else {
                        // Array of primitives (like skills)
                        target[key] = [...value];
                    }

                } else if (value && typeof value === "object") {
                    // Handle nested objects
                    if (!target[key]) target[key] = {};
                    updateFields(target[key], value);
                } else {
                    // Primitive values
                    target[key] = value;
                }
            }
        };
        console.log('======================================================', updateData)
        updateFields(user, updateData);

        const updatedUser = await user.save();
        return {
            status: 200,
            success: true,
            msg: "User updated successfully",
            user: updatedUser,
        };
    } catch (error) {
        console.error("Error updating user:", error);
        return { status: 500, success: false, msg: "Server error" };
    }
};


module.exports = {
    createUser,
    loginUser,
    getUserDetails,
    updateUserDetails
};