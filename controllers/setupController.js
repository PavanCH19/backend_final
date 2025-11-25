const { PDFParse } = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const User = require("../modules/userSchema");
const { executePythonModel } = require('../utils/pythonConnector');

// FIXED: Correct path - only go up 1 level to 'dl' folder
const fine_tune_path = path.join(__dirname, '../python_models/fine_tune/extractdata_resume.py');
const resume_classifier_path = path.join(__dirname, '../python_models/setup/src/prediction.py');
const mock_question_path = path.join(__dirname, "../python_models/question_recomendation/mock_question.py")

const MODEL_CONFIGS = {
  resume_classifier: {
    scriptPath: resume_classifier_path,
    pythonPath: 'python',
    envVars: {}
  },
  fine_tune: {
    scriptPath: fine_tune_path,
    pythonPath: 'python',
    envVars: {}
  },
  mock_domain_question_recomendataion: {
    scriptPath: mock_question_path,
    pythonPath: 'python',
    envVars: {}
  }
};

// ============================================
// EXTRACT TEXT FROM PDF
// ============================================
const extract_data = async (pdfFile) => {
  const uploadDir = path.join(__dirname, "../uploads");
  if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

  const uploadPath = path.join(uploadDir, pdfFile.name);
  await pdfFile.mv(uploadPath);

  const dataBuffer = fs.readFileSync(uploadPath);
  const parser = new PDFParse({ data: dataBuffer });

  const textResult = await parser.getText();

  fs.unlinkSync(uploadPath);
  await parser.destroy();

  return textResult;
};

// ============================================
// UPDATE USER PROFILE
// ============================================
const update_user_profile = async (userId, extractedData) => {
  try {
    if (!userId || !extractedData?.personal_info) {
      return {
        success: false,
        status: 400,
        message: "Invalid userId or extracted data.",
        data: null
      };
    }

    const { personal_info, known_skills } = extractedData;

    const user = await User.findById(userId);
    if (!user) {
      return {
        success: false,
        status: 404,
        message: "User not found.",
        data: null
      };
    }

    user.profile.name = personal_info.name || user.profile.name;
    user.profile.phone = personal_info.phone || user.profile.phone;

    const existingSkills = user.skills || [];
    user.skills = Array.from(new Set([...existingSkills, ...known_skills]));

    await user.save();

    return {
      success: true,
      status: 200,
      message: "User profile updated successfully.",
      data: {
        email: user.email,
        profile: user.profile,
        skills: user.skills
      }
    };
  } catch (error) {
    console.error("Error updating user profile:", error);
    return {
      success: false,
      status: 500,
      message: "Error updating user profile.",
      error: error.message,
      data: null
    };
  }
};

// ============================================
// PROCESS RESUME
// ============================================
const processResume = async (pdfFile, userId) => {
  try {
    if (!pdfFile) {
      return {
        success: false,
        status: 400,
        message: "No PDF file uploaded",
        data: null,
        fullText: null
      };
    }

    const textResult = await extract_data(pdfFile);

    const data = await executePythonModel(
      MODEL_CONFIGS.fine_tune,
      'extract_data_resume',
      textResult.text,
      60000
    );

    const updateResult = await update_user_profile(userId, data.result);

    return {
      success: updateResult.success,
      status: updateResult.status,
      message: updateResult.message,
      data: updateResult.data,
      fullText: textResult.text
    };
  } catch (error) {
    console.error("Error processing PDF:", error);
    return {
      success: false,
      status: 500,
      message: "Error processing PDF",
      error: error.message,
      data: null,
      fullText: null
    };
  }
};

// ============================================
// CLASSIFY RESUME
// ============================================
const classifyResume = async (resumeData) => {
  try {
    if (!resumeData || Object.keys(resumeData).length === 0) {
      return {
        success: false,
        status: 400,
        message: 'No resume data provided',
        data: null
      };
    }

    const result = await executePythonModel(
      MODEL_CONFIGS.resume_classifier,
      'main',
      resumeData,
      60000
    );

    return {
      success: true,
      status: 200,
      message: "Resume classified successfully",
      data: result
    };
  } catch (error) {
    console.error('Resume classification error:', error);
    return {
      success: false,
      status: 500,
      message: 'Resume classification error',
      error: error.message,
      data: null
    };
  }
};


const mock_domain_questions = async (userData) => {
  try {
    if (!userData || Object.keys(userData).length === 0) {
      return {
        success: false,
        status: 400,
        message: 'No user data provided',
        data: null
      };
    }

    const result = await executePythonModel(
      MODEL_CONFIGS.mock_domain_question_recomendataion,
      'main',
      userData,   // ✅ CORRECT
      60000
    );

    return {
      success: true,
      status: 200,
      message: "Mock questions generated successfully",
      data: result
    };
  } catch (error) {
    console.error('Mock question error:', error);
    return {
      success: false,
      status: 500,
      message: 'Mock question generation error',
      error: error.message,
      data: null
    };
  }
};


module.exports = { processResume, classifyResume, mock_domain_questions };
