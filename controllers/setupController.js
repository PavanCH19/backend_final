const pdf = require("pdf-parse");
const fs = require("fs");
const path = require("path");

const processResume = async (pdfFile) => {
    const {}
  try {
    if (!pdfFile) {
      return { status: 400, message: "No PDF file uploaded" };
    }

    // Save PDF temporarily
    const uploadPath = path.join(__dirname, "../uploads", pdfFile.name);
    await pdfFile.mv(uploadPath);

    // Read PDF into buffer
    const dataBuffer = fs.readFileSync(uploadPath);

    // Parse PDF
    const data = await pdf(dataBuffer);

    // Extract email using regex
    const emailMatch = data.text.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}/);
    const email = emailMatch ? emailMatch[0] : null;

    // Delete the file after processing
    fs.unlinkSync(uploadPath);

    return {
      status: 200,
      message: "PDF processed successfully",
      text: data.text,
      email: email
    };
  } catch (error) {
    console.error("Error processing PDF:", error);
    return { status: 500, message: "Error processing PDF", error: error.message };
  }
};

module.exports = {
  processResume
};
