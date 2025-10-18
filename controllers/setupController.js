const { PDFParse } = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const { text } = require("body-parser");

const processResume = async (pdfFile) => {
  try {
    if (!pdfFile) return { status: 400, message: "No PDF file uploaded" };

    const uploadDir = path.join(__dirname, "../uploads");
    if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

    const uploadPath = path.join(uploadDir, pdfFile.name);
    await pdfFile.mv(uploadPath);

    const dataBuffer = fs.readFileSync(uploadPath);
    const parser = new PDFParse({ data: dataBuffer });

    // Get text content
    const textResult = await parser.getText();

    // Get metadata and document info
    const infoResult = await parser.getInfo();

    fs.unlinkSync(uploadPath);
    await parser.destroy();
    console.log("PDF Info:", textResult.text);
    return {
      status: 200,
      message: "PDF processed successfully",
      fullText: textResult.text
    };
  } catch (error) {
    console.error("Error processing PDF:", error);
    return { status: 500, message: "Error processing PDF", error: error.message };
  }
};

module.exports = { processResume };