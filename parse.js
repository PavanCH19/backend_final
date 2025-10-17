const fs = require("fs");
const { PDFParse } = require("pdf-parse");
const resumePath = "C:\\Users\\pavan\\Desktop\\resume\\software_engineering.pdf";

const dataBuffer = fs.readFileSync(resumePath);
const parser = new PDFParse({ data: dataBuffer });

parser.getText().then((result) => {
  console.log("Full Text Content:\n");
  console.log(result.text);

  // Example: extract email using regex
  const emailMatch = result.text.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}/);
  if (emailMatch) {
    console.log("\nExtracted Email:", emailMatch[0]);
  }
}).catch(err => {
  console.error("Error parsing PDF:", err);
}).finally(async () => {
  await parser.destroy();
});