import re
import os
from pathlib import Path
import spacy
from spacy.matcher import PhraseMatcher
from shared_config import KNOWN_SKILLS

# CRITICAL: Set working directory and use absolute paths
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

# Use absolute path for model
MODEL_DIR = SCRIPT_DIR / "skill_ner_model"


def load_skill_ner_model(model_dir=None):
    """Load trained skill NER model"""
    if model_dir is None:
        model_dir = MODEL_DIR
    
    model_dir = str(model_dir)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"❌ Model not found at {model_dir}. Please run train.py first."
        )
    
    print(f"📂 Loading model from {model_dir}...")
    nlp = spacy.load(model_dir)
    return nlp


def setup_phrase_matcher(nlp):
    """Initialize PhraseMatcher with skill patterns"""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in KNOWN_SKILLS]
    matcher.add("SKILL", patterns)
    return matcher


def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_skills(nlp, matcher, text):
    """Extract skills from text using NER and phrase matching"""
    text_clean = clean_text(text)
    doc = nlp(text_clean)

    matched_skills = set()
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        matched_skills.add(span.text.lower())

    ner_skills = set([ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"])
    all_skills = matched_skills.union(ner_skills)

    known_set = set([k.lower() for k in KNOWN_SKILLS])
    known = sorted([s for s in all_skills if s in known_set])
    unknown = sorted([s for s in all_skills if s not in known_set])
    return known, unknown


def extract_personal_info(text):
    """Extract personal information from text"""
    lines = text.splitlines()
    name = ""
    for line in lines:
        line = line.strip()
        if line and re.match(r"^[A-Za-z\s\-\.]+$", line):
            name = line
            break
    email = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone = re.search(r"(\+?\d[\d\s-]{7,}\d)", text)
    linkedin = re.search(r"https?://(www\.)?linkedin\.com/[^\s,]+", text)
    github = re.search(r"https?://(www\.)?github\.com/[^\s,]+", text)
    return {
        "name": name,
        "email": email.group(0) if email else "",
        "phone": phone.group(0) if phone else "",
        "linkedin": linkedin.group(0) if linkedin else "",
        "github": github.group(0) if github else ""
    }


def extract_data_resume(test_resume_1):
    """Main function called from Node.js"""
    try:
        nlp = load_skill_ner_model()
        matcher = setup_phrase_matcher(nlp)
        personal_info = extract_personal_info(test_resume_1)
        known_skills_found, unknown = extract_skills(nlp, matcher, test_resume_1)
        
        return {
            "success": True,
            "personal_info": personal_info,
            "known_skills": known_skills_found,
            "unknown_skills": unknown
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    test_resume_1 = """
Software Engineer | Full-Stack(MERN) | Rest APIs | AWS | AI & Data-Driven System
Pavan Chandrappa Hottigoudra
+91 7483022523 | pavandvh27@gmail.com | https://linkedin.com | GitHub

Software Engineer with experience in full-stack development, cloud-native APIs, and AI-driven applications. 
Skilled in MERN stack, RestAPI, React.js, MongoDB, and AWS (Lambda, Cognito, DynamoDB) with expertise in secure authentication, 
scalable deployments, and data-driven system design. Actively practicing DSA on LeetCode and passionate about building robust, 
efficient, and impactful software solutions.

WORK EXPERIENCE
Intern – Backend Developer
Gandeevan Technologies, Bengaluru, India (Hybrid) Jul 2025 – Present
• Built secure RESTful APIs for authentication and data management, integrating AWS Cognito for identity and access control.
• Designed scalable serverless workflows and automated deployments via CI/CD pipelines, improving reliability and release speed.
• Enhanced API security through unit testing, validation, and best practices while collaborating with cross-functional teams.
Tech Stack: Node.js, Express.js, AWS (Cognito, DynamoDB, API Gateway, Lambda), CI/CD, Serverless Architecture.

SKILLS
Java, AWS, Python, C, HTML5, CSS3, MySQL, React.js, Vite, JSX, Bootstrap, Axios, Node.js, 
Express.js, REST API, JWT, Git, GitHub, Postman, Agile/Scrum, Responsive Design, DSA, System Design Basics
    """

    print("\n" + "="*80)
    print("TEST RESUME 1")
    print("="*80)
    
    result = extract_data_resume(test_resume_1)

    if result["success"]:
        print("🧑 Personal Info:", result["personal_info"])
        print("✅ Known Skills:", result["known_skills"])
        print("❓ Unknown Skills:", result["unknown_skills"])
    else:
        print("❌ Error:", result["error"])