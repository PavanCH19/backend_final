import random
import json
import os
import re
import spacy
from spacy.training import offsets_to_biluo_tags
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from spacy.matcher import PhraseMatcher

# ============================================================================
# 1. SKILLS AND TEMPLATES
# ============================================================================

known_skills = [
    # Data Science
    "Python", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", 
    "Matplotlib", "Seaborn", "Jupyter", "Docker", "Deep Learning", 
    "Machine Learning", "Statistics", "Data Visualization", "Big Data", 
    "Spark", "Hadoop", "Keras", "XGBoost", "Feature Engineering", 
    "A/B Testing", "Time Series", "Clustering", "Regression", "Classification",
    
    # Web Development
    "JavaScript", "React", "Vue.js", "Angular", "Node.js", "Express", "HTML", "CSS", 
    "MongoDB", "PostgreSQL", "MySQL", "Redis", "GraphQL", "REST API", "TypeScript", 
    "Webpack", "Git", "Bootstrap", "Sass", "Tailwind CSS", "Next.js", "Nuxt.js", 
    "Django", "Flask", "FastAPI", "Spring Boot", "PHP", "Laravel", "Ruby on Rails", 
    "OAuth", "JWT",
    
    # Mobile
    "Java", "Kotlin", "Swift", "React Native", "Flutter", "Dart", "iOS", "Android", 
    "Xcode", "Android Studio", "Firebase", "SQLite", "Core Data", "UIKit", "SwiftUI",
    "Jetpack Compose", "Realm", "Push Notifications", "In-App Purchases", "MapKit", "Camera API",
    
    # DevOps
    "Kubernetes", "AWS", "Azure", "GCP", "Jenkins", "Terraform", "Ansible", 
    "Linux", "Bash", "CI/CD", "Monitoring", "Nagios", "Prometheus", "Grafana", 
    "ELK Stack", "GitLab CI", "GitHub Actions", "CircleCI", "Chef", "Puppet", "Helm", 
    "Vault", "Consul", "Nginx", "Apache", "Load Balancing",
    
    # Security
    "Network Security", "Penetration Testing", "CISSP", "CEH", "Firewall", "Encryption", 
    "Wireshark", "Metasploit", "Nmap", "Risk Assessment", "Compliance", "SIEM",
    "Burp Suite", "Kali Linux", "SQL Injection", "XSS", "OWASP", "Security Policies",
    "IAM", "Zero Trust", "Threat Modeling", "Forensics", "Malware Analysis",
    
    # AI/ML
    "OpenCV", "Hugging Face", "Transformers", "NLP", "Reinforcement Learning", "GANs", 
    "Speech Recognition", "BERT", "LSTM", "Attention Mechanisms", "Computer Vision", 
    "Data Augmentation",
    
    # Cloud Computing
    "Serverless", "CloudFormation", "Cloud Security", "Lambda", "DevOps Integration",
    
    # Blockchain
    "Solidity", "Ethereum", "Hyperledger", "Smart Contracts", "Web3.js", "Truffle", 
    "Ganache", "Metamask", "DeFi", "NFT", "Consensus Mechanisms", "Cryptography", 
    "Blockchain Architecture", "Rust", "Polkadot", "Solana",
    
    # Game Development
    "Unity", "Unreal Engine", "C#", "C++", "3D Modeling", "Blender", "Shader Programming", 
    "Game Physics", "AI for Games", "Animation", "VR Integration", "Level Design", 
    "Audio Design", "Optimization",
    
    # Embedded Systems
    "Microcontrollers", "ARM", "Raspberry Pi", "Arduino", "IoT", "Sensors", 
    "RTOS", "UART", "SPI", "I2C", "PCB Design", "Firmware", "MQTT", "Embedded Linux",
    
    # AR/VR
    "XR Interaction Toolkit", "Oculus SDK", "ARKit", "ARCore", "Vuforia", 
    "Motion Tracking", "Spatial Mapping", "Hand Tracking",
    
    # UI/UX
    "Figma", "Adobe XD", "Sketch", "InVision", "Wireframing", "Prototyping", "User Research",
    "Design Systems", "Typography", "Color Theory", "Accessibility", "Interaction Design",
    "User Testing", "Journey Mapping"
]

sentence_templates = [
    "Worked on {skills} to build scalable backend systems.",
    "Developed applications using {skills}.",
    "Implemented {skills} in various projects.",
    "Proficient with {skills} for modern software development.",
    "Hands-on experience with {skills}, improving performance and reliability.",
    "Led a team implementing {skills} in enterprise projects.",
    "Designed solutions using {skills} to optimize workflow.",
    "Built RESTful APIs leveraging {skills} for high-performance systems.",
    "Collaborated with cross-functional teams using {skills}.",
    "Optimized existing systems with {skills}, enhancing scalability.",
    "Created microservices architectures using {skills}.",
    "Developed machine learning models using {skills} to predict outcomes.",
    "Worked on cloud-native applications with {skills}.",
    "Integrated {skills} to automate CI/CD pipelines.",
    "Applied {skills} to improve data processing and analytics.",
    "Designed UI/UX features using {skills} frameworks.",
    "Implemented secure authentication and authorization using {skills}.",
    "Performed data visualization and reporting with {skills}.",
    "Worked on containerization and orchestration using {skills}.",
    "Developed testing and QA pipelines using {skills}.",
    "Built serverless applications using {skills} in cloud environments.",
    "Created ETL pipelines leveraging {skills}.",
    "Implemented DevOps best practices using {skills}.",
    "Designed scalable database solutions with {skills}.",
    "Developed real-time applications using {skills} for efficient performance.",
    "Enhanced system monitoring and logging using {skills}.",
    "Implemented AI and deep learning solutions using {skills}.",
    "Worked with big data technologies like {skills} for analytics projects.",
    "Participated in Agile/Scrum teams utilizing {skills} for project management.",
    "Built full-stack applications using {skills} from frontend to backend.",
    "Improved API performance and reliability using {skills}.",
    "Strong knowledge of {skills} with practical implementation in projects.",
    "Experienced in designing and deploying applications using {skills}.",
    "Technical expertise in {skills} with focus on scalable solutions.",
    "Familiar with agile methodologies and tools, using {skills} for delivery.",
    "Involved in end-to-end software development lifecycle with {skills}.",
    "Proven track record of using {skills} in production environments.",
    "Trained and mentored junior developers in {skills}.",
    "Contributed to open-source projects leveraging {skills}.",
    "Successfully migrated legacy systems to modern stacks with {skills}.",
    "Actively used {skills} in hackathons and coding challenges.",
    "Strong problem-solving ability demonstrated through {skills}.",
    "Practical exposure to real-time problem solving using {skills}.",
    "Worked on academic and personal projects using {skills}.",
    "Researched and implemented innovative solutions with {skills}.",
    "Developed proof of concepts and prototypes using {skills}.",
    "Hands-on involvement in testing and debugging with {skills}.",
    "Collaborated with clients to deliver solutions using {skills}.",
    "Created technical documentation and reports involving {skills}.",
    "Recognized for delivering high-quality results with {skills}.",
    "Applied {skills} in both independent and team-based projects.",
    "Demonstrated adaptability by quickly learning {skills}.",
    "Certified in {skills} with industry-recognized credentials.",
    "Integrated third-party tools and services with {skills}.",
    "Utilized {skills} in research and academic publications.",
    "Focused on clean code and best practices while working with {skills}.",
    "Accomplished multiple internships/projects using {skills}.",
    "Implemented automation pipelines and tools with {skills}.",
    "Leveraged {skills} to support data-driven decision making.",
    "Practical implementation of coursework projects with {skills}.",
    "Experience in deployment and monitoring using {skills}.",
    "Contributed to cross-border team projects with {skills}.",
    "Strong foundation in {skills} with real-world project exposure.",
]

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def remove_overlapping_entities(entities):
    """Sort by start position, then prefer longer spans"""
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    cleaned = []
    last_end = -1
    for start, end, label in entities:
        if start >= last_end:  # keep only non-overlapping
            cleaned.append((start, end, label))
            last_end = end
    return cleaned


def generate_cv_training_data(num_examples):
    """Generate synthetic CV training data with skill entity annotations"""
    TRAIN_DATA = []
    for _ in range(num_examples):
        num_sentences = random.randint(2, 4)
        cv_text = ""
        entities = []
        cursor = 0

        for _ in range(num_sentences):
            selected_skills = random.sample(known_skills, random.randint(2, 5))
            template = random.choice(sentence_templates)
            sentence = template.format(skills=", ".join(selected_skills))
            cv_text += sentence + " "

            # Capture entity positions
            for skill in selected_skills:
                start_idx = sentence.find(skill, 0) + cursor
                if start_idx != -1:
                    end_idx = start_idx + len(skill)
                    entities.append((start_idx, end_idx, "SKILL"))

            cursor += len(sentence) + 1  # +1 for space

        # Remove overlaps
        entities = remove_overlapping_entities(entities)
        TRAIN_DATA.append((cv_text.strip(), {"entities": entities}))

    return TRAIN_DATA


def clean_training_data(nlp, train_data):
    """
    Validate and clean training data so that all entities align with spaCy tokens.
    Misaligned spans are removed.
    """
    cleaned_data = []
    for text, ann in train_data:
        doc = nlp.make_doc(text)
        entities = ann.get("entities", [])

        # Check alignment
        tags = offsets_to_biluo_tags(doc, entities)

        if "-" in tags:  
            # filter only valid entities
            valid_entities = []
            for start, end, label in entities:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:  # only keep spans that align
                    valid_entities.append((span.start_char, span.end_char, label))
            entities = valid_entities

        if entities:  # keep only if we have valid entities left
            cleaned_data.append((text, {"entities": entities}))

    return cleaned_data


def debug_train_data(train_data):
    """Debug and validate training data for issues"""
    print("🔎 Running TRAIN_DATA checks...\n")

    for i, (text, ann) in enumerate(train_data):
        entities = ann.get("entities", [])
        spans = []

        for start, end, label in entities:
            span = text[start:end]

            # 1️⃣ Check: Whitespace or punctuation around entity
            if span != span.strip():
                print(f"[Whitespace Issue] Example {i} -> '{span}' in: {text}")

            # 2️⃣ Check: Misaligned indices
            if text[start:end] != span:
                print(f"[Index Issue] Example {i} -> ({start}, {end}) gives '{span}' but text slice is '{text[start:end]}'")

            # 3️⃣ Check: Overlapping entities
            for s, e, l in spans:
                if (start < e and end > s):
                    print(f"[Overlap Issue] Example {i} -> '{span}' overlaps with '{text[s:e]}'")
            spans.append((start, end, label))

    print("\n✅ Finished checking TRAIN_DATA.")


def train_skill_ner(TRAIN_DATA, output_dir, n_epochs, model_name="en_core_web_lg"):
    """
    Fine-tune SpaCy NER model to detect SKILL entities and save it.
    
    Args:
        TRAIN_DATA (list): List of training examples [(text, {"entities": [...]})]
        output_dir (str): Directory to save the trained model
        n_epochs (int): Number of training epochs
        model_name (str): spaCy model to load (default: en_core_web_sm)
    """
    # Load SpaCy and add NER
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"❌ Model '{model_name}' not found. Downloading...")
        os.system(f"python -m spacy download {model_name}")
        nlp = spacy.load(model_name)
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    if "SKILL" not in ner.labels:
        ner.add_label("SKILL")

    # Fine-tune NER with epoch printing
    if TRAIN_DATA:
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.resume_training()
            for epoch in range(n_epochs):
                random.shuffle(TRAIN_DATA)
                losses = {}
                batches = minibatch(TRAIN_DATA, size=compounding(4.0, 16.0, 1.5))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    examples = [Example.from_dict(nlp.make_doc(t), a) for t, a in batch]
                    nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
                print(f"Epoch {epoch+1}/{n_epochs} — Losses: {losses}")

        # Save the trained model
        nlp.to_disk(output_dir)
        print(f"✅ Model trained and saved to {output_dir}")

    return nlp


def get_or_train_model(TRAIN_DATA, model_dir="./skill_ner_model"):
    """Load trained model if present, else train new one"""
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print("📂 Loading saved model...")
        return spacy.load(model_dir)
    else:
        if TRAIN_DATA is None:
            raise ValueError("⚠ TRAIN_DATA must be provided to train the model because model folder is missing.")
        print("⚡ No saved model found. Training new model...")
        nlp = train_skill_ner(TRAIN_DATA, model_dir, n_epochs=20)
        return nlp


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

    known_set = set([k.lower() for k in known_skills])
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

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Generate training data
    print("🔄 Generating training data...")
    TRAIN_DATA = generate_cv_training_data(500)
    
    # Save to JSON
    with open("elaborated_skill_train_data.json", "w") as f:
        json.dump(TRAIN_DATA, f, indent=4)
    print(f"✅ Generated {len(TRAIN_DATA)} training examples")
    
    # Show samples
    print("\n📋 Sample training data:")
    for i in range(min(3, len(TRAIN_DATA))):
        print(TRAIN_DATA[i])
    
    # Clean training data
    print("\n🧹 Cleaning training data...")
    nlp_tmp = spacy.blank("en")
    TRAIN_DATA = clean_training_data(nlp_tmp, TRAIN_DATA)
    print(f"✅ Cleaned dataset size: {len(TRAIN_DATA)} samples")
    
    # Debug training data
    debug_train_data(TRAIN_DATA)
    
    # Train or load model
    print("\n🤖 Loading/Training model...")
    nlp = get_or_train_model(TRAIN_DATA)
    
    # Set up PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in known_skills]
    matcher.add("SKILL", patterns)
    
    # Test on sample resumes
    test_resume_1 = """
FRANK WU
Full-Stack / Cloud
+65 9123 4567 | frank.wu@example.sg | https://linkedin.com/in/frankwu | https://github.com/frankwu
React native apps, React.js SPA, NodeJs API services, Express.JS routes.
Database work with MySQL, MongoDB, and Redis caching.
    """
    
    test_resume_2 = """
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
    personal_info = extract_personal_info(test_resume_1)
    known_skills_found, unknown = extract_skills(nlp, matcher, test_resume_1)
    print("🧑 Personal Info:", personal_info)
    print("✅ Known Skills:", known_skills_found)
    print("❓ Unknown Skills:", unknown)
    
    print("\n" + "="*80)
    print("TEST RESUME 2")
    print("="*80)
    personal_info = extract_personal_info(test_resume_2)
    known_skills_found, unknown = extract_skills(nlp, matcher, test_resume_2)
    print("🧑 Personal Info:", personal_info)
    print("✅ Known Skills:", known_skills_found)
    print("❓ Unknown Skills:", unknown)