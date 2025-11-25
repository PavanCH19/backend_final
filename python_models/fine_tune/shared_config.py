# ============================================================================
# shared_config.py - Shared configuration and constants
# ============================================================================

import random
import spacy
from spacy.matcher import PhraseMatcher

# Skills catalog
KNOWN_SKILLS = [
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

SENTENCE_TEMPLATES = [
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

MODEL_DIR = "./skill_ner_model"
MODEL_NAME = "en_core_web_lg"