def get_domain_requirements():
    """Define and return domain requirements for different job categories"""
    return {
        "data_science": {
            "domain": "Data Science",
            "required_skills": ["Python", "Pandas", "NumPy", "Scikit-learn", "PyTorch", "Docker", "Deep Learning"]
        },
        "web_development": {
            "domain": "Web Development",
            "required_skills": ["JavaScript", "React", "Node.js", "HTML", "CSS", "MongoDB", "Express"]
        },
        "mobile_development": {
            "domain": "Mobile Development",
            "required_skills": ["Java", "Kotlin", "Swift", "React Native", "Flutter", "iOS", "Android"]
        },
        "devops": {
            "domain": "DevOps",
            "required_skills": ["Docker", "Kubernetes", "AWS", "Jenkins", "Terraform", "Linux", "CI/CD"]
        },
        "cybersecurity": {
            "domain": "Cybersecurity",
            "required_skills": ["Network Security", "Penetration Testing", "CISSP", "Firewall", "Encryption", "Python"]
        },
        "ai_ml": {
            "domain": "AI & Machine Learning",
            "required_skills": ["Python", "TensorFlow", "PyTorch", "Keras", "Deep Learning", "NLP", "Computer Vision"]
        },
        "cloud_computing": {
            "domain": "Cloud Computing",
            "required_skills": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform"]
        },
        "blockchain": {
            "domain": "Blockchain Development",
            "required_skills": ["Solidity", "Ethereum", "Smart Contracts", "Web3.js", "Cryptography"]
        },
        "game_development": {
            "domain": "Game Development",
            "required_skills": ["Unity", "Unreal Engine", "C#", "C++", "Game Physics"]
        },
        "embedded_systems": {
            "domain": "Embedded Systems",
            "required_skills": ["C", "C++", "Microcontrollers", "RTOS", "IoT"]
        },
        "ar_vr": {
            "domain": "AR / VR Development",
            "required_skills": ["Unity", "C#", "3D Modeling", "XR Toolkit", "Oculus SDK"]
        },
        "ui_ux_design": {
            "domain": "UI / UX Design",
            "required_skills": ["Figma", "Wireframing", "Prototyping", "User Research", "Accessibility"]
        },

        # 🔥 LANGUAGE / FRAMEWORK DOMAINS
        "java_development": {
            "domain": "Java Development",
            "required_skills": ["Java", "OOP", "Spring Boot", "Hibernate", "Microservices"]
        },
        "python_development": {
            "domain": "Python Development",
            "required_skills": ["Python", "Django", "Flask", "FastAPI", "REST APIs"]
        },
        "nodejs_development": {
            "domain": "Node.js Development",
            "required_skills": ["Node.js", "Express", "MongoDB", "JWT", "Async Programming"]
        },
        "javascript_development": {
            "domain": "JavaScript Development",
            "required_skills": ["JavaScript", "ES6+", "DOM", "Promises", "Browser APIs"]
        },
        "react_development": {
            "domain": "React Development",
            "required_skills": ["React", "Hooks", "Redux", "React Router", "Performance Optimization"]
        }
    }
def get_skill_pools():
    """Return comprehensive skill pools for each domain"""
    return {
        "data_science": [
            "Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn",
            "TensorFlow", "PyTorch", "Statistics", "ML", "DL"
        ],
        "web_development": [
            "JavaScript", "React", "Node.js", "HTML", "CSS",
            "MongoDB", "PostgreSQL", "REST API", "JWT"
        ],
        "mobile_development": [
            "Java", "Kotlin", "Swift", "Flutter", "React Native"
        ],
        "devops": [
            "Docker", "Kubernetes", "AWS", "Terraform", "CI/CD", "Linux"
        ],
        "cybersecurity": [
            "Network Security", "Penetration Testing", "Encryption",
            "Wireshark", "Metasploit", "OWASP"
        ],
        "ai_ml": [
            "Python", "TensorFlow", "PyTorch", "NLP",
            "Computer Vision", "Transformers"
        ],
        "cloud_computing": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "IAM"
        ],
        "blockchain": [
            "Solidity", "Ethereum", "Smart Contracts", "Web3", "Cryptography"
        ],
        "game_development": [
            "Unity", "Unreal Engine", "C#", "C++", "Game AI"
        ],
        "embedded_systems": [
            "C", "C++", "RTOS", "IoT", "Microcontrollers"
        ],
        "ar_vr": [
            "Unity", "XR Toolkit", "ARKit", "ARCore", "3D Modeling"
        ],
        "ui_ux_design": [
            "Figma", "Wireframes", "User Research", "Accessibility"
        ],

        # 🔥 Language / Framework
        "java": [
            "Java", "Spring Boot", "Hibernate", "JPA", "Microservices", "JUnit"
        ],
        "python": [
            "Python", "Django", "Flask", "FastAPI", "AsyncIO"
        ],
        "nodejs": [
            "Node.js", "Express", "MongoDB", "JWT", "Socket.io"
        ],
        "javascript": [
            "JavaScript", "ES6+", "DOM", "Promises", "Event Loop"
        ],
        "react": [
            "React", "Hooks", "Redux", "Next.js", "Performance Optimization"
        ]
    }
def get_project_templates():
    """Return project templates for each domain"""
    return {
        "data_science": [
            "Customer Churn Prediction",
            "Recommendation System",
            "Fraud Detection"
        ],
        "web_development": [
            "E-commerce Website",
            "Social Media Platform",
            "Online Learning Platform"
        ],
        "mobile_development": [
            "Fitness Tracker App",
            "Food Delivery App",
            "Banking App"
        ],
        "devops": [
            "CI/CD Pipeline",
            "Infrastructure as Code",
            "Monitoring System"
        ],
        "cybersecurity": [
            "Vulnerability Assessment",
            "Penetration Testing Tool",
            "Security Dashboard"
        ],
        "ai_ml": [
            "Chatbot",
            "Face Recognition",
            "Text Summarizer"
        ],
        "cloud_computing": [
            "Serverless Web App",
            "Cloud Cost Optimizer"
        ],
        "blockchain": [
            "NFT Marketplace",
            "DeFi Platform"
        ],
        "game_development": [
            "2D Platformer",
            "3D Adventure Game"
        ],
        "embedded_systems": [
            "Smart Home Automation",
            "IoT Weather Station"
        ],
        "ar_vr": [
            "Virtual Museum",
            "AR Shopping App"
        ],
        "ui_ux_design": [
            "Design System",
            "Mobile App Redesign"
        ],

        # 🔥 Language / Framework
        "java": [
            "Spring Boot REST API",
            "Banking Management System"
        ],
        "python": [
            "FastAPI Backend",
            "Automation Tool"
        ],
        "nodejs": [
            "Authentication API",
            "Real-time Chat Server"
        ],
        "javascript": [
            "Interactive Dashboard",
            "Browser Game"
        ],
        "react": [
            "Admin Dashboard",
            "E-commerce Frontend"
        ]
    }
def get_job_titles():
    """Return job titles for each domain"""
    return {
        "data_science": ["Data Scientist", "ML Engineer", "Data Analyst"],
        "web_development": ["Frontend Developer", "Backend Developer", "Full Stack Developer"],
        "mobile_development": ["Android Developer", "iOS Developer", "Flutter Developer"],
        "devops": ["DevOps Engineer", "SRE", "Cloud Engineer"],
        "cybersecurity": ["Security Analyst", "Penetration Tester"],
        "ai_ml": ["AI Engineer", "ML Engineer"],
        "cloud_computing": ["Cloud Architect", "Cloud Engineer"],
        "blockchain": ["Blockchain Developer", "Web3 Engineer"],
        "game_development": ["Game Developer", "Unity Developer"],
        "embedded_systems": ["Embedded Engineer", "Firmware Developer"],
        "ar_vr": ["AR Developer", "VR Engineer"],
        "ui_ux_design": ["UX Designer", "UI Designer"],

        # 🔥 Language / Framework
        "java": ["Java Developer", "Backend Engineer"],
        "python": ["Python Developer", "API Developer"],
        "nodejs": ["Node.js Developer", "Backend Engineer"],
        "javascript": ["JavaScript Developer", "Frontend Developer"],
        "react": ["React Developer", "UI Engineer"]
    }