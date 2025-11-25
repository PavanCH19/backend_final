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
            "required_skills": ["Network Security", "Penetration Testing", "CISSP", "Firewall", "Encryption", "Python", "Risk Assessment"]
        },
        "ai_ml": {
            "domain": "Artificial Intelligence & Machine Learning",
            "required_skills": ["Python", "TensorFlow", "PyTorch", "Keras", "Deep Learning", "NLP", "Computer Vision"]
        },
        "cloud_computing": {
            "domain": "Cloud Computing",
            "required_skills": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Cloud Security"]
        },
        "blockchain": {
            "domain": "Blockchain Development",
            "required_skills": ["Solidity", "Ethereum", "Smart Contracts", "Web3.js", "Truffle", "Blockchain Architecture", "Cryptography"]
        },
        "game_development": {
            "domain": "Game Development",
            "required_skills": ["Unity", "Unreal Engine", "C#", "C++", "Game Physics", "3D Modeling", "Shader Programming"]
        },
        "embedded_systems": {
            "domain": "Embedded Systems",
            "required_skills": ["C", "C++", "Microcontrollers", "RTOS", "IoT", "Sensors", "PCB Design"]
        },
        "ar_vr": {
            "domain": "AR/VR Development",
            "required_skills": ["Unity", "Unreal Engine", "C#", "Blender", "3D Modeling", "XR Interaction Toolkit", "Oculus SDK"]
        },
        "ui_ux_design": {
            "domain": "UI/UX Design",
            "required_skills": ["Figma", "Adobe XD", "Wireframing", "Prototyping", "User Research", "Design Systems", "Accessibility"]
        }
    }


def get_skill_pools():
    """Return comprehensive skill pools for each domain"""
    return {
        "data_science": [
            "Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", 
            "Matplotlib", "Seaborn", "Jupyter", "Docker", "Deep Learning", "Machine Learning", 
            "Statistics", "Data Visualization", "Big Data", "Spark", "Hadoop", "Keras", "XGBoost",
            "Feature Engineering", "A/B Testing", "Time Series", "Clustering", "Regression", "Classification"
        ],
        "web_dev": [
            "JavaScript", "React", "Vue.js", "Angular", "Node.js", "Express", "HTML", "CSS", 
            "MongoDB", "PostgreSQL", "MySQL", "Redis", "GraphQL", "REST API", "TypeScript", 
            "Webpack", "Git", "Bootstrap", "Sass", "Tailwind CSS", "Next.js", "Nuxt.js", "Django",
            "Flask", "FastAPI", "Spring Boot", "PHP", "Laravel", "Ruby on Rails", "OAuth", "JWT"
        ],
        "mobile": [
            "Java", "Kotlin", "Swift", "React Native", "Flutter", "Dart", "iOS", "Android", 
            "Xcode", "Android Studio", "Firebase", "SQLite", "Core Data", "UIKit", "SwiftUI",
            "Jetpack Compose", "Realm", "Push Notifications", "In-App Purchases", "MapKit", "Camera API"
        ],
        "devops": [
            "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Jenkins", "Terraform", "Ansible", 
            "Linux", "Bash", "Python", "CI/CD", "Git", "Monitoring", "Nagios", "Prometheus",
            "Grafana", "ELK Stack", "GitLab CI", "GitHub Actions", "CircleCI", "Chef", "Puppet",
            "Helm", "Vault", "Consul", "Nginx", "Apache", "Load Balancing"
        ],
        "security": [
            "Network Security", "Penetration Testing", "CISSP", "CEH", "Firewall", "Encryption", 
            "Python", "Wireshark", "Metasploit", "Nmap", "Risk Assessment", "Compliance", "SIEM",
            "Burp Suite", "Kali Linux", "SQL Injection", "XSS", "OWASP", "Security Policies",
            "IAM", "Zero Trust", "Threat Modeling", "Forensics", "Malware Analysis"
        ],
        "ai_ml": [
            "Python", "TensorFlow", "PyTorch", "Keras", "OpenCV", "Hugging Face", "Transformers", 
            "NLP", "Deep Learning", "Reinforcement Learning", "GANs", "Speech Recognition", 
            "BERT", "LSTM", "Attention Mechanisms", "Computer Vision", "Data Augmentation"
        ],
        "cloud": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Serverless", 
            "CloudFormation", "Cloud Security", "IAM", "Lambda", "DevOps Integration", "Load Balancing"
        ],
        "blockchain": [
            "Solidity", "Ethereum", "Hyperledger", "Smart Contracts", "Web3.js", "Truffle", 
            "Ganache", "Metamask", "DeFi", "NFT", "Consensus Mechanisms", "Cryptography", 
            "Blockchain Architecture", "Rust", "Polkadot", "Solana"
        ],
        "game_development": [
            "Unity", "Unreal Engine", "C#", "C++", "3D Modeling", "Blender", "Shader Programming", 
            "Game Physics", "AI for Games", "Animation", "VR Integration", "Level Design", 
            "Audio Design", "Optimization"
        ],
        "embedded": [
            "C", "C++", "Microcontrollers", "ARM", "Raspberry Pi", "Arduino", "IoT", "Sensors", 
            "RTOS", "UART", "SPI", "I2C", "PCB Design", "Firmware", "MQTT", "Embedded Linux"
        ],
        "ar_vr": [
            "Unity", "Unreal Engine", "C#", "3D Modeling", "XR Interaction Toolkit", "Oculus SDK", 
            "ARKit", "ARCore", "Vuforia", "Motion Tracking", "Spatial Mapping", "Hand Tracking"
        ],
        "ui_ux": [
            "Figma", "Adobe XD", "Sketch", "InVision", "Wireframing", "Prototyping", "User Research",
            "Design Systems", "Typography", "Color Theory", "Accessibility", "Interaction Design",
            "User Testing", "Journey Mapping"
        ]
    }


def get_project_templates():
    """Return project templates for each domain"""
    return {
        "data_science": ["Customer Churn Prediction", "Sales Forecasting Model", "Recommendation System", 
                        "Fraud Detection Algorithm", "Image Classification", "Natural Language Processing",
                        "Sentiment Analysis Tool", "Price Prediction Model", "Supply Chain Optimization",
                        "Credit Risk Assessment", "Anomaly Detection System", "Market Basket Analysis"],
        "web_dev": ["E-commerce Website", "Social Media Platform", "Portfolio Website", "Blog Platform", 
                   "Task Management App", "Real-time Chat Application", "Video Streaming Service",
                   "Online Learning Platform", "Restaurant Booking System", "Job Portal",
                   "Content Management System", "Music Player App", "Weather Dashboard"],
        "mobile": ["Weather App", "Fitness Tracker", "Food Delivery App", "Social Media App", 
                  "Game Application", "Banking App", "E-commerce App", "Music Streaming App",
                  "Travel Planner", "Expense Tracker", "Recipe App", "Language Learning App"],
        "devops": ["CI/CD Pipeline Setup", "Infrastructure as Code", "Container Orchestration", 
                  "Monitoring Dashboard", "Automated Deployment", "Cloud Migration",
                  "Log Aggregation System", "Backup and Recovery", "Auto-scaling Setup",
                  "Multi-cloud Strategy", "Disaster Recovery Plan", "GitOps Implementation"],
        "security": ["Vulnerability Assessment", "Security Audit", "Network Monitoring System", 
                    "Incident Response Plan", "Security Training Program", "Compliance Framework",
                    "Penetration Testing Report", "Security Information Dashboard", "Access Control System",
                    "Threat Intelligence Platform", "Data Loss Prevention", "Security Automation"],
        "ai_ml": ["Chatbot with NLP", "Image Captioning", "Voice Recognition", "Text Summarizer", 
                 "Recommendation Engine", "Face Detection System", "Autonomous Driving Simulation",
                 "GAN-based Image Generation", "AI-based Tutoring System", "Reinforcement Learning Game Agent"],
        "cloud": ["Serverless Architecture", "Cloud Cost Optimizer", "Scalable Web App on AWS", 
                 "Cloud Backup System", "Multi-region Deployment", "Containerized Microservices", 
                 "Load Balancer Setup", "Cloud Monitoring System", "Auto-healing Infrastructure"],
        "blockchain": ["NFT Marketplace", "DeFi Lending Platform", "Decentralized Voting System", 
                      "Supply Chain Blockchain", "Smart Contract Wallet", "Crypto Payment Gateway",
                      "Blockchain-based Identity Verification", "DAO Platform", "Token Exchange System"],
        "game_development": ["2D Platformer", "3D Adventure Game", "Racing Game", "Multiplayer Shooter", 
                    "Puzzle Game", "VR Exploration Game", "Strategy Simulation", "Educational Game"],
        "embedded": ["Smart Home Automation", "IoT Weather Station", "Industrial Sensor Network", 
                    "Smart Parking System", "Energy Monitoring Device", "Wearable Health Tracker", 
                    "Drone Controller", "IoT Security Camera"],
        "ar_vr": ["Virtual Museum Tour", "AR Furniture Placement", "VR Training Simulator", 
                  "AR Shopping App", "VR Real Estate Tour", "Immersive Education Platform", 
                  "Mixed Reality Game"],
        "ui_ux": ["Design System for SaaS App", "Redesign of E-commerce Checkout Flow", "Mobile-first Dashboard", 
                 "Accessibility Improvement Project", "Onboarding Flow Design", "Dark Mode Interface",
                 "User Research Report", "Design Prototype for Social App"]
    }


def get_job_titles():
    """Return job titles for each domain"""
    return {
        "data_science": ["Data Scientist", "ML Engineer", "Data Analyst", "Research Scientist",
                        "Business Intelligence Analyst", "Data Engineer", "Analytics Manager",
                        "Quantitative Analyst", "Data Architect"],
        "web_dev": ["Frontend Developer", "Backend Developer", "Full Stack Developer", "Web Developer",
                   "UI Developer", "JavaScript Developer", "React Developer", "Node.js Developer",
                   "Software Engineer", "Web Architect"],
        "mobile": ["iOS Developer", "Android Developer", "Mobile Developer", "App Developer",
                  "React Native Developer", "Flutter Developer", "Mobile Architect",
                  "Mobile UI/UX Developer", "Mobile QA Engineer"],
        "devops": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Infrastructure Engineer",
                  "Platform Engineer", "Release Engineer", "Build Engineer", "Systems Engineer",
                  "Automation Engineer"],
        "security": ["Security Analyst", "Cybersecurity Engineer", "Security Consultant", "SOC Analyst",
                    "Penetration Tester", "Security Architect", "Information Security Manager",
                    "Threat Intelligence Analyst", "Security Operations Engineer"],
        "ai_ml": ["AI Engineer", "ML Engineer", "Deep Learning Engineer", "NLP Engineer",
                 "Computer Vision Engineer", "AI Research Scientist", "Data Scientist"],
        "cloud": ["Cloud Engineer", "Cloud Solutions Architect", "Cloud Administrator", "DevOps Engineer",
                 "Cloud Security Specialist", "Site Reliability Engineer"],
        "blockchain": ["Blockchain Developer", "Smart Contract Engineer", "Web3 Developer", 
                      "Blockchain Architect", "DeFi Engineer", "Crypto Developer"],
        "game_development": ["Game Developer", "Unity Developer", "Unreal Developer", "Game Designer", 
                    "Technical Artist", "Gameplay Programmer", "Level Designer"],
        "embedded": ["Embedded Engineer", "IoT Developer", "Firmware Engineer", "Hardware Design Engineer",
                    "Systems Engineer", "Embedded Software Developer"],
        "ar_vr": ["AR Developer", "VR Developer", "Unity XR Engineer", "3D Interaction Designer", 
                  "Immersive Experience Designer", "AR/VR Research Engineer"],
        "ui_ux": ["UI Designer", "UX Designer", "Product Designer", "Interaction Designer",
                 "Design Researcher", "Visual Designer", "UX Strategist"]
    }
