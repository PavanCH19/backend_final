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
        }
    }


def get_skill_pools():
    """Return comprehensive skill pools for each domain"""
    return {
        "data_science": ["Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", 
                        "Matplotlib", "Seaborn", "Jupyter", "Docker", "Deep Learning", "Machine Learning", 
                        "Statistics", "Data Visualization", "Big Data", "Spark", "Hadoop", "Keras", "XGBoost",
                        "Feature Engineering", "A/B Testing", "Time Series", "Clustering", "Regression", "Classification"],
        "web_dev": ["JavaScript", "React", "Vue.js", "Angular", "Node.js", "Express", "HTML", "CSS", 
                   "MongoDB", "PostgreSQL", "MySQL", "Redis", "GraphQL", "REST API", "TypeScript", 
                   "Webpack", "Git", "Bootstrap", "Sass", "Tailwind CSS", "Next.js", "Nuxt.js", "Django",
                   "Flask", "FastAPI", "Spring Boot", "PHP", "Laravel", "Ruby on Rails", "OAuth", "JWT"],
        "mobile": ["Java", "Kotlin", "Swift", "React Native", "Flutter", "Dart", "iOS", "Android", 
                  "Xcode", "Android Studio", "Firebase", "SQLite", "Core Data", "UIKit", "SwiftUI",
                  "Jetpack Compose", "Realm", "Push Notifications", "In-App Purchases", "MapKit", "Camera API"],
        "devops": ["Docker", "Kubernetes", "AWS", "Azure", "GCP", "Jenkins", "Terraform", "Ansible", 
                  "Linux", "Bash", "Python", "CI/CD", "Git", "Monitoring", "Nagios", "Prometheus",
                  "Grafana", "ELK Stack", "GitLab CI", "GitHub Actions", "CircleCI", "Chef", "Puppet",
                  "Helm", "Vault", "Consul", "Nginx", "Apache", "Load Balancing"],
        "security": ["Network Security", "Penetration Testing", "CISSP", "CEH", "Firewall", "Encryption", 
                    "Python", "Wireshark", "Metasploit", "Nmap", "Risk Assessment", "Compliance", "SIEM",
                    "Burp Suite", "Kali Linux", "SQL Injection", "XSS", "OWASP", "Security Policies",
                    "IAM", "Zero Trust", "Threat Modeling", "Forensics", "Malware Analysis"]
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
                    "Threat Intelligence Platform", "Data Loss Prevention", "Security Automation"]
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
                    "Threat Intelligence Analyst", "Security Operations Engineer"]
    }
