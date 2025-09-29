# Enhanced Resume Classifier - Adding Missing Steps A & B
# File: enhanced_resume_classifier.py

import os
import json
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import re
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings('ignore')

# Previous setup code (Steps 0-1 remain the same)
print("=== Step 0: Project Setup ===")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Create project structure
project_folders = ['data', 'models', 'notebooks', 'src', 'data/domain_requirements']
for folder in project_folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"Created folder: {folder}")

# Domain requirements (same as before)
domain_requirements = {
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

# Save domain requirements
for domain_key, requirements in domain_requirements.items():
    file_path = f"data/domain_requirements/{domain_key}.json"
    with open(file_path, 'w') as f:
        json.dump(requirements, f, indent=2)
    print(f"Saved: {file_path}")

# Generate synthetic dataset (same function as before, but we'll clean it after)
def generate_synthetic_resumes(n_samples=2000):
    """Generate synthetic resume data following the specified schema"""
    
    # Skill pools for different domains
    all_skills = {
        "data_science": ["Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", 
                        "Matplotlib", "Seaborn", "Jupyter", "Docker", "Deep Learning", "Machine Learning", 
                        "Statistics", "Data Visualization", "Big Data", "Spark", "Hadoop", "python3", "PYTHON",
                        "pyTorch DL", "sql database mgmt", "tensorflow 2.0", "scikit learn"],
        "web_dev": ["JavaScript", "React", "Vue.js", "Angular", "Node.js", "Express", "HTML", "CSS", 
                   "MongoDB", "PostgreSQL", "MySQL", "Redis", "GraphQL", "REST API", "TypeScript", 
                   "Webpack", "Git", "Bootstrap", "Sass", "javasript", "reactjs", "nodejs"],
        "mobile": ["Java", "Kotlin", "Swift", "React Native", "Flutter", "Dart", "iOS", "Android", 
                  "Xcode", "Android Studio", "Firebase", "SQLite", "Core Data", "UIKit", "SwiftUI"],
        "devops": ["Docker", "Kubernetes", "AWS", "Azure", "GCP", "Jenkins", "Terraform", "Ansible", 
                  "Linux", "Bash", "Python", "CI/CD", "Git", "Monitoring", "Nagios", "Prometheus"],
        "security": ["Network Security", "Penetration Testing", "CISSP", "CEH", "Firewall", "Encryption", 
                    "Python", "Wireshark", "Metasploit", "Nmap", "Risk Assessment", "Compliance", "SIEM"]
    }
    
    # Project templates (with some messy data to clean)
    project_templates = {
        "data_science": ["Customer Churn Prediction", "Sales Forecasting Model", "Recommendation System", 
                        "Fraud Detection Algorithm", "Image Classification", "Natural Language Processing",
                        "ML Project using Python", "Data Analysis Project", "NLP Project 2"],
        "web_dev": ["E-commerce Website", "Social Media Platform", "Portfolio Website", "Blog Platform", 
                   "Task Management App", "Real-time Chat Application", "Web App using React"],
        "mobile": ["Weather App", "Fitness Tracker", "Food Delivery App", "Social Media App", 
                  "Game Application", "Banking App", "Mobile Project"],
        "devops": ["CI/CD Pipeline Setup", "Infrastructure as Code", "Container Orchestration", 
                  "Monitoring Dashboard", "Automated Deployment", "Cloud Migration"],
        "security": ["Vulnerability Assessment", "Security Audit", "Network Monitoring System", 
                    "Incident Response Plan", "Security Training Program", "Compliance Framework"]
    }
    
    # Job titles (with variations to normalize)
    job_titles = {
        "data_science": ["Data Scientist", "ML Engineer", "Data Analyst", "Research Scientist", 
                        "Sr. Data Scientist", "Junior Data Analyst", "Machine Learning Eng"],
        "web_dev": ["Frontend Developer", "Backend Developer", "Full Stack Developer", "Web Developer",
                   "Sr. Frontend Dev", "Junior Web Developer"],
        "mobile": ["iOS Developer", "Android Developer", "Mobile Developer", "App Developer",
                  "Senior Mobile Dev", "Mobile App Developer"],
        "devops": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Infrastructure Engineer",
                  "Senior DevOps Eng", "Cloud Architect"],
        "security": ["Security Analyst", "Cybersecurity Engineer", "Security Consultant", "SOC Analyst",
                    "Sr. Security Analyst", "Information Security Specialist"]
    }
    
    domains = list(domain_requirements.keys())
    resumes = []
    
    for i in range(n_samples):
        # Choose preferred domain
        preferred_domain_key = random.choice(domains)
        preferred_domain = domain_requirements[preferred_domain_key]["domain"]
        
        # Generate skills (with messy variations)
        skill_key_mapping = {
            "data_science": "data_science",
            "web_development": "web_dev", 
            "mobile_development": "mobile",
            "devops": "devops",
            "cybersecurity": "security"
        }
        domain_skills = all_skills[skill_key_mapping.get(preferred_domain_key, "data_science")]
        other_skills = []
        for skill_set in all_skills.values():
            other_skills.extend(skill_set)
        other_skills = list(set(other_skills) - set(domain_skills))
        
        # Select skills with bias towards preferred domain
        n_domain_skills = random.randint(3, 8)
        n_other_skills = random.randint(0, 4)
        
        selected_skills = random.sample(domain_skills, min(n_domain_skills, len(domain_skills)))
        selected_skills.extend(random.sample(other_skills, min(n_other_skills, len(other_skills))))
        
        # Add some corrupted entries occasionally
        if random.random() < 0.05:  # 5% chance of corruption
            selected_skills.append("")  # Empty skill
            selected_skills.append("   ")  # Whitespace only
        
        # Generate projects
        domain_projects = project_templates[skill_key_mapping.get(preferred_domain_key, "data_science")]
        n_projects = random.randint(0, 4)  # Allow 0 projects for some candidates
        selected_projects = random.sample(domain_projects, min(n_projects, len(domain_projects)))
        
        # Generate work experience with messy data
        n_jobs = random.randint(0, 4)  # Allow 0 experience
        work_experience = []
        domain_job_titles = job_titles[skill_key_mapping.get(preferred_domain_key, "data_science")]

        for _ in range(n_jobs):
            title = random.choice(domain_job_titles)
            # Add some messy years data
            if random.random() < 0.1:  # 10% chance of messy data
                years = random.choice(["2.5 yrs", "N/A", -1, 150])  # Messy data
            else:
                years = random.randint(1, 8)
            work_experience.append({"title": title, "years": years})
        
        # Generate test score with some out-of-range values
        if random.random() < 0.05:  # 5% chance of out-of-range
            test_score = random.choice([-10, 120, None])
        else:
            test_score = max(0, min(100, int(np.random.normal(65, 20))))
        
        # Create resume (some might be corrupted)
        resume = {
            "skills": selected_skills,
            "projects": selected_projects,
            "work_experience": work_experience,
            "test_score": test_score,
            "preferred_domain": preferred_domain,
            "id": f"candidate_{i+1:04d}"
        }
        
        resumes.append(resume)
    
    return resumes

# Generate synthetic dataset with messy data
print("\n=== Step 1: Generating Synthetic Dataset (with messy data) ===")
synthetic_resumes = generate_synthetic_resumes(2000)
print(f"Generated {len(synthetic_resumes)} synthetic resumes (with intentional messy data)")

# ===============================
# STEP A: DATA CLEANING (MISSING STEP)
# ===============================

print("\n=== STEP A: DATA CLEANING (Deep Dive) ===")

class AdvancedDataCleaner:
    """
    Advanced data cleaning with embedding-based skill normalization and fuzzy matching
    """
    
    def __init__(self):
        # Load sentence transformer for embedding-based similarity
        print("Loading sentence transformer model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build canonical skill vocabulary from domain requirements
        self.canonical_skills = self._build_canonical_skills()
        self.canonical_embeddings = self.embedder.encode(self.canonical_skills)
        
        # Canonical job titles
        self.canonical_job_titles = [
            "data scientist", "data analyst", "machine learning engineer", 
            "frontend developer", "backend developer", "full stack developer",
            "mobile developer", "ios developer", "android developer",
            "devops engineer", "cloud engineer", "security analyst",
            "software engineer", "intern", "consultant"
        ]
        self.job_title_embeddings = self.embedder.encode(self.canonical_job_titles)
        
        # Statistics tracking
        self.cleaning_stats = {
            'skills_normalized': 0,
            'projects_cleaned': 0,
            'job_titles_normalized': 0,
            'test_scores_fixed': 0,
            'records_removed': 0
        }
    
    def _build_canonical_skills(self):
        """Build canonical skill vocabulary from domain requirements"""
        canonical_skills = set()
        for domain_data in domain_requirements.values():
            canonical_skills.update(skill.lower() for skill in domain_data.get('required_skills', []))
        
        # Add common variations
        additional_skills = [
            "python", "javascript", "java", "sql", "html", "css", "react", "node.js",
            "tensorflow", "pytorch", "pandas", "numpy", "docker", "kubernetes", 
            "aws", "mongodb", "postgresql", "git", "linux", "bash"
        ]
        canonical_skills.update(additional_skills)
        
        return sorted(list(canonical_skills))
    
    def normalize_skill(self, skill, similarity_threshold=0.8):
        """
        A.1 Skill normalization with embedding similarity and fuzzy matching
        """
        if not skill or not skill.strip():
            return None
        
        # Basic normalization
        cleaned_skill = skill.strip().lower()
        cleaned_skill = re.sub(r'[^\w\s]', ' ', cleaned_skill)  # Remove punctuation
        cleaned_skill = re.sub(r'\s+', ' ', cleaned_skill).strip()  # Normalize whitespace
        
        if len(cleaned_skill) < 2:
            return None
        
        # Direct match check
        if cleaned_skill in self.canonical_skills:
            return cleaned_skill
        
        # Embedding-based similarity
        skill_embedding = self.embedder.encode([cleaned_skill])
        similarities = cosine_similarity(skill_embedding, self.canonical_embeddings)[0]
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity >= similarity_threshold:
            self.cleaning_stats['skills_normalized'] += 1
            return self.canonical_skills[best_match_idx]
        
        # Fuzzy matching fallback
        best_fuzzy_score = 0
        best_fuzzy_match = None
        
        for canonical_skill in self.canonical_skills:
            fuzzy_score = fuzz.ratio(cleaned_skill, canonical_skill)
            if fuzzy_score > best_fuzzy_score and fuzzy_score >= 85:  # 85% threshold
                best_fuzzy_score = fuzzy_score
                best_fuzzy_match = canonical_skill
        
        if best_fuzzy_match:
            self.cleaning_stats['skills_normalized'] += 1
            return best_fuzzy_match
        
        # Keep original if no good match found
        return cleaned_skill
    
    def clean_skills_list(self, skills):
        """Clean and normalize entire skills list"""
        if not skills:
            return []
        
        cleaned_skills = []
        for skill in skills:
            normalized_skill = self.normalize_skill(skill)
            if normalized_skill:
                cleaned_skills.append(normalized_skill)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in cleaned_skills:
            if skill not in seen:
                seen.add(skill)
                unique_skills.append(skill)
        
        return unique_skills
    
    def clean_projects_list(self, projects):
        """
        A.2 Project title cleaning
        """
        if not projects:
            return []
        
        stopwords = {"and", "the", "project", "using", "with", "for", "in", "on", "a", "an"}
        cleaned_projects = []
        
        for project in projects:
            if not project or not project.strip():
                continue
            
            # Basic cleaning
            cleaned = project.lower().strip()
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Remove stopwords but keep important technical terms
            tokens = cleaned.split()
            important_tokens = []
            
            for token in tokens:
                if token not in stopwords and len(token) > 1:
                    # Keep technical terms
                    if any(tech in token for tech in ['nlp', 'cnn', 'rnn', 'gan', 'lstm', 'api', 'ml', 'ai']):
                        important_tokens.append(token)
                    elif token not in stopwords:
                        important_tokens.append(token)
            
            if len(important_tokens) >= 2:  # At least 2 meaningful tokens
                cleaned_project = ' '.join(important_tokens)
                cleaned_projects.append(cleaned_project)
                self.cleaning_stats['projects_cleaned'] += 1
        
        return cleaned_projects
    
    def normalize_job_title(self, title):
        """
        A.3 Work experience cleaning - normalize job titles
        """
        if not title or not title.strip():
            return None
        
        # Basic cleaning
        cleaned_title = title.lower().strip()
        cleaned_title = re.sub(r'\b(sr|senior|jr|junior|lead|principal)\b\.?', '', cleaned_title)
        cleaned_title = re.sub(r'[^\w\s]', ' ', cleaned_title)
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
        
        if len(cleaned_title) < 3:
            return None
        
        # Embedding-based similarity to canonical job titles
        title_embedding = self.embedder.encode([cleaned_title])
        similarities = cosine_similarity(title_embedding, self.job_title_embeddings)[0]
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity >= 0.7:  # Lower threshold for job titles
            self.cleaning_stats['job_titles_normalized'] += 1
            return self.canonical_job_titles[best_match_idx]
        
        return cleaned_title
    
    def clean_work_experience(self, work_experience):
        """Clean work experience list"""
        if not work_experience:
            return []
        
        cleaned_experience = []
        for exp in work_experience:
            if not isinstance(exp, dict):
                continue
            
            # Clean job title
            title = exp.get('title', '')
            normalized_title = self.normalize_job_title(title)
            
            # Clean years
            years = exp.get('years', 0)
            if isinstance(years, str):
                # Handle string years like "2.5 yrs", "N/A"
                if years.lower() in ['n/a', 'na', '', 'none']:
                    years = 0
                else:
                    # Extract numeric part
                    numeric_match = re.search(r'(\d+\.?\d*)', years)
                    years = float(numeric_match.group(1)) if numeric_match else 0
            
            # Clamp years to reasonable range
            if not isinstance(years, (int, float)):
                years = 0
            years = max(0, min(50, float(years)))  # Clamp to [0, 50]
            
            if normalized_title:
                cleaned_experience.append({
                    'title': normalized_title,
                    'years': years
                })
        
        return cleaned_experience
    
    def clean_test_score(self, test_score):
        """
        A.4 Test score cleaning and normalization
        """
        if test_score is None or test_score == '':
            return 0
        
        # Handle string scores
        if isinstance(test_score, str):
            numeric_match = re.search(r'(\d+\.?\d*)', test_score)
            test_score = float(numeric_match.group(1)) if numeric_match else 0
        
        # Ensure numeric
        try:
            test_score = float(test_score)
        except (ValueError, TypeError):
            test_score = 0
        
        # Clamp to [0, 100]
        original_score = test_score
        test_score = max(0, min(100, test_score))
        
        if abs(original_score - test_score) > 0.1:
            self.cleaning_stats['test_scores_fixed'] += 1
        
        return test_score
    
    def is_corrupted_record(self, resume):
        """
        A.5 Identify corrupted records for removal
        """
        # Check for empty core fields
        skills = resume.get('skills', [])
        projects = resume.get('projects', [])
        test_score = resume.get('test_score')
        domain = resume.get('preferred_domain', '')
        
        # Remove if all core fields are empty
        if (not skills or len(skills) == 0) and \
           (not projects or len(projects) == 0) and \
           (test_score is None or test_score == 0) and \
           not domain.strip():
            return True
        
        # Remove if missing domain
        if not domain or not domain.strip():
            return True
        
        return False
    
    def clean_dataset(self, resumes):
        """
        Complete data cleaning pipeline
        """
        print("Starting comprehensive data cleaning...")
        
        cleaned_resumes = []
        duplicates = set()
        
        for resume in resumes:
            # Check for corruption first
            if self.is_corrupted_record(resume):
                self.cleaning_stats['records_removed'] += 1
                continue
            
            # Clean each component
            cleaned_resume = resume.copy()
            
            # Clean skills
            cleaned_resume['skills'] = self.clean_skills_list(resume.get('skills', []))
            
            # Clean projects
            cleaned_resume['projects'] = self.clean_projects_list(resume.get('projects', []))
            
            # Clean work experience
            cleaned_resume['work_experience'] = self.clean_work_experience(resume.get('work_experience', []))
            
            # Clean test score
            cleaned_resume['test_score'] = self.clean_test_score(resume.get('test_score'))
            
            # Create duplicate detection key
            dup_key = (
                tuple(sorted(cleaned_resume['skills'])),
                tuple(sorted(cleaned_resume['projects'])),
                cleaned_resume['test_score'],
                cleaned_resume['preferred_domain']
            )
            
            # Check for duplicates
            if dup_key in duplicates:
                self.cleaning_stats['records_removed'] += 1
                continue
            
            duplicates.add(dup_key)
            cleaned_resumes.append(cleaned_resume)
        
        print(f"Data cleaning completed. Cleaned {len(cleaned_resumes)} resumes.")
        print(f"Cleaning statistics: {self.cleaning_stats}")
        
        return cleaned_resumes

# Apply data cleaning
cleaner = AdvancedDataCleaner()
cleaned_resumes = cleaner.clean_dataset(synthetic_resumes)

print(f"Original dataset: {len(synthetic_resumes)} resumes")
print(f"Cleaned dataset: {len(cleaned_resumes)} resumes")
print(f"Removed: {len(synthetic_resumes) - len(cleaned_resumes)} corrupted/duplicate records")

# Save cleaned dataset
with open('data/cleaned_resumes.json', 'w') as f:
    json.dump(cleaned_resumes, f, indent=2)

# Show cleaning examples
print("\n=== Data Cleaning Examples ===")
sample_original = synthetic_resumes[0]
sample_cleaned = cleaned_resumes[0] if cleaned_resumes else None

if sample_cleaned:
    print("Original skills:", sample_original.get('skills', [])[:5])
    print("Cleaned skills:", sample_cleaned.get('skills', [])[:5])
    print("Original projects:", sample_original.get('projects', []))
    print("Cleaned projects:", sample_cleaned.get('projects', []))

# ===============================
# STEP B: LABEL BALANCING (MISSING STEP)
# ===============================

print("\n=== STEP B: LABEL BALANCING (Class Normalization) ===")

# First, we need to generate labels for the cleaned data
def create_ground_truth_labels(resumes, domain_requirements):
    """Create ground truth labels using rule-based approach"""
    
    labeled_resumes = []
    label_stats = {"Fit": 0, "Partial Fit": 0, "Not Fit": 0}
    
    for resume in resumes:
        # Get domain requirements for this candidate's preferred domain
        domain_key = None
        for key, req in domain_requirements.items():
            if req["domain"] == resume["preferred_domain"]:
                domain_key = key
                break
        
        if domain_key is None:
            print(f"Warning: No requirements found for domain {resume['preferred_domain']}")
            continue
            
        required_skills = set(domain_requirements[domain_key]["required_skills"])
        candidate_skills = set(resume["skills"])
        
        # Calculate skill match ratio
        matched_skills = len(required_skills.intersection(candidate_skills))
        total_required_skills = len(required_skills)
        skill_match_ratio = matched_skills / total_required_skills if total_required_skills > 0 else 0
        
        # Calculate normalized test score
        test_score_norm = resume["test_score"] / 100.0
        
        # Count projects
        project_count = len(resume["projects"])
        
        # Apply labeling rules
        if (skill_match_ratio >= 0.70) and (test_score_norm >= 0.75) and (project_count >= 1):
            label = "Fit"
        elif (0.40 <= skill_match_ratio < 0.70) or (0.50 <= test_score_norm < 0.75):
            label = "Partial Fit"
        elif (skill_match_ratio < 0.40) or (test_score_norm < 0.50):
            label = "Not Fit"
        else:
            label = "Partial Fit"  # Default case
        
        # Add computed metrics and label to resume
        labeled_resume = resume.copy()
        labeled_resume.update({
            "label": label,
            "skill_match_ratio": round(skill_match_ratio, 3),
            "test_score_norm": round(test_score_norm, 3),
            "project_count": project_count,
            "matched_skills": matched_skills,
            "total_required_skills": total_required_skills
        })
        
        labeled_resumes.append(labeled_resume)
        label_stats[label] += 1
    
    return labeled_resumes, label_stats

# Generate initial labels
print("Generating initial labels for cleaned data...")
labeled_resumes, initial_label_stats = create_ground_truth_labels(cleaned_resumes, domain_requirements)

class LabelBalancer:
    """
    Label balancing with multiple strategies
    """
    
    def __init__(self):
        self.balancing_stats = {}
    
    def analyze_distribution(self, labeled_resumes):
        """B.1 Analyze label distribution"""
        label_counts = {}
        for resume in labeled_resumes:
            label = resume['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total = len(labeled_resumes)
        print("\n=== Initial Label Distribution ===")
        print(f"{'Label':<15} {'Count':<8} {'Percentage':<12}")
        print("-" * 35)
        
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"{label:<15} {count:<8} {percentage:>8.1f}%")
        
        # Detect imbalance
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nImbalance ratio: {imbalance_ratio:.2f} (max/min)")
        is_imbalanced = imbalance_ratio > 2.0  # Threshold for imbalance
        
        return label_counts, is_imbalanced
    
    def oversample_minority_classes(self, labeled_resumes, target_size=None):
        """
        B.2 Option 2: Oversampling minority classes
        """
        # Group by labels
        label_groups = {}
        for resume in labeled_resumes:
            label = resume['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(resume)
        
        # Determine target size
        if target_size is None:
            target_size = max(len(group) for group in label_groups.values())
        
        print(f"\n=== Oversampling to target size: {target_size} ===")
        
        balanced_resumes = []
        for label, group in label_groups.items():
            current_size = len(group)
            
            if current_size < target_size:
                # Oversample by duplicating with small variations
                oversampled = resample(group, 
                                     replace=True, 
                                     n_samples=target_size, 
                                     random_state=SEED)
                
                # Add slight variations to oversampled data to avoid exact duplicates
                for i, resume in enumerate(oversampled):
                    if i >= current_size:  # Only modify duplicated ones
                        # Create a copy and add small variation to ID
                        varied_resume = resume.copy()
                        varied_resume['id'] = f"{resume['id']}_dup_{i - current_size + 1}"
                        balanced_resumes.append(varied_resume)
                    else:
                        balanced_resumes.append(resume)
                
                print(f"  {label}: {current_size} → {target_size} (oversampled +{target_size - current_size})")
            else:
                balanced_resumes.extend(group)
                print(f"  {label}: {current_size} (no change needed)")
        
        return balanced_resumes
    
    def compute_class_weights(self, label_counts):
        """
        B.2 Option 3: Compute class weights for model training
        """
        total_samples = sum(label_counts.values())
        n_classes = len(label_counts)
        
        class_weights = {}
        
        print(f"\n=== Computing Class Weights ===")
        print(f"Total samples: {total_samples}, Classes: {n_classes}")
        
        for label, count in label_counts.items():
            weight = total_samples / (count * n_classes)
            class_weights[label] = weight
            print(f"  {label}: weight = {total_samples} / ({count} × {n_classes}) = {weight:.3f}")
        
        return class_weights
    
    def hybrid_balancing(self, labeled_resumes, oversample_factor=0.5):
        """
        B.2 Option 4: Hybrid approach (light oversampling + class weights)
        """
        print(f"\n=== Hybrid Balancing (oversample factor: {oversample_factor}) ===")
        
        # Analyze current distribution
        label_counts, is_imbalanced = self.analyze_distribution(labeled_resumes)
        
        if not is_imbalanced:
            print("Dataset is already balanced. Skipping balancing.")
            return labeled_resumes, label_counts
        
        # Light oversampling
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        target_size = int(max_count * (1 - oversample_factor) + min_count * oversample_factor)
        
        balanced_resumes = self.oversample_minority_classes(labeled_resumes, target_size)
        
        # Compute class weights for the balanced dataset
        new_label_counts, _ = self.analyze_distribution(balanced_resumes)
        class_weights = self.compute_class_weights(new_label_counts)
        
        return balanced_resumes, class_weights

# Apply label balancing
print("Analyzing label distribution...")
balancer = LabelBalancer()
initial_counts, is_imbalanced = balancer.analyze_distribution(labeled_resumes)

if is_imbalanced:
    print("Dataset is imbalanced. Applying hybrid balancing...")
    balanced_resumes, class_weights = balancer.hybrid_balancing(labeled_resumes, oversample_factor=0.3)
    print(f"Balanced dataset size: {len(balanced_resumes)}")
else:
    print("Dataset is already balanced.")
    balanced_resumes = labeled_resumes
    class_weights = balancer.compute_class_weights(initial_counts)

# Save balanced dataset
with open('data/balanced_labeled_resumes.json', 'w') as f:
    json.dump(balanced_resumes, f, indent=2)

# Save class weights for model training
with open('data/class_weights.json', 'w') as f:
    json.dump(class_weights, f, indent=2)

# Visualize before/after balancing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Before balancing
labels_before = list(initial_counts.keys())
counts_before = list(initial_counts.values())
ax1.bar(labels_before, counts_before, color=['lightcoral', 'orange', 'lightgreen'])
ax1.set_title('Before Balancing')
ax1.set_xlabel('Labels')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(counts_before):
    ax1.text(i, v + max(counts_before)*0.01, str(v), ha='center')

# After balancing
balanced_counts = {}
for resume in balanced_resumes:
    label = resume['label']
    balanced_counts[label] = balanced_counts.get(label, 0) + 1

labels_after = list(balanced_counts.keys())
counts_after = list(balanced_counts.values())
ax2.bar(labels_after, counts_after, color=['lightcoral', 'orange', 'lightgreen'])
ax2.set_title('After Balancing')
ax2.set_xlabel('Labels')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(counts_after):
    ax2.text(i, v + max(counts_after)*0.01, str(v), ha='center')

plt.tight_layout()
plt.savefig('data/label_balancing_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== Steps A & B Complete ===")
print("Added missing steps:")
print("✓ Step A: Advanced Data Cleaning")
print("  - Embedding-based skill normalization")
print("  - Project title cleaning")
print("  - Job title standardization") 
print("  - Test score validation")
print("  - Corrupted record removal")
print("✓ Step B: Label Balancing")
print("  - Distribution analysis")
print("  - Hybrid oversampling")
print("  - Class weight computation")

# Continue with the rest of the original pipeline (Steps 3-12)
print("\n=== Continuing with Enhanced Pipeline ===")

# Load required libraries for the rest of the pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set tensorflow random seed
tf.random.set_seed(SEED)

# Step 3 - Feature Engineering (Enhanced with cleaned data)
print("\n=== Step 3: Enhanced Feature Engineering ===")

def build_skill_vocabulary(resumes, domain_requirements):
    """Build comprehensive skill vocabulary from cleaned resumes and domain requirements"""
    all_skills = set()
    
    # Add skills from all resumes (already cleaned and normalized)
    for resume in resumes:
        all_skills.update(resume.get('skills', []))
    
    # Add required skills from all domains (normalize to match cleaning)
    for domain_data in domain_requirements.values():
        normalized_required = [skill.lower() for skill in domain_data.get('required_skills', [])]
        all_skills.update(normalized_required)
    
    skill_vocab = sorted(list(all_skills))
    return skill_vocab

def encode_skills(candidate_skills, skill_vocab):
    """Convert candidate skills list to binary vector"""
    skill_vector = np.zeros(len(skill_vocab), dtype=int)
    
    # Skills are already normalized during cleaning
    normalized_candidate_skills = set(candidate_skills)
    
    for i, vocab_skill in enumerate(skill_vocab):
        if vocab_skill in normalized_candidate_skills:
            skill_vector[i] = 1
            
    return skill_vector

def compute_skill_matches(candidate_skills, required_skills):
    """Compute matched and missing skills for a specific domain"""
    # Normalize required skills to match cleaned data
    candidate_set = set(candidate_skills)  # Already cleaned
    required_set = {skill.lower() for skill in required_skills}
    
    # Compute intersections
    matched_skills = list(candidate_set.intersection(required_set))
    missing_skills = list(required_set - candidate_set)
    
    # Calculate ratio
    skill_match_ratio = len(matched_skills) / len(required_set) if required_set else 0.0
    
    return matched_skills, missing_skills, skill_match_ratio

def extract_project_features(projects):
    """Extract features from cleaned projects list"""
    project_count = len(projects) if projects else 0
    project_text = " ".join(projects) if projects else ""
    return project_count, project_text

def extract_experience_features(work_experience):
    """Extract features from cleaned work experience"""
    if not work_experience:
        return 0.0, 0, ""
    
    # Total years of experience
    years_experience = sum(item.get('years', 0) for item in work_experience)
    
    # Maximum years in any single role
    max_years = max(item.get('years', 0) for item in work_experience)
    
    # Concatenate job titles for text features (already normalized)
    job_titles = [item.get('title', '') for item in work_experience]
    experience_text = " ".join(job_titles)
    
    return float(years_experience), max_years, experience_text

def normalize_test_score(test_score):
    """Normalize test score to [0,1] range (already cleaned)"""
    if test_score is None:
        return 0.0
    
    # Test scores are already clamped to [0,100] during cleaning
    test_score_norm = test_score / 100.0
    return test_score_norm

# Rest of the feature engineering classes remain the same...
class ResumeFeatureScaler:
    """Scaler for numeric resume features"""
    def __init__(self):
        self.project_scaler = StandardScaler()
        self.experience_scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, numeric_features):
        if len(numeric_features) == 0:
            return self
            
        numeric_array = np.array(numeric_features)
        
        if numeric_array.shape[1] >= 3:
            # Years experience and max years
            experience_data = numeric_array[:, :2].reshape(-1, 2)
            self.experience_scaler.fit(experience_data)
            
            # Project count
            project_data = numeric_array[:, 2:3].reshape(-1, 1)
            self.project_scaler.fit(project_data)
            
        self.is_fitted = True
        return self
    
    def transform(self, numeric_features):
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        if len(numeric_features) == 0:
            return np.array([])
            
        numeric_array = np.array(numeric_features)
        
        if numeric_array.ndim == 1:
            numeric_array = numeric_array.reshape(1, -1)
        
        scaled_features = []
        
        if numeric_array.shape[1] >= 3:
            # Scale experience features
            experience_data = numeric_array[:, :2]
            scaled_experience = self.experience_scaler.transform(experience_data)
            
            # Scale project features  
            project_data = numeric_array[:, 2:3]
            scaled_projects = self.project_scaler.transform(project_data)
            
            # Combine scaled features
            scaled_features = np.concatenate([scaled_experience, scaled_projects], axis=1)
        
        return scaled_features
    
    def fit_transform(self, numeric_features):
        return self.fit(numeric_features).transform(numeric_features)

def extract_all_features(resume, skill_vocab, domain_requirements):
    """Extract all features from a single resume (enhanced for cleaned data)"""
    # Get domain requirements
    domain_key = None
    for key, req in domain_requirements.items():
        if req["domain"] == resume["preferred_domain"]:
            domain_key = key
            break
    
    if domain_key is None:
        raise ValueError(f"No requirements found for domain {resume['preferred_domain']}")
    
    required_skills = domain_requirements[domain_key]["required_skills"]
    
    # Extract individual feature components
    candidate_skills = resume.get('skills', [])
    projects = resume.get('projects', [])
    work_experience = resume.get('work_experience', [])
    test_score = resume.get('test_score', 0)
    
    # Compute skill features
    skill_vector = encode_skills(candidate_skills, skill_vocab)
    matched_skills, missing_skills, skill_match_ratio = compute_skill_matches(
        candidate_skills, required_skills
    )
    
    # Compute project features
    project_count, project_text = extract_project_features(projects)
    
    # Compute experience features
    years_experience, max_years, experience_text = extract_experience_features(work_experience)
    
    # Normalize test score
    test_score_norm = normalize_test_score(test_score)
    
    # Combine numeric features for scaling
    numeric_features = [years_experience, max_years, project_count]
    
    # Return comprehensive feature dictionary
    return {
        'skill_vector': skill_vector,
        'skill_match_ratio': skill_match_ratio,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'project_count': project_count,
        'project_text': project_text,
        'years_experience': years_experience,
        'max_years': max_years,
        'experience_text': experience_text,
        'test_score': test_score,
        'test_score_norm': test_score_norm,
        'numeric_features': numeric_features,
        'domain': resume['preferred_domain'],
        'id': resume['id']
    }

# Apply enhanced feature engineering
print("Building enhanced skill vocabulary from cleaned data...")
skill_vocab = build_skill_vocabulary(balanced_resumes, domain_requirements)
print(f"Built skill vocabulary with {len(skill_vocab)} unique skills")

print("\nExtracting features from balanced resumes...")
all_features = []
for resume in balanced_resumes:
    try:
        features = extract_all_features(resume, skill_vocab, domain_requirements)
        features['label'] = resume['label']  # Add label for supervised learning
        all_features.append(features)
    except Exception as e:
        print(f"Error processing resume {resume.get('id', 'unknown')}: {e}")
        continue

print(f"Successfully extracted features from {len(all_features)} resumes")

# Prepare numeric features for scaling
numeric_feature_matrix = []
for features in all_features:
    numeric_feature_matrix.append(features['numeric_features'])

# Initialize and fit scaler
scaler = ResumeFeatureScaler()
scaler.fit(numeric_feature_matrix)

# Apply scaling to all features
for features in all_features:
    scaled_numeric = scaler.transform([features['numeric_features']])
    features['scaled_numeric_features'] = scaled_numeric[0]

# Continue with Steps 4-12 using the enhanced, cleaned, and balanced dataset...
print("\nEnhanced feature engineering completed with cleaned and balanced data!")
print("Ready to proceed with remaining steps (4-12) using improved dataset quality.")

# Step 4 - Final Feature Vector Construction (same as before but with cleaned data)
print("\n=== Step 4: Final Feature Vector Construction (Enhanced) ===")

class FeatureVectorBuilder:
    """Builds final feature vectors for model input using parallel branches"""
    
    def __init__(self, skill_vocab_size, use_text_embeddings=False, text_embedding_dim=128):
        self.skill_vocab_size = skill_vocab_size
        self.use_text_embeddings = use_text_embeddings
        self.text_embedding_dim = text_embedding_dim
        
        # Initialize text vectorizers (will be fitted on training data)
        self.project_vectorizer = TfidfVectorizer(
            max_features=64, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.experience_vectorizer = TfidfVectorizer(
            max_features=64,
            stop_words='english', 
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.is_fitted = False
        
    def fit_text_vectorizers(self, all_features):
        """Fit text vectorizers on training data"""
        # Extract all project and experience texts
        project_texts = []
        experience_texts = []
        
        for features in all_features:
            project_text = features.get('project_text', '')
            experience_text = features.get('experience_text', '')
            
            # Use placeholder if empty to avoid fitting issues
            project_texts.append(project_text if project_text else 'no projects')
            experience_texts.append(experience_text if experience_text else 'no experience')
        
        # Fit vectorizers
        self.project_vectorizer.fit(project_texts)
        self.experience_vectorizer.fit(experience_texts)
        
        self.is_fitted = True
        print("Text vectorizers fitted successfully")
    
    def build_skill_branch(self, features):
        """Branch 1: Skills branch - Returns: skill_vector (binary) + skill_match_ratio"""
        skill_vector = features['skill_vector']
        skill_match_ratio = np.array([features['skill_match_ratio']])
        
        # Combine skill vector with match ratio
        skill_branch = np.concatenate([skill_vector, skill_match_ratio])
        return skill_branch
    
    def build_numeric_branch(self, features):
        """Branch 2: Numeric branch"""
        test_score_norm = features['test_score_norm']
        skill_match_ratio = features['skill_match_ratio']
        
        # Get scaled numeric features
        scaled_features = features['scaled_numeric_features']
        years_experience_scaled = scaled_features[0]
        project_count_scaled = scaled_features[2]
        
        numeric_branch = np.array([
            test_score_norm,
            project_count_scaled, 
            years_experience_scaled,
            skill_match_ratio
        ])
        
        return numeric_branch
    
    def build_text_branch(self, features):
        """Branch 3: Text branch (optional)"""
        if not self.use_text_embeddings or not self.is_fitted:
            return np.array([])
        
        project_text = features.get('project_text', '')
        experience_text = features.get('experience_text', '')
        
        # Use placeholder if empty
        if not project_text:
            project_text = 'no projects'
        if not experience_text:
            experience_text = 'no experience'
        
        # Transform to TF-IDF vectors
        project_vector = self.project_vectorizer.transform([project_text]).toarray().flatten()
        experience_vector = self.experience_vectorizer.transform([experience_text]).toarray().flatten()
        
        # Combine text features
        text_branch = np.concatenate([project_vector, experience_vector])
        return text_branch
    
    def build_final_vector(self, features):
        """Concatenate all branches into final feature vector"""
        # Build individual branches
        skill_branch = self.build_skill_branch(features)
        numeric_branch = self.build_numeric_branch(features)
        
        # Start with skill and numeric branches
        branches = [skill_branch, numeric_branch]
        
        # Add text branch if enabled
        if self.use_text_embeddings and self.is_fitted:
            text_branch = self.build_text_branch(features)
            if len(text_branch) > 0:
                branches.append(text_branch)
        
        # Concatenate all branches
        final_vector = np.concatenate(branches)
        return final_vector
    
    def get_feature_dimensions(self):
        """Return dimensions of each branch and final vector"""
        skill_dim = self.skill_vocab_size + 1  # +1 for skill_match_ratio
        numeric_dim = 4  # [test_score_norm, project_count_scaled, years_experience_scaled, skill_match_ratio]
        text_dim = 128 if self.use_text_embeddings else 0  # 64 + 64 for project + experience TF-IDF
        
        final_dim = skill_dim + numeric_dim + text_dim
        
        return {
            'skill_branch_dim': skill_dim,
            'numeric_branch_dim': numeric_dim, 
            'text_branch_dim': text_dim,
            'final_vector_dim': final_dim
        }

# Initialize enhanced feature vector builder
skill_vocab_size = len(skill_vocab)
use_text_features = True

vector_builder = FeatureVectorBuilder(
    skill_vocab_size=skill_vocab_size,
    use_text_embeddings=use_text_features,
    text_embedding_dim=128
)

# Fit text vectorizers
if use_text_features:
    print("Fitting text vectorizers on enhanced data...")
    vector_builder.fit_text_vectorizers(all_features)

# Build feature vectors for all resumes
print("Building final feature vectors from enhanced data...")
feature_vectors = []
labels = []

for features in all_features:
    try:
        final_vector = vector_builder.build_final_vector(features)
        feature_vectors.append(final_vector)
        labels.append(features['label'])
    except Exception as e:
        print(f"Error building vector for {features.get('id', 'unknown')}: {e}")
        continue

# Convert to numpy arrays
X = np.array(feature_vectors)
y = np.array(labels)

print(f"Built feature vectors for {len(feature_vectors)} resumes (enhanced with cleaning & balancing)")

# Display enhanced feature vector dimensions
dimensions = vector_builder.get_feature_dimensions()
print(f"\n=== Enhanced Feature Vector Dimensions ===")
print(f"Total features: {dimensions['final_vector_dim']}")
print(f"Dataset shape: {X.shape}")
print(f"Feature sparsity: {np.mean(X == 0):.3f}")

# Save enhanced artifacts
print("\n=== Saving Enhanced Artifacts ===")
np.save('data/X_features_enhanced.npy', X)
np.save('data/y_labels_enhanced.npy', y)

import pickle
import joblib

# Save all enhanced preprocessing artifacts
joblib.dump(vector_builder, 'artifacts/enhanced_feature_vector_builder.pkl')
joblib.dump(scaler, 'artifacts/enhanced_feature_scaler.pkl')

with open('artifacts/enhanced_skill_vocabulary.json', 'w') as f:
    json.dump(skill_vocab, f, indent=2)

print("Enhanced preprocessing complete! Key improvements:")
print("✓ Skill normalization with embeddings (python3 → python)")
print("✓ Job title standardization (Sr. Data Scientist → data scientist)")  
print("✓ Project cleaning and stopword removal")
print("✓ Test score validation and clamping")
print("✓ Duplicate and corrupted record removal")
print("✓ Class balancing with hybrid oversampling")
print("✓ Clean, balanced dataset ready for model training")

print(f"\nDataset quality metrics:")
print(f"- Original: {len(synthetic_resumes)} resumes")  
print(f"- After cleaning: {len(cleaned_resumes)} resumes")
print(f"- After balancing: {len(balanced_resumes)} resumes")
print(f"- Final feature matrix: {X.shape}")