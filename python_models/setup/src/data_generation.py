import json
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data_config import get_domain_requirements, get_skill_pools, get_project_templates, get_job_titles

def get_skill_key_mapping():
    """Return mapping between domain names and skill pool keys"""
    return {
        "data_science": "data_science",
        "web_development": "web_dev", 
        "mobile_development": "mobile",
        "devops": "devops",
        "cybersecurity": "security"
    }


def save_domain_requirements(domain_requirements, filepath='data/domain_requirements.json'):
    """Save domain requirements to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(domain_requirements, f, indent=2)
    print(f"Saved → {filepath}")


def generate_fit_resume_skills(required_skills, domain_skills):
    """Generate skills for a 'fit' candidate"""
    required_count = max(1, int(len(required_skills) * random.uniform(0.7, 1.0)))
    selected_required = random.sample(required_skills, required_count)
    additional_domain = random.sample([s for s in domain_skills if s not in selected_required], random.randint(2, 5))
    selected_skills = selected_required + additional_domain
    test_score = int(random.uniform(75, 100))
    n_projects = random.randint(1, 4)
    
    return selected_skills, test_score, n_projects


def generate_partial_fit_resume_skills(required_skills, domain_skills):
    """Generate skills for a 'partial_fit' candidate"""
    if random.choice([True, False]):
        required_count = max(1, int(len(required_skills) * random.uniform(0.4, 0.69)))
        selected_required = random.sample(required_skills, required_count)
        additional_domain = random.sample([s for s in domain_skills if s not in selected_required], random.randint(1, 3))
        selected_skills = selected_required + additional_domain
        test_score = int(random.uniform(30, 100))
    else:
        required_count = max(0, int(len(required_skills) * random.uniform(0.0, 1.0)))
        selected_required = random.sample(required_skills, required_count) if required_count > 0 else []
        additional_domain = random.sample([s for s in domain_skills if s not in selected_required], random.randint(1, 4))
        selected_skills = selected_required + additional_domain
        test_score = int(random.uniform(50, 74))
    
    n_projects = random.randint(0, 3)
    return selected_skills, test_score, n_projects


def generate_not_fit_resume_skills(required_skills, domain_skills, other_skills):
    """Generate skills for a 'not_fit' candidate"""
    if random.choice([True, False]):
        required_count = max(0, int(len(required_skills) * random.uniform(0.0, 0.39)))
        selected_required = random.sample(required_skills, required_count) if required_count > 0 else []
        
        if random.choice([True, False]):
            selected_skills = selected_required + random.sample(other_skills, random.randint(2, 6))
        else:
            additional_domain = random.sample([s for s in domain_skills if s not in selected_required], random.randint(0, 2))
            selected_skills = selected_required + additional_domain
        test_score = int(random.uniform(0, 100))
    else:
        required_count = max(0, int(len(required_skills) * random.uniform(0.0, 1.0)))
        selected_required = random.sample(required_skills, required_count) if required_count > 0 else []
        additional_domain = random.sample([s for s in domain_skills if s not in selected_required], random.randint(1, 4))
        selected_skills = selected_required + additional_domain
        test_score = int(random.uniform(0, 49))
    
    n_projects = random.randint(0, 2)
    return selected_skills, test_score, n_projects


def generate_work_experience(job_titles_list, n_jobs):
    """Generate work experience entries"""
    return [{"title": random.choice(job_titles_list), "years": random.randint(1, 8)} for _ in range(n_jobs)]


def generate_resume_for_category(category, candidate_id, domains, domain_requirements, all_skills, 
                                 project_templates, job_titles, skill_key_mapping):
    """Generate a resume targeting a specific fit category"""
    
    preferred_domain_key = random.choice(domains)
    preferred_domain = domain_requirements[preferred_domain_key]["domain"]
    required_skills = domain_requirements[preferred_domain_key]["required_skills"]
    
    domain_skills = all_skills[skill_key_mapping[preferred_domain_key]]
    other_skills = list(set(sum(all_skills.values(), [])) - set(domain_skills))
    
    # Generate skills based on category
    if category == 'fit':
        selected_skills, test_score, n_projects = generate_fit_resume_skills(required_skills, domain_skills)
    elif category == 'partial_fit':
        selected_skills, test_score, n_projects = generate_partial_fit_resume_skills(required_skills, domain_skills)
    else:  # not_fit
        selected_skills, test_score, n_projects = generate_not_fit_resume_skills(required_skills, domain_skills, other_skills)
    
    # Add random other skills
    if len(selected_skills) < 8:
        selected_skills.extend(random.sample(other_skills, random.randint(0, 3)))
    
    # Generate projects
    domain_projects = project_templates[skill_key_mapping[preferred_domain_key]]
    selected_projects = random.sample(domain_projects, min(n_projects, len(domain_projects)))
    
    # Generate work experience
    n_jobs = random.randint(1, 4)
    domain_job_titles = job_titles[skill_key_mapping[preferred_domain_key]]
    work_experience = generate_work_experience(domain_job_titles, n_jobs)
    
    return {
        "skills": list(set(selected_skills)),
        "projects": selected_projects,
        "work_experience": work_experience,
        "test_score": test_score,
        "preferred_domain": preferred_domain,
        "id": f"candidate_{candidate_id:04d}"
    }


def generate_balanced_resumes(n_samples=2000, target_distribution=None, domain_requirements=None,
                              all_skills=None, project_templates=None, job_titles=None, skill_key_mapping=None):
    """Generate synthetic resume data with balanced labels"""
    
    if target_distribution is None:
        target_distribution = {'fit': 0.33, 'partial_fit': 0.34, 'not_fit': 0.33}
    
    if domain_requirements is None:
        domain_requirements = get_domain_requirements()
    if all_skills is None:
        all_skills = get_skill_pools()
    if project_templates is None:
        project_templates = get_project_templates()
    if job_titles is None:
        job_titles = get_job_titles()
    if skill_key_mapping is None:
        skill_key_mapping = get_skill_key_mapping()
    
    n_fit = int(n_samples * target_distribution['fit'])
    n_partial = int(n_samples * target_distribution['partial_fit'])
    n_not_fit = n_samples - n_fit - n_partial
    
    domains = list(domain_requirements.keys())
    resumes = []
    candidate_id = 1
    
    for _ in range(n_fit):
        resumes.append(generate_resume_for_category('fit', candidate_id, domains, domain_requirements, 
                                                    all_skills, project_templates, job_titles, skill_key_mapping))
        candidate_id += 1
    
    for _ in range(n_partial):
        resumes.append(generate_resume_for_category('partial_fit', candidate_id, domains, domain_requirements,
                                                    all_skills, project_templates, job_titles, skill_key_mapping))
        candidate_id += 1
    
    for _ in range(n_not_fit):
        resumes.append(generate_resume_for_category('not_fit', candidate_id, domains, domain_requirements,
                                                    all_skills, project_templates, job_titles, skill_key_mapping))
        candidate_id += 1
    
    random.shuffle(resumes)
    return resumes


def save_resumes_to_json(resumes, filepath='data/synthetic_resumes.json'):
    """Save generated resumes to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(resumes, f, indent=2)
    print(f"Saved → {filepath}")


# def main():
#     """Main function to execute the balanced dataset generation"""
#     print("=== Step 1: Balanced Dataset Generation ===\n")
    
#     # Get domain requirements and save
#     domain_requirements = get_domain_requirements()
#     save_domain_requirements(domain_requirements)
    
#     # Generate and save balanced resumes
#     synthetic_resumes = generate_balanced_resumes(n_samples=2000)
#     save_resumes_to_json(synthetic_resumes)
    
#     print(f"\nGenerated {len(synthetic_resumes)} synthetic resumes")


# if __name__ == "__main__":
#     main()