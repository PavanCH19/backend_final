import json
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
import keras
import sys
import types

# ============================================================================
# Path Setup - MUST BE FIRST
# ============================================================================

current_path = Path(__file__).resolve().parent

# Ensure project root is in sys.path
for parent in [current_path, *current_path.parents]:
    if (parent / "models").exists() and (parent / "artifacts").exists():
        BASE_PATH = parent
        break
else:
    raise FileNotFoundError("Could not locate project root (missing models/artifacts folders).")

MODELS_DIR = BASE_PATH / "models"
ARTIFACTS_DIR = BASE_PATH / "artifacts"
DATA_DIR = BASE_PATH / "data"

# Add src/ to sys.path
SRC_DIR = BASE_PATH / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ============================================================================
# Create Fake 'src' Package BEFORE Loading Pickles
# ============================================================================

if 'src.feature_engineering' not in sys.modules:
    try:
        import feature_engineering  # your actual file in src/
        src_module = types.ModuleType('src')
        src_module.feature_engineering = feature_engineering
        sys.modules['src'] = src_module
        sys.modules['src.feature_engineering'] = feature_engineering
        print("✓ Created src package mapping for pickle compatibility")
    except ImportError as e:
        print(f"Warning: Could not import feature_engineering: {e}")

# ============================================================================
# Load Pickled Artifacts (Now Safe)
# ============================================================================

scaler = joblib.load(ARTIFACTS_DIR / 'feature_scaler.pkl')
feature_builder = joblib.load(ARTIFACTS_DIR / 'feature_vector_builder.pkl')
label_encoder = joblib.load(ARTIFACTS_DIR / 'label_encoder.pkl')


# ============================================================================
# Helper Functions
# ============================================================================

def get_score_description(test_score):
    if test_score >= 85:
        return "Excellent"
    elif test_score >= 75:
        return "High"
    elif test_score >= 60:
        return "Good"
    elif test_score >= 50:
        return "Fair"
    else:
        return "Low"


def get_skills_description(skill_match_ratio, skills_fraction):
    if skill_match_ratio >= 0.8:
        return f"covers most required skills {skills_fraction}"
    elif skill_match_ratio >= 0.6:
        return f"covers many required skills {skills_fraction}"
    elif skill_match_ratio >= 0.4:
        return f"covers some required skills {skills_fraction}"
    else:
        return f"covers few required skills {skills_fraction}"


def get_experience_description(years_experience):
    if years_experience >= 3:
        return f"{int(years_experience)} years of solid experience"
    elif years_experience >= 1:
        year_text = "year" if years_experience == 1 else "years"
        return f"{int(years_experience)} {year_text} of experience"
    else:
        return "limited professional experience"


def get_project_description(project_count):
    if project_count >= 3:
        return f"strong portfolio ({project_count} projects)"
    elif project_count >= 1:
        project_text = "project" if project_count == 1 else "projects"
        return f"{project_count} {project_text}"
    else:
        return "no projects listed"


def get_recommendation(predicted_label, missing_skills):
    if predicted_label == "Partial Fit" and missing_skills:
        key_missing = missing_skills[:2]
        if key_missing:
            return f" Recommend gaining experience in {', '.join(key_missing)}."
    return ""


def get_domain_suggestion(predicted_label, alternative_domains):
    if predicted_label in ["Partial Fit", "Not Fit"] and alternative_domains:
        top_domain = alternative_domains[0]
        match_pct = int(top_domain['skill_match_ratio'] * 100)
        return (f" Consider applying for {top_domain['domain']} roles "
               f"({match_pct}% skill match with {top_domain['matched_skills_count']}"
               f"/{top_domain['required_skills_count']} required skills).")
    return ""


def generate_explanation(test_score, skill_match_ratio, matched_skills, 
                        missing_skills, project_count, years_experience, 
                        predicted_label, confidence, alternative_domains=None):
    score_desc = get_score_description(test_score)
    total_required = len(matched_skills) + len(missing_skills)
    skills_fraction = f"({len(matched_skills)}/{total_required} matched)"
    skills_desc = get_skills_description(skill_match_ratio, skills_fraction)
    missing_desc = f", but lacks {', '.join(missing_skills[:3])}" if missing_skills else ""
    exp_desc = get_experience_description(years_experience)
    proj_desc = get_project_description(project_count)
    recommendation = get_recommendation(predicted_label, missing_skills)
    domain_suggestion = get_domain_suggestion(predicted_label, alternative_domains)
    
    return (f"{score_desc} test score ({int(test_score)}/100) and {skills_desc}"
            f"{missing_desc}. Has {proj_desc} and {exp_desc}. "
            f"Model confidence: {confidence:.2f} → {predicted_label}."
            f"{recommendation}{domain_suggestion}")


def format_alternative_domains(suggestions, precision=3):
    if not suggestions:
        return None
    formatted = []
    for i, suggestion in enumerate(suggestions, 1):
        formatted.append({
            "rank": i,
            "domain": suggestion['domain'],
            "skill_match_ratio": round(suggestion['skill_match_ratio'], precision),
            "matched_skills_count": suggestion['matched_count'],
            "required_skills_count": suggestion['required_count'],
            "matched_skills": suggestion['matched_skills'][:5],
            "key_missing_skills": suggestion['missing_skills'][:3]
        })
    return formatted


def build_feature_summary(resume_features, include_raw_scores, precision):
    feature_summary = {
        "skill_match_ratio": round(resume_features['skill_match_ratio'], precision),
        "years_experience": int(resume_features['years_experience']),
        "test_score_norm": round(resume_features['test_score_norm'], precision),
        "project_count": int(resume_features['project_count'])
    }
    
    if include_raw_scores:
        feature_summary["test_score_raw"] = int(resume_features['test_score'])
    
    return feature_summary


def build_metadata(resume_features):
    return {
        "domain": resume_features['domain'],
        "candidate_id": resume_features['id'],
        "classification_timestamp": datetime.now().isoformat(),
        "model_version": "1.0"
    }


def build_error_result(resume_json, error_message):
    return {
        "error": f"Classification failed: {error_message}",
        "candidate_id": resume_json.get('id', 'unknown'),
        "timestamp": datetime.now().isoformat()
    }


def build_classification_result(label, confidence, resume_features, 
                                include_raw_scores, precision):
    feature_summary = build_feature_summary(resume_features, include_raw_scores, precision)
    
    alternative_domains = None
    if label in ["Partial Fit", "Not Fit", "Fit"]:
        alternative_domains = format_alternative_domains(
            resume_features.get('alternative_domains', []), precision
        )
    
    explanation = generate_explanation(
        resume_features['test_score'],
        resume_features['skill_match_ratio'],
        resume_features['matched_skills'],
        resume_features['missing_skills'],
        resume_features['project_count'],
        resume_features['years_experience'],
        label,
        confidence,
        alternative_domains
    )
    
    result = {
        "label": label,
        "confidence": confidence,
        "matched_skills": resume_features['matched_skills'],
        "missing_skills": resume_features['missing_skills'],
        "feature_summary": feature_summary,
        "explanation": explanation,
        "metadata": build_metadata(resume_features)
    }
    
    if alternative_domains:
        result["alternative_domain_suggestions"] = alternative_domains
    
    return result


def prepare_inputs(X, model, skill_vocab, use_text_branch=False):
    """Prepare inputs for model prediction"""
    skill_dim = len(skill_vocab) + 1
    numeric_dim = X.shape[1] - skill_dim
    
    if use_text_branch:
        numeric_dim -= model.input[-1].shape[1]
    
    skill_features = X[:, :skill_dim]
    numeric_features = X[:, skill_dim:skill_dim + numeric_dim]
    inputs = [skill_features, numeric_features]
    
    if use_text_branch:
        text_features = X[:, skill_dim + numeric_dim:]
        inputs.append(text_features)
    
    return inputs


def classify_resume(resume_json, model, feature_builder, label_encoder, 
                   skill_vocab, domain_requirements, scaler, use_text_branch=False,
                   include_raw_scores=True, precision=3):
    """Classify a single resume and return structured JSON output"""
    try:
        from feature_engineering import extract_all_features
        
        resume_features = extract_all_features(
            resume_json, skill_vocab, domain_requirements
        )
        
        scaled_numeric = scaler.transform([resume_features['numeric_features']])
        resume_features['scaled_numeric_features'] = scaled_numeric[0]
        
        final_vector = feature_builder.build_final_vector(resume_features)
        model_inputs = prepare_inputs(final_vector.reshape(1, -1), model, skill_vocab, use_text_branch)
        class_probs = model.predict(model_inputs, verbose=0)[0]
        
        pred_idx = np.argmax(class_probs)
        label = label_encoder.classes_[pred_idx]
        confidence = round(float(class_probs[pred_idx]), precision)
        
        result = build_classification_result(
            label, confidence, resume_features, include_raw_scores, precision
        )
        return result
    
    except Exception as e:
        return build_error_result(resume_json, str(e))


def save_results(results, output_file):
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results → {output_file}")

# ============================================================================
# Pipeline Loading Functions
# ============================================================================

def get_required_artifact_files():
    return [
        MODELS_DIR / 'resume_classifier_complete.h5',
        ARTIFACTS_DIR / 'feature_scaler.pkl',
        ARTIFACTS_DIR / 'label_encoder.pkl',
        ARTIFACTS_DIR / 'feature_vector_builder.pkl',
        ARTIFACTS_DIR / 'skill_vocabulary.json',
        ARTIFACTS_DIR / 'domain_requirements.json',
        ARTIFACTS_DIR / 'model_config.json'
    ]


def check_required_files(required_files):
    missing = [str(f) for f in required_files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts, run training first: {missing}")
    return True


def load_model_artifacts():
    # Load model with safe_mode=False to allow Lambda layers
    model = keras.models.load_model(
        MODELS_DIR / 'resume_classifier_complete.h5',
        safe_mode=False
    )
    # scaler, label_encoder, feature_builder already loaded at module level
    return model, scaler, label_encoder, feature_builder


def load_json_artifacts():
    with open(ARTIFACTS_DIR / 'skill_vocabulary.json', 'r') as f:
        skill_vocab = json.load(f)
    with open(ARTIFACTS_DIR / 'domain_requirements.json', 'r') as f:
        domain_requirements = json.load(f)
    with open(ARTIFACTS_DIR / 'model_config.json', 'r') as f:
        model_config = json.load(f)
    return skill_vocab, domain_requirements, model_config


# ============================================================================
# Sample Resume Functions
# ============================================================================

def get_sample_resumes():
    return [
        # -------------------- WEB / FULL STACK --------------------
        {
            "skills": [
                "JavaScript", "React", "Node.js", "HTML", "CSS", "MongoDB", "Express",
                "Python", "SQL", "Docker", "AWS", "CI/CD"
            ],
            "projects": ["Social Media Platform", "E-commerce Website"],
            "work_experience": [
                {"title": "Full Stack Developer", "years": 5}
            ],
            "test_score": 48,
            "preferred_domain": "Web Development",
            "id": "candidate_1250"
        },

        # -------------------- MOBILE DEVELOPMENT --------------------
        {
            "skills": [
                "Flutter", "Kotlin", "Java", "React Native", "Swift",
                "iOS", "Android", "Firebase", "AWS"
            ],
            "projects": ["Mobile Banking App", "Fitness Tracker"],
            "work_experience": [
                {"title": "Mobile Developer", "years": 4}
            ],
            "test_score": 82,
            "preferred_domain": "Mobile Development",
            "id": "candidate_1251"
        },

        # -------------------- DEVOPS --------------------
        {
            "skills": [
                "Docker", "Kubernetes", "AWS", "Jenkins", "Terraform",
                "Linux", "CI/CD", "Python"
            ],
            "projects": ["Infrastructure as Code", "Monitoring Dashboard"],
            "work_experience": [
                {"title": "DevOps Engineer", "years": 6}
            ],
            "test_score": 85,
            "preferred_domain": "DevOps",
            "id": "candidate_1252"
        },

        # -------------------- CYBERSECURITY --------------------
        {
            "skills": [
                "Network Security", "Penetration Testing", "CISSP",
                "Firewall", "Encryption", "Python", "AWS"
            ],
            "projects": ["Vulnerability Assessment", "Security Information Dashboard"],
            "work_experience": [
                {"title": "Cybersecurity Engineer", "years": 5}
            ],
            "test_score": 79,
            "preferred_domain": "Cybersecurity",
            "id": "candidate_1253"
        },

        # -------------------- DATA SCIENCE --------------------
        {
            "skills": [
                "Python", "Pandas", "NumPy", "Scikit-learn",
                "TensorFlow", "SQL", "Statistics"
            ],
            "projects": ["Customer Churn Prediction", "Sales Forecasting Model"],
            "work_experience": [
                {"title": "Data Scientist", "years": 3}
            ],
            "test_score": 88,
            "preferred_domain": "Data Science",
            "id": "candidate_1254"
        },

        # -------------------- AI / ML --------------------
        {
            "skills": [
                "Python", "TensorFlow", "PyTorch",
                "Deep Learning", "NLP", "Computer Vision"
            ],
            "projects": ["Chatbot with NLP", "Face Recognition System"],
            "work_experience": [
                {"title": "ML Engineer", "years": 4}
            ],
            "test_score": 91,
            "preferred_domain": "AI & Machine Learning",
            "id": "candidate_1255"
        },

        # -------------------- CLOUD COMPUTING --------------------
        {
            "skills": [
                "AWS", "Azure", "Docker", "Kubernetes",
                "Terraform", "IAM"
            ],
            "projects": ["Serverless Web App", "Cloud Cost Optimizer"],
            "work_experience": [
                {"title": "Cloud Engineer", "years": 5}
            ],
            "test_score": 84,
            "preferred_domain": "Cloud Computing",
            "id": "candidate_1256"
        },

        # -------------------- BLOCKCHAIN --------------------
        {
            "skills": [
                "Solidity", "Ethereum", "Smart Contracts",
                "Web3.js", "Cryptography"
            ],
            "projects": ["NFT Marketplace", "DeFi Platform"],
            "work_experience": [
                {"title": "Blockchain Developer", "years": 3}
            ],
            "test_score": 78,
            "preferred_domain": "Blockchain Development",
            "id": "candidate_1257"
        },

        # -------------------- GAME DEVELOPMENT --------------------
        {
            "skills": [
                "Unity", "C#", "Game Physics",
                "3D Modeling", "Shader Programming"
            ],
            "projects": ["2D Platformer", "3D Adventure Game"],
            "work_experience": [
                {"title": "Game Developer", "years": 4}
            ],
            "test_score": 73,
            "preferred_domain": "Game Development",
            "id": "candidate_1258"
        },

        # -------------------- EMBEDDED SYSTEMS --------------------
        {
            "skills": [
                "C", "C++", "Microcontrollers",
                "RTOS", "IoT", "Sensors"
            ],
            "projects": ["Smart Home Automation", "IoT Weather Station"],
            "work_experience": [
                {"title": "Embedded Engineer", "years": 5}
            ],
            "test_score": 80,
            "preferred_domain": "Embedded Systems",
            "id": "candidate_1259"
        },

        # -------------------- AR / VR --------------------
        {
            "skills": [
                "Unity", "C#", "XR Toolkit",
                "3D Modeling", "Oculus SDK"
            ],
            "projects": ["Virtual Museum", "VR Training Simulator"],
            "work_experience": [
                {"title": "AR/VR Developer", "years": 3}
            ],
            "test_score": 77,
            "preferred_domain": "AR / VR Development",
            "id": "candidate_1260"
        },

        # -------------------- UI / UX --------------------
        {
            "skills": [
                "Figma", "Wireframing", "Prototyping",
                "User Research", "Accessibility"
            ],
            "projects": ["Design System", "Mobile App Redesign"],
            "work_experience": [
                {"title": "UX Designer", "years": 4}
            ],
            "test_score": 74,
            "preferred_domain": "UI / UX Design",
            "id": "candidate_1261"
        },

        # ==================== LANGUAGE / FRAMEWORK DOMAINS ====================

        # -------------------- JAVA --------------------
        {
            "skills": [
                "Java", "Spring Boot", "Hibernate",
                "JPA", "Microservices", "REST APIs"
            ],
            "projects": ["Banking Management System", "Spring Boot REST API"],
            "work_experience": [
                {"title": "Java Developer", "years": 5}
            ],
            "test_score": 83,
            "preferred_domain": "Java Development",
            "id": "candidate_1262"
        },

        # -------------------- PYTHON --------------------
        {
            "skills": [
                "Python", "Django", "Flask",
                "FastAPI", "AsyncIO", "REST APIs"
            ],
            "projects": ["FastAPI Backend", "Automation Tool"],
            "work_experience": [
                {"title": "Python Developer", "years": 4}
            ],
            "test_score": 86,
            "preferred_domain": "Python Development",
            "id": "candidate_1263"
        },

        # -------------------- NODE.JS --------------------
        {
            "skills": [
                "Node.js", "Express", "MongoDB",
                "JWT", "Socket.io", "REST APIs"
            ],
            "projects": ["Authentication API", "Real-time Chat Server"],
            "work_experience": [
                {"title": "Node.js Developer", "years": 4}
            ],
            "test_score": 81,
            "preferred_domain": "Node.js Development",
            "id": "candidate_1264"
        },

        # -------------------- JAVASCRIPT --------------------
        {
            "skills": [
                "JavaScript", "ES6+", "DOM",
                "Promises", "Event Loop"
            ],
            "projects": ["Interactive Dashboard", "Browser Game"],
            "work_experience": [
                {"title": "JavaScript Developer", "years": 3}
            ],
            "test_score": 75,
            "preferred_domain": "JavaScript Development",
            "id": "candidate_1265"
        },

        # -------------------- REACT --------------------
        {
            "skills": [
                "React", "Hooks", "Redux",
                "Next.js", "Performance Optimization"
            ],
            "projects": ["Admin Dashboard", "E-commerce Frontend"],
            "work_experience": [
                {"title": "React Developer", "years": 4}
            ],
            "test_score": 87,
            "preferred_domain": "React Development",
            "id": "candidate_1266"
        }
    ]

# ============================================================================
# Main Pipeline Function
# ============================================================================

def main(data):
    """
    Main function called by Node.js bridge
    Args:
        data: Dictionary containing resume information
    Returns:
        Dictionary with classification results
    """
    print("\n=== Resume Classification Pipeline ===")
    
    required_files = get_required_artifact_files()
    check_required_files(required_files)
    
    model, scaler, label_encoder, feature_builder = load_model_artifacts()
    skill_vocab, domain_requirements, model_config = load_json_artifacts()
    use_text_branch = model_config.get("use_text_branch", False)
    
    print("✓ Pipeline loaded from saved artifacts")
    
    # Process the resume data received from Node.js
    results = classify_resume(
        data, 
        model, 
        feature_builder, 
        label_encoder,
        skill_vocab, 
        domain_requirements, 
        scaler, 
        use_text_branch
    )
    
    print("\n✓ Classification Complete")
    
    # Return only the results (bridge.py will wrap this in success response)
    return results


# # Keep your existing if __name__ == "__main__" for local testing
# if __name__ == "__main__":
#     print(f"Base path set to: {BASE_PATH}")
#     sample_resumes = get_sample_resumes()
    
#     # For local testing, process all sample
#     for i, resume in enumerate(sample_resumes):
#         print(f"\n--- Classifying Sample Resume {i+1}/{len(sample_resumes)} ---")
#         result = main(resume)
#         print(json.dumps(result, indent=2))

from data_config import get_domain_requirements

def get_all_expected_domains():
    return set(
        d["domain"]
        for d in get_domain_requirements().values()
    )
def extract_predicted_domain(result: dict):
    """
    Safely extract predicted domain from classification result
    """
    if not isinstance(result, dict):
        return None

    # CURRENT OFFICIAL LOCATION (your pipeline output)
    if "metadata" in result and isinstance(result["metadata"], dict):
        domain = result["metadata"].get("domain")
        if domain:
            return domain

    # Fallbacks (future-proofing)
    for key in ["predicted_domain", "best_domain", "domain"]:
        if key in result:
            return result[key]

    return None


    
if __name__ == "__main__":
    print(f"Base path set to: {BASE_PATH}")

    sample_resumes = get_sample_resumes()
    expected_domains = get_all_expected_domains()
    detected_domains = set()
    failed_resumes = []

    for i, resume in enumerate(sample_resumes):
        print(f"\n--- Classifying Sample Resume {i+1}/{len(sample_resumes)} ---")

        try:
            result = main(resume)
            print(json.dumps(result, indent=2))

            predicted_domain = extract_predicted_domain(result)

            if predicted_domain:
                detected_domains.add(predicted_domain)
            else:
                print("❌ Domain not found in result structure")
                failed_resumes.append(resume["id"])

        except Exception as e:
            print(f"❌ Error while processing {resume['id']}: {str(e)}")
            failed_resumes.append(resume["id"])

    # ---------------- SUMMARY ----------------
    print("\n================ DOMAIN COVERAGE REPORT ================")

    print(f"Expected Domains ({len(expected_domains)}):")
    for d in sorted(expected_domains):
        print(f"  ✔ {d}")

    print(f"\nDetected Domains ({len(detected_domains)}):")
    for d in sorted(detected_domains):
        print(f"  ✅ {d}")

    missing_domains = expected_domains - detected_domains

    if missing_domains:
        print("\n❌ MISSING DOMAINS (No resume classified into these):")
        for d in sorted(missing_domains):
            print(f"  ❌ {d}")
    else:
        print("\n🎉 SUCCESS: All domains classified at least once!")

    if failed_resumes:
        print("\n⚠ Failed Resume IDs:")
        for r in failed_resumes:
            print(f"  - {r}")
    else:
        print("\n✅ All resumes processed successfully")