# import json
# import joblib
# import numpy as np
# from datetime import datetime
# from pathlib import Path
# import keras
# import sys
# import types

# # ============================================================================
# # Path Setup - MUST BE FIRST
# # ============================================================================

# current_path = Path(__file__).resolve().parent

# # Ensure project root is in sys.path
# for parent in [current_path, *current_path.parents]:
#     if (parent / "models").exists() and (parent / "artifacts").exists():
#         BASE_PATH = parent
#         break
# else:
#     raise FileNotFoundError("Could not locate project root (missing models/artifacts folders).")

# MODELS_DIR = BASE_PATH / "models"
# ARTIFACTS_DIR = BASE_PATH / "artifacts"
# DATA_DIR = BASE_PATH / "data"

# # Add src/ to sys.path
# SRC_DIR = BASE_PATH / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# # ============================================================================
# # Create Fake 'src' Package BEFORE Loading Pickles
# # ============================================================================

# if 'src.feature_engineering' not in sys.modules:
#     try:
#         import feature_engineering  # your actual file in src/
#         src_module = types.ModuleType('src')
#         src_module.feature_engineering = feature_engineering
#         sys.modules['src'] = src_module
#         sys.modules['src.feature_engineering'] = feature_engineering
#         print("✓ Created src package mapping for pickle compatibility")
#     except ImportError as e:
#         print(f"Warning: Could not import feature_engineering: {e}")

# # ============================================================================
# # Load Pickled Artifacts (Now Safe)
# # ============================================================================

# scaler = joblib.load(ARTIFACTS_DIR / 'feature_scaler.pkl')
# feature_builder = joblib.load(ARTIFACTS_DIR / 'feature_vector_builder.pkl')
# label_encoder = joblib.load(ARTIFACTS_DIR / 'label_encoder.pkl')


# # ============================================================================
# # Pipeline Class
# # ============================================================================

# class ResumeClassificationPipeline:
#     """Complete pipeline for resume classification with JSON output"""
    
#     def __init__(self, model, feature_builder, label_encoder, skill_vocab, 
#                  domain_requirements, scaler, use_text_branch=False):
#         self.model = model
#         self.feature_builder = feature_builder
#         self.label_encoder = label_encoder
#         self.skill_vocab = skill_vocab
#         self.domain_requirements = domain_requirements
#         self.scaler = scaler
#         self.use_text_branch = use_text_branch
    
#     def prepare_inputs(self, X):
#         """Prepare inputs for model prediction"""
#         skill_dim = len(self.skill_vocab) + 1
#         numeric_dim = X.shape[1] - skill_dim
        
#         if self.use_text_branch:
#             numeric_dim -= self.model.input[-1].shape[1]
        
#         skill_features = X[:, :skill_dim]
#         numeric_features = X[:, skill_dim:skill_dim + numeric_dim]
#         inputs = [skill_features, numeric_features]
        
#         if self.use_text_branch:
#             text_features = X[:, skill_dim + numeric_dim:]
#             inputs.append(text_features)
        
#         return inputs
    
#     def classify_resume(self, resume_json, include_raw_scores=True, precision=3):
#         """Classify a single resume and return structured JSON output"""
#         try:
#             from feature_engineering import extract_all_features
            
#             resume_features = extract_all_features(
#                 resume_json, self.skill_vocab, self.domain_requirements
#             )
            
#             scaled_numeric = self.scaler.transform([resume_features['numeric_features']])
#             resume_features['scaled_numeric_features'] = scaled_numeric[0]
            
#             final_vector = self.feature_builder.build_final_vector(resume_features)
#             model_inputs = self.prepare_inputs(final_vector.reshape(1, -1))
#             class_probs = self.model.predict(model_inputs, verbose=0)[0]
            
#             pred_idx = np.argmax(class_probs)
#             label = self.label_encoder.classes_[pred_idx]
#             confidence = round(float(class_probs[pred_idx]), precision)
            
#             result = self._build_classification_result(
#                 label, confidence, resume_features, include_raw_scores, precision
#             )
#             return result
        
#         except Exception as e:
#             return self._build_error_result(resume_json, str(e))
    
#     def _build_classification_result(self, label, confidence, resume_features, 
#                                     include_raw_scores, precision):
#         feature_summary = self._build_feature_summary(
#             resume_features, include_raw_scores, precision
#         )
        
#         alternative_domains = None
#         if label in ["Partial Fit", "Not Fit", "Fit"]:
#             alternative_domains = self._format_alternative_domains(
#                 resume_features.get('alternative_domains', []), precision
#             )
        
#         explanation = self._generate_explanation(
#             resume_features['test_score'],
#             resume_features['skill_match_ratio'],
#             resume_features['matched_skills'],
#             resume_features['missing_skills'],
#             resume_features['project_count'],
#             resume_features['years_experience'],
#             label,
#             confidence,
#             alternative_domains
#         )
        
#         result = {
#             "label": label,
#             "confidence": confidence,
#             "matched_skills": resume_features['matched_skills'],
#             "missing_skills": resume_features['missing_skills'],
#             "feature_summary": feature_summary,
#             "explanation": explanation,
#             "metadata": self._build_metadata(resume_features)
#         }
        
#         if alternative_domains:
#             result["alternative_domain_suggestions"] = alternative_domains
        
#         return result
    
#     def _build_feature_summary(self, resume_features, include_raw_scores, precision):
#         feature_summary = {
#             "skill_match_ratio": round(resume_features['skill_match_ratio'], precision),
#             "years_experience": int(resume_features['years_experience']),
#             "test_score_norm": round(resume_features['test_score_norm'], precision),
#             "project_count": int(resume_features['project_count'])
#         }
        
#         if include_raw_scores:
#             feature_summary["test_score_raw"] = int(resume_features['test_score'])
        
#         return feature_summary
    
#     def _build_metadata(self, resume_features):
#         return {
#             "domain": resume_features['domain'],
#             "candidate_id": resume_features['id'],
#             "classification_timestamp": datetime.now().isoformat(),
#             "model_version": "1.0"
#         }
    
#     def _build_error_result(self, resume_json, error_message):
#         return {
#             "error": f"Classification failed: {error_message}",
#             "candidate_id": resume_json.get('id', 'unknown'),
#             "timestamp": datetime.now().isoformat()
#         }
    
#     def _format_alternative_domains(self, suggestions, precision=3):
#         if not suggestions:
#             return None
#         formatted = []
#         for i, suggestion in enumerate(suggestions, 1):
#             formatted.append({
#                 "rank": i,
#                 "domain": suggestion['domain'],
#                 "skill_match_ratio": round(suggestion['skill_match_ratio'], precision),
#                 "matched_skills_count": suggestion['matched_count'],
#                 "required_skills_count": suggestion['required_count'],
#                 "matched_skills": suggestion['matched_skills'][:5],
#                 "key_missing_skills": suggestion['missing_skills'][:3]
#             })
#         return formatted
    
#     def _generate_explanation(self, test_score, skill_match_ratio, matched_skills, 
#                              missing_skills, project_count, years_experience, 
#                              predicted_label, confidence, alternative_domains=None):
#         score_desc = self._get_score_description(test_score)
#         total_required = len(matched_skills) + len(missing_skills)
#         skills_fraction = f"({len(matched_skills)}/{total_required} matched)"
#         skills_desc = self._get_skills_description(skill_match_ratio, skills_fraction)
#         missing_desc = f", but lacks {', '.join(missing_skills[:3])}" if missing_skills else ""
#         exp_desc = self._get_experience_description(years_experience)
#         proj_desc = self._get_project_description(project_count)
#         recommendation = self._get_recommendation(predicted_label, missing_skills)
#         domain_suggestion = self._get_domain_suggestion(predicted_label, alternative_domains)
        
#         return (f"{score_desc} test score ({int(test_score)}/100) and {skills_desc}"
#                 f"{missing_desc}. Has {proj_desc} and {exp_desc}. "
#                 f"Model confidence: {confidence:.2f} → {predicted_label}."
#                 f"{recommendation}{domain_suggestion}")
    
#     def _get_score_description(self, test_score):
#         if test_score >= 85: return "Excellent"
#         elif test_score >= 75: return "High"
#         elif test_score >= 60: return "Good"
#         elif test_score >= 50: return "Fair"
#         else: return "Low"
    
#     def _get_skills_description(self, skill_match_ratio, skills_fraction):
#         if skill_match_ratio >= 0.8: return f"covers most required skills {skills_fraction}"
#         elif skill_match_ratio >= 0.6: return f"covers many required skills {skills_fraction}"
#         elif skill_match_ratio >= 0.4: return f"covers some required skills {skills_fraction}"
#         else: return f"covers few required skills {skills_fraction}"
    
#     def _get_experience_description(self, years_experience):
#         if years_experience >= 3: return f"{int(years_experience)} years of solid experience"
#         elif years_experience >= 1:
#             year_text = "year" if years_experience == 1 else "years"
#             return f"{int(years_experience)} {year_text} of experience"
#         else: return "limited professional experience"
    
#     def _get_project_description(self, project_count):
#         if project_count >= 3: return f"strong portfolio ({project_count} projects)"
#         elif project_count >= 1:
#             project_text = "project" if project_count == 1 else "projects"
#             return f"{project_count} {project_text}"
#         else: return "no projects listed"
    
#     def _get_recommendation(self, predicted_label, missing_skills):
#         if predicted_label == "Partial Fit" and missing_skills:
#             key_missing = missing_skills[:2]
#             if key_missing:
#                 return f" Recommend gaining experience in {', '.join(key_missing)}."
#         return ""
    
#     def _get_domain_suggestion(self, predicted_label, alternative_domains):
#         if predicted_label in ["Partial Fit", "Not Fit"] and alternative_domains:
#             top_domain = alternative_domains[0]
#             match_pct = int(top_domain['skill_match_ratio'] * 100)
#             return (f" Consider applying for {top_domain['domain']} roles "
#                    f"({match_pct}% skill match with {top_domain['matched_skills_count']}"
#                    f"/{top_domain['required_skills_count']} required skills).")
#         return ""
    
#     def batch_classify(self, resume_list, output_file=None):
#         results = [self.classify_resume(resume) for resume in resume_list]
#         if output_file:
#             self._save_results(results, output_file)
#         return results
    
#     def _save_results(self, results, output_file):
#         Path(output_file).parent.mkdir(exist_ok=True, parents=True)
#         with open(output_file, 'w') as f:
#             json.dump(results, f, indent=2)
#         print(f"Saved {len(results)} results → {output_file}")


# # ============================================================================
# # Pipeline Loading Functions
# # ============================================================================

# def get_required_artifact_files():
#     return [
#         MODELS_DIR / 'resume_classifier_complete.h5',
#         ARTIFACTS_DIR / 'feature_scaler.pkl',
#         ARTIFACTS_DIR / 'label_encoder.pkl',
#         ARTIFACTS_DIR / 'feature_vector_builder.pkl',
#         ARTIFACTS_DIR / 'skill_vocabulary.json',
#         ARTIFACTS_DIR / 'domain_requirements.json',
#         ARTIFACTS_DIR / 'model_config.json'
#     ]


# def check_required_files(required_files):
#     missing = [str(f) for f in required_files if not Path(f).exists()]
#     if missing:
#         raise FileNotFoundError(f"Missing artifacts, run training first: {missing}")
#     return True


# def load_model_artifacts():
#     model = keras.models.load_model(MODELS_DIR / 'resume_classifier_complete.h5')
#     # scaler, label_encoder, feature_builder already loaded at module level
#     return model, scaler, label_encoder, feature_builder


# def load_json_artifacts():
#     with open(ARTIFACTS_DIR / 'skill_vocabulary.json', 'r') as f:
#         skill_vocab = json.load(f)
#     with open(ARTIFACTS_DIR / 'domain_requirements.json', 'r') as f:
#         domain_requirements = json.load(f)
#     with open(ARTIFACTS_DIR / 'model_config.json', 'r') as f:
#         model_config = json.load(f)
#     return skill_vocab, domain_requirements, model_config


# def load_classification_pipeline():
#     model, scaler, label_encoder, feature_builder = load_model_artifacts()
#     skill_vocab, domain_requirements, model_config = load_json_artifacts()
    
#     pipeline = ResumeClassificationPipeline(
#         model=model,
#         feature_builder=feature_builder,
#         label_encoder=label_encoder,
#         skill_vocab=skill_vocab,
#         domain_requirements=domain_requirements,
#         scaler=scaler,
#         use_text_branch=model_config.get("use_text_branch", False)
#     )
#     return pipeline


# # ============================================================================
# # Sample Resume Functions
# # ============================================================================

# def get_sample_resumes():
#     return [
#         {
#             "skills": [
#                 "JavaScript", "React", "Node.js", "HTML", "CSS", "MongoDB", "Express",
#                 "Python", "SQL", "Docker", "AWS", "CI/CD", "Kubernetes", "Terraform",
#                 "Penetration Testing", "Network Security"
#             ],
#             "projects": ["Social Media Platform", "E-commerce Website", "CI/CD Pipeline Setup"],
#             "work_experience": [
#                 {"title": "Full Stack Developer", "years": 5},
#                 {"title": "DevOps Engineer", "years": 2}
#             ],
#             "test_score": 78,
#             "preferred_domain": "Web Development",
#             "id": "candidate_1250"
#         },
#         {
#             "skills": [
#                 "Flutter", "Kotlin", "Java", "React Native", "Swift", "iOS", "Android",
#                 "Python", "SQL", "Docker", "React.js", "Tailwind CSS", "AWS", "Terraform"
#             ],
#             "projects": ["Mobile Banking App", "Fitness Tracker", "Expense Tracker"],
#             "work_experience": [
#                 {"title": "Mobile Developer", "years": 4},
#                 {"title": "Data Analyst", "years": 1}
#             ],
#             "test_score": 82,
#             "preferred_domain": "Mobile Development",
#             "id": "candidate_1251"
#         },
#         {
#             "skills": [
#                 "Docker", "Kubernetes", "AWS", "Jenkins", "Terraform", "Linux", "CI/CD",
#                 "Python", "React.js", "Node.js", "SQL", "Flutter", "Penetration Testing"
#             ],
#             "projects": ["Infrastructure as Code", "Automated Deployment", "Monitoring Dashboard"],
#             "work_experience": [
#                 {"title": "DevOps Engineer", "years": 6},
#                 {"title": "Full Stack Developer", "years": 2}
#             ],
#             "test_score": 85,
#             "preferred_domain": "DevOps",
#             "id": "candidate_1252"
#         },
#         {
#             "skills": [
#                 "Network Security", "Penetration Testing", "CISSP", "Firewall", "Encryption",
#                 "Python", "Docker", "AWS", "React.js", "SQL", "Kubernetes"
#             ],
#             "projects": ["Vulnerability Assessment", "Incident Response Plan", "Security Information Dashboard"],
#             "work_experience": [
#                 {"title": "Cybersecurity Engineer", "years": 5},
#                 {"title": "DevOps Engineer", "years": 1}
#             ],
#             "test_score": 79,
#             "preferred_domain": "Cybersecurity",
#             "id": "candidate_1253"
#         },
#         {
#             "skills": [
#                 "JavaScript", "Vue.js", "HTML", "CSS", "Node.js", "Express", "React",
#                 "Docker", "Python", "AWS", "Kubernetes", "Flutter", "Penetration Testing"
#             ],
#             "projects": ["Task Management App", "E-commerce Website", "Mobile Inventory App"],
#             "work_experience": [
#                 {"title": "Frontend Developer", "years": 3},
#                 {"title": "Mobile Developer", "years": 2}
#             ],
#             "test_score": 76,
#             "preferred_domain": "Web Development",
#             "id": "candidate_1254"
#         }
#     ]


# # ============================================================================
# # Main Pipeline Function
# # ============================================================================

# def main():
#     print("\n=== Step 10: Final JSON Output Generation ===")
    
#     required_files = get_required_artifact_files()
#     check_required_files(required_files)
    
#     classification_pipeline = load_classification_pipeline()
#     print("✓ Pipeline loaded from saved artifacts")
    
#     sample_resumes = get_sample_resumes()
    
#     print(f"\nGenerating JSON outputs for {len(sample_resumes)} resumes...")
#     results = classification_pipeline.batch_classify(
#         sample_resumes, 
#         output_file=DATA_DIR / 'sample_json_outputs.json'
#     )
    
#     print("\nExample output of all resumes:\n")
#     for i, resume in enumerate(results, start=1):
#         print(f"Resume {i}:")
#         print(json.dumps(resume, indent=2))
#         print("-" * 80)  # separator between resumes
    
#     print("\n✓ Step 10 Complete - Pipeline ready for production use")
    
#     return classification_pipeline, results


# # ============================================================================
# # Usage Example
# # ============================================================================

# if __name__ == "__main__":
#     print(f"Base path set to: {BASE_PATH}")
#     classification_pipeline, results = main()




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
        {
            "skills": [
                "JavaScript", "React", "Node.js", "HTML", "CSS", "MongoDB", "Express",
                "Python", "SQL", "Docker", "AWS", "CI/CD", "Kubernetes", "Terraform",
                "Penetration Testing", "Network Security"
            ],
            "projects": ["Social Media Platform", "E-commerce Website", "CI/CD Pipeline Setup"],
            "work_experience": [
                {"title": "Full Stack Developer", "years": 5},
                {"title": "DevOps Engineer", "years": 2}
            ],
            "test_score": 48,
            "preferred_domain": "Web Development",
            "id": "candidate_1250"
        },
        {
            "skills": [
                "Flutter", "Kotlin", "Java", "React Native", "Swift", "iOS", "Android",
                "Python", "SQL", "Docker", "React.js", "Tailwind CSS", "AWS", "Terraform"
            ],
            "projects": ["Mobile Banking App", "Fitness Tracker", "Expense Tracker"],
            "work_experience": [
                {"title": "Mobile Developer", "years": 4},
                {"title": "Data Analyst", "years": 1}
            ],
            "test_score": 82,
            "preferred_domain": "Mobile Development",
            "id": "candidate_1251"
        },
        {
            "skills": [
                "Docker", "Kubernetes", "AWS", "Jenkins", "Terraform", "Linux", "CI/CD",
                "Python", "React.js", "Node.js", "SQL", "Flutter", "Penetration Testing"
            ],
            "projects": ["Infrastructure as Code", "Automated Deployment", "Monitoring Dashboard"],
            "work_experience": [
                {"title": "DevOps Engineer", "years": 6},
                {"title": "Full Stack Developer", "years": 2}
            ],
            "test_score": 85,
            "preferred_domain": "DevOps",
            "id": "candidate_1252"
        },
        {
            "skills": [
                "Network Security", "Penetration Testing", "CISSP", "Firewall", "Encryption",
                "Python", "Docker", "AWS", "React.js", "SQL", "Kubernetes"
            ],
            "projects": ["Vulnerability Assessment", "Incident Response Plan", "Security Information Dashboard"],
            "work_experience": [
                {"title": "Cybersecurity Engineer", "years": 5},
                {"title": "DevOps Engineer", "years": 1}
            ],
            "test_score": 79,
            "preferred_domain": "Cybersecurity",
            "id": "candidate_1253"
        },
        {
            "skills": [
                "JavaScript", "Vue.js", "HTML", "CSS", "Node.js", "Express", "React",
                "Docker", "Python", "AWS", "Kubernetes", "Flutter", "Penetration Testing"
            ],
            "projects": ["Task Management App", "E-commerce Website", "Mobile Inventory App"],
            "work_experience": [
                {"title": "Frontend Developer", "years": 3},
                {"title": "Mobile Developer", "years": 2}
            ],
            "test_score": 76,
            "preferred_domain": "Web Development",
            "id": "candidate_1254"
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


# Keep your existing if __name__ == "__main__" for local testing
if __name__ == "__main__":
    print(f"Base path set to: {BASE_PATH}")
    sample_resumes = get_sample_resumes()
    
    # For local testing, process first sample
    results = main(sample_resumes[0])
    print(json.dumps(results, indent=2))