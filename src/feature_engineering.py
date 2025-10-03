import json
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Skill Vocabulary Functions
# ============================================================================

def build_skill_vocabulary(resumes, domain_requirements):
    """Build comprehensive skill vocabulary from resumes and domain requirements"""
    all_skills = set()
    
    # Collect skills from resumes
    for resume in resumes:
        all_skills.update(resume.get('skills', []))
    
    # Collect skills from domain requirements
    for domain_data in domain_requirements.values():
        all_skills.update(domain_data.get('required_skills', []))
    
    # Clean and normalize skills
    cleaned_skills = {
        skill.strip().lower() 
        for skill in all_skills 
        if skill.strip() and len(skill.strip()) > 1
    }
    
    return sorted(list(cleaned_skills))


def encode_skills(candidate_skills, skill_vocab):
    """Convert skills to binary vector representation"""
    skill_vector = np.zeros(len(skill_vocab), dtype=int)
    normalized_skills = {skill.strip().lower() for skill in candidate_skills}
    
    for i, vocab_skill in enumerate(skill_vocab):
        if vocab_skill in normalized_skills:
            skill_vector[i] = 1
    
    return skill_vector


# ============================================================================
# Skill Matching Functions
# ============================================================================

def compute_skill_matches(candidate_skills, required_skills):
    """Compute matched and missing skills with match ratio"""
    candidate_set = {skill.strip().lower() for skill in candidate_skills}
    required_set = {skill.strip().lower() for skill in required_skills}
    
    matched_skills = list(candidate_set.intersection(required_set))
    missing_skills = list(required_set - candidate_set)
    skill_match_ratio = len(matched_skills) / len(required_set) if required_set else 0.0
    
    return matched_skills, missing_skills, skill_match_ratio


def suggest_alternative_domains(candidate_skills, current_domain, domain_requirements, top_n=3):
    """Suggest alternative domains based on skill match"""
    suggestions = []
    
    for domain_key, domain_info in domain_requirements.items():
        domain_name = domain_info["domain"]
        
        # Skip current domain
        if domain_name == current_domain:
            continue
        
        required_skills = domain_info["required_skills"]
        matched, missing, ratio = compute_skill_matches(candidate_skills, required_skills)
        
        suggestions.append({
            'domain': domain_name,
            'domain_key': domain_key,
            'skill_match_ratio': ratio,
            'matched_skills': matched,
            'missing_skills': missing,
            'matched_count': len(matched),
            'required_count': len(required_skills)
        })
    
    # Sort by match ratio in descending order
    suggestions.sort(key=lambda x: x['skill_match_ratio'], reverse=True)
    return suggestions[:top_n]


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_project_features(projects):
    """Extract features from project list"""
    project_count = len(projects) if projects else 0
    project_text = " ".join(projects) if projects else ""
    return project_count, project_text


def extract_experience_features(work_experience):
    """Extract features from work experience"""
    if not work_experience:
        return 0.0, 0, ""
    
    years_experience = sum(item.get('years', 0) for item in work_experience)
    max_years = max(item.get('years', 0) for item in work_experience)
    job_titles = [item.get('title', '') for item in work_experience]
    experience_text = " ".join(job_titles)
    
    return float(years_experience), max_years, experience_text


def normalize_test_score(test_score):
    """Normalize test score to 0-1 range"""
    return max(0, min(100, float(test_score))) / 100.0


def extract_all_features(resume, skill_vocab, domain_requirements):
    """Extract all features from a single resume"""
    # Find domain key
    domain_key = next(
        (k for k, v in domain_requirements.items() if v["domain"] == resume["preferred_domain"]), 
        None
    )
    if domain_key is None:
        raise ValueError(f"No requirements found for domain {resume['preferred_domain']}")
    
    # Extract basic data
    required_skills = domain_requirements[domain_key]["required_skills"]
    candidate_skills = resume.get('skills', [])
    projects = resume.get('projects', [])
    work_experience = resume.get('work_experience', [])
    test_score = resume.get('test_score', 0)
    
    # Compute features
    skill_vector = encode_skills(candidate_skills, skill_vocab)
    matched_skills, missing_skills, skill_match_ratio = compute_skill_matches(
        candidate_skills, required_skills
    )
    project_count, project_text = extract_project_features(projects)
    years_experience, max_years, experience_text = extract_experience_features(work_experience)
    test_score_norm = normalize_test_score(test_score)
    numeric_features = [years_experience, max_years, project_count]
    alternative_domains = suggest_alternative_domains(
        candidate_skills, resume['preferred_domain'], domain_requirements, top_n=3
    )
    
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
        'id': resume['id'],
        'alternative_domains': alternative_domains
    }


# ============================================================================
# Feature Scaling
# ============================================================================

class ResumeFeatureScaler:
    """Scaler for numeric resume features"""
    
    def __init__(self):
        self.project_scaler = StandardScaler()
        self.experience_scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, numeric_features):
        """Fit the scaler on numeric features"""
        if len(numeric_features) == 0:
            return self
        
        numeric_array = np.array(numeric_features)
        if numeric_array.shape[1] >= 3:
            self.experience_scaler.fit(numeric_array[:, :2])
            self.project_scaler.fit(numeric_array[:, 2:3])
        
        self.is_fitted = True
        return self
    
    def transform(self, numeric_features):
        """Transform numeric features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        numeric_array = np.array(numeric_features)
        if numeric_array.ndim == 1:
            numeric_array = numeric_array.reshape(1, -1)
        
        if numeric_array.shape[1] >= 3:
            scaled_experience = self.experience_scaler.transform(numeric_array[:, :2])
            scaled_projects = self.project_scaler.transform(numeric_array[:, 2:3])
            return np.concatenate([scaled_experience, scaled_projects], axis=1)
        
        return np.array([])
    
    def fit_transform(self, numeric_features):
        """Fit and transform in one step"""
        return self.fit(numeric_features).transform(numeric_features)


# ============================================================================
# Batch Processing Functions
# ============================================================================

def extract_features_from_resumes(labeled_resumes, skill_vocab, domain_requirements):
    """Extract features from all resumes"""
    all_features = []
    
    for resume in labeled_resumes:
        try:
            features = extract_all_features(resume, skill_vocab, domain_requirements)
            features['label'] = resume['label']
            all_features.append(features)
        except Exception as e:
            print(f"Error processing resume {resume.get('id', 'unknown')}: {e}")
            continue
    
    return all_features


def scale_numeric_features(all_features):
    """Scale numeric features for all extracted features"""
    # Extract numeric feature matrix
    numeric_feature_matrix = [f['numeric_features'] for f in all_features]
    
    # Fit scaler
    scaler = ResumeFeatureScaler()
    scaler.fit(numeric_feature_matrix)
    
    # Transform and add scaled features
    for features in all_features:
        scaled_numeric = scaler.transform([features['numeric_features']])
        features['scaled_numeric_features'] = scaled_numeric[0]
    
    return scaler


# ============================================================================
# Save/Load Functions
# ============================================================================

def save_skill_vocabulary(skill_vocab, filepath='data/skill_vocab.json'):
    """Save skill vocabulary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(skill_vocab, f, indent=2)
    print(f"Saved skill vocabulary → {filepath}")


def prepare_sample_features_for_json(features_list, n_samples=5):
    """Prepare sample features for JSON serialization"""
    sample_features = features_list[:n_samples]
    
    # Convert numpy arrays to lists for JSON serialization
    for features in sample_features:
        features['skill_vector'] = features['skill_vector'].tolist()
        features['scaled_numeric_features'] = features['scaled_numeric_features'].tolist()
    
    return sample_features


def save_sample_features(all_features, filepath='data/sample_features.json', n_samples=5):
    """Save sample features to JSON file"""
    sample_features = prepare_sample_features_for_json(all_features, n_samples)
    
    with open(filepath, 'w') as f:
        json.dump(sample_features, f, indent=2)
    print(f"Saved sample features → {filepath}")


def print_feature_summary(all_features):
    """Print summary of extracted features"""
    if not all_features:
        print("No features to summarize")
        return
    
    sample = all_features[0]
    
    print(f"\nFeature Summary (sample):")
    print(f"  Skill match ratio: {sample['skill_match_ratio']:.3f}")
    print(f"  Projects: {sample['project_count']} | Experience: {sample['years_experience']:.0f} yrs")
    
    if sample['alternative_domains']:
        top_alt = sample['alternative_domains'][0]
        print(f"  Top alternative: {top_alt['domain']} ({top_alt['skill_match_ratio']:.3f} match)")


# ============================================================================
# Main Pipeline Function
# ============================================================================

# def main(labeled_resumes, domain_requirements):
#     """Main function to execute feature engineering pipeline"""
#     print("\n=== Step 3: Feature Engineering ===")
    
#     # Build vocabulary
#     skill_vocab = build_skill_vocabulary(labeled_resumes, domain_requirements)
#     print(f"Skill vocabulary: {len(skill_vocab)} unique skills")
    
#     # Extract features
#     all_features = extract_features_from_resumes(labeled_resumes, skill_vocab, domain_requirements)
#     print(f"Extracted features from {len(all_features)} resumes")
    
#     # Scale numeric features
#     scaler = scale_numeric_features(all_features)
    
#     # Save artifacts
#     save_skill_vocabulary(skill_vocab)
#     save_sample_features(all_features)
    
#     # Print summary
#     print_feature_summary(all_features)
    
#     return all_features, skill_vocab, scaler


# ============================================================================
# Usage Example
# ============================================================================

# if __name__ == "__main__":
#     # Load data (assuming these are already loaded)
#     with open('data/labeled_synthetic_resumes.json', 'r') as f:
#         labeled_resumes = json.load(f)
    
#     with open('data/domain_requirements.json', 'r') as f:
#         domain_requirements = json.load(f)
    
#     # Run feature engineering
#     all_features, skill_vocab, scaler = main(labeled_resumes, domain_requirements)


import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================================
# Feature Vector Builder Class
# ============================================================================

class FeatureVectorBuilder:
    """Builds final feature vectors using parallel branches"""
    
    def __init__(self, skill_vocab_size, use_text_embeddings=False):
        self.skill_vocab_size = skill_vocab_size
        self.use_text_embeddings = use_text_embeddings
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
        """Fit TF-IDF vectorizers on text data"""
        project_texts = [f.get('project_text', '') or 'no projects' for f in all_features]
        experience_texts = [f.get('experience_text', '') or 'no experience' for f in all_features]
        
        self.project_vectorizer.fit(project_texts)
        self.experience_vectorizer.fit(experience_texts)
        self.is_fitted = True
    
    def build_skill_branch(self, features):
        """Branch 1: Skills (binary vector + match ratio)"""
        return np.concatenate([features['skill_vector'], [features['skill_match_ratio']]])
    
    def build_numeric_branch(self, features):
        """Branch 2: Numeric features"""
        scaled = features['scaled_numeric_features']
        return np.array([
            features['test_score_norm'],
            scaled[2],  # project_count_scaled
            scaled[0],  # years_experience_scaled
            features['skill_match_ratio']
        ])
    
    def build_text_branch(self, features):
        """Branch 3: Text embeddings (optional)"""
        if not self.use_text_embeddings or not self.is_fitted:
            return np.array([])
        
        project_text = features.get('project_text', '') or 'no projects'
        experience_text = features.get('experience_text', '') or 'no experience'
        
        project_vector = self.project_vectorizer.transform([project_text]).toarray().flatten()
        experience_vector = self.experience_vectorizer.transform([experience_text]).toarray().flatten()
        
        return np.concatenate([project_vector, experience_vector])
    
    def build_final_vector(self, features):
        """Concatenate all branches into final feature vector"""
        branches = [self.build_skill_branch(features), self.build_numeric_branch(features)]
        
        if self.use_text_embeddings and self.is_fitted:
            text_branch = self.build_text_branch(features)
            if len(text_branch) > 0:
                branches.append(text_branch)
        
        return np.concatenate(branches)
    
    def get_feature_dimensions(self):
        """Get dimensions of each feature branch"""
        skill_dim = self.skill_vocab_size + 1
        numeric_dim = 4
        text_dim = 128 if self.use_text_embeddings else 0
        
        return {
            'skill_branch_dim': skill_dim,
            'numeric_branch_dim': numeric_dim,
            'text_branch_dim': text_dim,
            'final_vector_dim': skill_dim + numeric_dim + text_dim
        }


# ============================================================================
# Feature Vector Construction Functions
# ============================================================================

def build_feature_vectors(all_features, vector_builder):
    """Build feature vectors and labels from extracted features"""
    feature_vectors = []
    labels = []
    
    for features in all_features:
        try:
            feature_vectors.append(vector_builder.build_final_vector(features))
            labels.append(features['label'])
        except Exception as e:
            print(f"Error building vector for {features.get('id', 'unknown')}: {e}")
            continue
    
    X = np.array(feature_vectors)
    y = np.array(labels)
    
    return X, y


def calculate_sparsity(X):
    """Calculate sparsity of feature matrix"""
    return np.mean(X == 0)


def get_label_distribution(y):
    """Get unique labels and their counts"""
    unique_labels, label_counts = np.unique(y, return_counts=True)
    return unique_labels, label_counts


def print_feature_vector_summary(X, y, dimensions):
    """Print summary of feature vectors"""
    print(f"Feature vectors built: {X.shape}")
    print(f"  Skill: {dimensions['skill_branch_dim']} | "
          f"Numeric: {dimensions['numeric_branch_dim']} | "
          f"Text: {dimensions['text_branch_dim']}")
    print(f"  Sparsity: {calculate_sparsity(X):.2%}")
    
    unique_labels, label_counts = get_label_distribution(y)
    print(f"\nLabel distribution: {dict(zip(unique_labels, label_counts))}")


# ============================================================================
# Label Mapping Functions
# ============================================================================

def create_label_mapping(unique_labels):
    """Create bidirectional label mapping"""
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    return {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'unique_labels': unique_labels.tolist()
    }


# ============================================================================
# Save/Load Functions
# ============================================================================

def save_feature_matrix(X, filepath='data/X_features.npy'):
    """Save feature matrix to numpy file"""
    np.save(filepath, X)
    print(f"Saved feature matrix → {filepath}")


def save_labels(y, filepath='data/y_labels.npy'):
    """Save labels to numpy file"""
    np.save(filepath, y)
    print(f"Saved labels → {filepath}")


def save_vector_builder(vector_builder, filepath='data/feature_vector_builder.pkl'):
    """Save vector builder using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(vector_builder, f)
    print(f"Saved vector builder → {filepath}")


def save_label_mapping(unique_labels, filepath='data/label_mapping.json'):
    """Save label mapping to JSON file"""
    label_mapping = create_label_mapping(unique_labels)
    
    with open(filepath, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved label mapping → {filepath}")


def save_feature_dimensions(dimensions, filepath='data/feature_dimensions.json'):
    """Save feature dimensions to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(dimensions, f, indent=2)
    print(f"Saved feature dimensions → {filepath}")


def load_vector_builder(filepath='data/feature_vector_builder.pkl'):
    """Load vector builder from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_feature_matrix(filepath='data/X_features.npy'):
    """Load feature matrix from numpy file"""
    return np.load(filepath)


def load_labels(filepath='data/y_labels.npy'):
    """Load labels from numpy file"""
    return np.load(filepath)


def load_label_mapping(filepath='data/label_mapping.json'):
    """Load label mapping from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================================
# Save All Artifacts
# ============================================================================

def save_all_artifacts(X, y, vector_builder, dimensions):
    """Save all feature construction artifacts"""
    # Save feature matrix and labels
    save_feature_matrix(X)
    save_labels(y)
    
    # Save vector builder
    save_vector_builder(vector_builder)
    
    # Save label mapping
    unique_labels, _ = get_label_distribution(y)
    save_label_mapping(unique_labels)
    
    # Save dimensions
    save_feature_dimensions(dimensions)
    
    print("\nAll artifacts saved successfully!")


# ============================================================================
# Main Pipeline Function
# ============================================================================

# def main(all_features, skill_vocab, use_text_features=True):
#     """Main function to execute feature vector construction pipeline"""
#     print("\n=== Step 4: Feature Vector Construction ===")
    
#     # Initialize vector builder
#     vector_builder = FeatureVectorBuilder(
#         skill_vocab_size=len(skill_vocab), 
#         use_text_embeddings=use_text_features
#     )
    
#     # Fit text vectorizers if using text features
#     if use_text_features:
#         vector_builder.fit_text_vectorizers(all_features)
    
#     # Build feature vectors
#     X, y = build_feature_vectors(all_features, vector_builder)
    
#     # Get dimensions
#     dimensions = vector_builder.get_feature_dimensions()
    
#     # Print summary
#     print_feature_vector_summary(X, y, dimensions)
    
#     # Save all artifacts
#     save_all_artifacts(X, y, vector_builder, dimensions)
    
#     return X, y, vector_builder, dimensions


# # ============================================================================
# # Usage Example
# # ============================================================================

# if __name__ == "__main__":
#     # Load data (assuming these are already available)
#     with open('data/skill_vocab.json', 'r') as f:
#         skill_vocab = json.load(f)
    
#     # Load features from previous step
#     # all_features would come from the feature engineering step
#     # For demonstration, you would pass it from the previous pipeline
    
#     # Run feature vector construction
#     X, y, vector_builder, dimensions = main(all_features, skill_vocab, use_text_features=True)
    
