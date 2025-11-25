import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns


# ============================================================================
# Set Random Seed
# ============================================================================

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    tf.random.set_seed(seed)
    np.random.seed(seed)


# ============================================================================
# Model Architecture Class
# ============================================================================

class ResumeClassifierModel:
    """Hybrid neural network for resume classification with parallel branches"""
    
    def __init__(self, skill_vocab_size, numeric_dim, text_dim=0, num_classes=3):
        self.skill_vocab_size = skill_vocab_size
        self.numeric_dim = numeric_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.use_text_branch = text_dim > 0
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_compiled = False
    
    def build_skill_branch(self, skill_input):
        """Build skill branch - REDUCED for balance"""
        x = layers.Dense(128, activation='relu', name='skill_dense_1')(skill_input)  # 256→128
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu', name='skill_dense_2')(x)  # 128→64
        return x
    
    def build_numeric_branch(self, numeric_input):
        """Build numeric branch - ENHANCED for test score importance"""
        # Extract test score (first feature)
        test_score = layers.Lambda(lambda x: x[:, 0:1], name='extract_test_score')(numeric_input)
        # Extract composite features (credibility, competency, etc.)
        composite = layers.Lambda(lambda x: x[:, 1:6], name='extract_composite')(numeric_input)
        # Extract other numeric
        other_numeric = layers.Lambda(lambda x: x[:, 6:], name='extract_other')(numeric_input)
        
        # Test score gets dedicated pathway
        test_pathway = layers.Dense(32, activation='relu', name='test_dense_1')(test_score)
        test_pathway = layers.Dense(16, activation='relu', name='test_dense_2')(test_pathway)
        
        # Composite features pathway
        composite_pathway = layers.Dense(32, activation='relu', name='composite_dense')(composite)
        
        # Other numeric features
        other_pathway = layers.Dense(16, activation='relu', name='other_dense')(other_numeric)
        
        # Combine all numeric pathways
        x = layers.concatenate([test_pathway, composite_pathway, other_pathway], name='numeric_concat')
        x = layers.Dense(64, activation='relu', name='numeric_fusion')(x)  # 16→64 OUTPUT
        
        return x
    
    def build_text_branch(self, text_input):
        """Build text branch of the network"""
        x = layers.Dense(128, activation='relu')(text_input)
        x = layers.Dense(64, activation='relu')(x)
        return x
    
    def build_fusion_layers(self, concat):
        """Build fusion layers after concatenation"""
        x = layers.Dense(128, activation='relu')(concat)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        return x
    
    def build_model(self):
        """Build hybrid model architecture with parallel branches"""
        # Define inputs
        skill_input = Input(shape=(self.skill_vocab_size + 1,), name='skill_input')
        numeric_input = Input(shape=(self.numeric_dim,), name='numeric_input')
        inputs = [skill_input, numeric_input]
        
        # Build branches
        skill_branch = self.build_skill_branch(skill_input)
        numeric_branch = self.build_numeric_branch(numeric_input)
        branches_to_concat = [skill_branch, numeric_branch]
        
        # Add text branch if enabled
        if self.use_text_branch:
            text_input = Input(shape=(self.text_dim,), name='text_input')
            inputs.append(text_input)
            text_branch = self.build_text_branch(text_input)
            branches_to_concat.append(text_branch)
        
        # Concatenate branches
        concat = layers.concatenate(branches_to_concat)
        
        # Fusion layers
        fused = self.build_fusion_layers(concat)
        
        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax')(fused)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=output)
        return self.model
    
    def compile_model(self, learning_rate=1e-3):
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        self.is_compiled = True
    
    def prepare_inputs(self, X):
        """Prepare input arrays for the model from feature matrix"""
        skill_dim = self.skill_vocab_size + 1
        numeric_dim = self.numeric_dim
        
        skill_features = X[:, :skill_dim]
        numeric_features = X[:, skill_dim:skill_dim + numeric_dim]
        inputs = [skill_features, numeric_features]
        
        if self.use_text_branch:
            text_features = X[:, skill_dim + numeric_dim:]
            inputs.append(text_features)
        
        return inputs
    
    def get_model_summary(self):
        """Get model summary as string"""
        if self.model is None:
            return "Model not built yet"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_skill_vocabulary(filepath='data/skill_vocab.json'):
    """Load skill vocabulary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_feature_dimensions(filepath='data/feature_dimensions.json'):
    """Load feature dimensions from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_label_mapping(filepath='data/label_mapping.json'):
    """Load label mapping from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_feature_matrix(filepath='data/X_features.npy'):
    """Load feature matrix from numpy file"""
    return np.load(filepath)


def load_all_model_data():
    """Load all necessary data for model building"""
    skill_vocab = load_skill_vocabulary()
    dimensions = load_feature_dimensions()
    label_mapping = load_label_mapping()
    X = load_feature_matrix()
    
    return skill_vocab, dimensions, label_mapping, X


# ============================================================================
# Validation Functions
# ============================================================================

def validate_feature_dimensions(X, skill_vocab, dimensions):
    """Validate that loaded features match expected dimensions"""
    print(f"Loaded features shape: {X.shape}")
    print(f"Skill vocab size: {len(skill_vocab)}")
    print(f"Expected skill branch: {len(skill_vocab) + 1} (vocab + match_ratio)")
    print(f"Feature dimensions: {dimensions}")
    
    expected_dim = dimensions['final_vector_dim']
    actual_dim = X.shape[1]
    
    if expected_dim != actual_dim:
        print(f"\nWARNING: Dimension mismatch!")
        print(f"  Expected: {expected_dim}")
        print(f"  Actual: {actual_dim}")
        return False
    
    print("\n✓ Dimensions validated successfully")
    return True


def print_model_input_shapes(model):
    """Print expected input shapes for the model"""
    print(f"\nModel expects:")
    for i, inp in enumerate(model.inputs):
        print(f"  Input {i} ({inp.name}): {inp.shape}")


# ============================================================================
# Model Creation Functions
# ============================================================================

def create_model(skill_vocab_size, dimensions, num_classes):
    """Create and compile the resume classifier model"""
    model_classifier = ResumeClassifierModel(
        skill_vocab_size=skill_vocab_size,
        numeric_dim=dimensions['numeric_branch_dim'],
        text_dim=dimensions['text_branch_dim'],
        num_classes=num_classes
    )
    
    model_classifier.build_model()
    model_classifier.compile_model()
    
    return model_classifier


# ============================================================================
# Main Pipeline Function
# ============================================================================

# def main():
#     """Main function to execute model architecture setup"""
#     print("\n=== Step 5: Model Architecture ===")
    
#     # Set random seed
#     set_random_seed(42)
    
#     # Load all data
#     skill_vocab, dimensions, label_mapping, X = load_all_model_data()
    
#     # Validate dimensions
#     validate_feature_dimensions(X, skill_vocab, dimensions)
    
#     # Create model
#     model_classifier = create_model(
#         skill_vocab_size=len(skill_vocab),
#         dimensions=dimensions,
#         num_classes=len(label_mapping['unique_labels'])
#     )
    
#     # Print model info
#     print_model_input_shapes(model_classifier.model)
#     print("\n" + "="*70)
#     print("Model Summary:")
#     print("="*70)
#     model_classifier.model.summary()
    
#     return model_classifier, skill_vocab, dimensions, label_mapping, X


# ============================================================================
# Model Saving/Loading Functions
# ============================================================================

def save_model(model, filepath='models/resume_classifier.h5'):
    """Save trained model to file"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Model saved → {filepath}")


def load_model(filepath='models/resume_classifier.h5'):
    """Load trained model from file"""
    return keras.models.load_model(filepath)


def save_model_classifier(model_classifier, filepath='models/model_classifier.pkl'):
    """Save model classifier object using pickle"""
    import pickle
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_classifier, f)
    print(f"Model classifier saved → {filepath}")


# ============================================================================
# Usage Example
# ============================================================================

# if __name__ == "__main__":
#     model_classifier, skill_vocab, dimensions, label_mapping, X = main()