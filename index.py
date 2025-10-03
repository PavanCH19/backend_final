# ============================================================================
# Resume Classifier Implementation - Complete Pipeline
# File: resume_classifier.ipynb
# ============================================================================

import os
import json
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================================
# STEP 0: PROJECT SETUP
# ============================================================================

def setup_project():
    """Initialize project structure and random seeds"""
    print("=== Step 0: Project Setup ===\n")
    
    # Set random seeds for reproducibility
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Create project structure
    project_folders = ['data', 'models', 'data/domain_requirements', "src"]
    for folder in project_folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder}")
    
    print("✓ Project structure created successfully!\n")
    return SEED


# ============================================================================
# STEP 1: BALANCED DATASET GENERATION
# ============================================================================

def generate_dataset():
    """Generate and save balanced synthetic resumes"""
    print("=== Step 1: Balanced Dataset Generation ===\n")
    
    from src.data_generation import (
        get_domain_requirements, 
        save_domain_requirements, 
        generate_balanced_resumes, 
        save_resumes_to_json
    )
    
    # Get domain requirements and save
    domain_requirements = get_domain_requirements()
    save_domain_requirements(domain_requirements)
    
    # Generate and save balanced resumes
    synthetic_resumes = generate_balanced_resumes(n_samples=2000)
    save_resumes_to_json(synthetic_resumes)
    
    print(f"✓ Generated {len(synthetic_resumes)} synthetic resumes\n")
    return domain_requirements, synthetic_resumes


# ============================================================================
# STEP 2: LABELLING DATA
# ============================================================================

def label_resumes():
    """Load resumes and calculate classification labels"""
    print("=== Step 2: Labelling Data ===\n")
    
    from src.labelling_data import (
        load_domain_requirements, 
        load_resumes, 
        calculate_labels, 
        save_labeled_resumes, 
        analyze_labeled_resumes
    )
    
    # Load domain requirements and resumes
    domain_requirements = load_domain_requirements()
    synthetic_resumes = load_resumes()
    
    # Calculate labels
    labeled_resumes = calculate_labels(synthetic_resumes, domain_requirements)
    
    # Save labeled resumes
    save_labeled_resumes(labeled_resumes)
    
    # Analyze and visualize
    analyze_labeled_resumes(labeled_resumes)
    
    print("✓ Labelling complete\n")
    return labeled_resumes


# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def engineer_features():
    """Extract and engineer features from labeled resumes"""
    print("=== Step 3: Feature Engineering ===\n")
    
    from src.feature_engineering import (
        build_skill_vocabulary, 
        extract_features_from_resumes, 
        scale_numeric_features, 
        save_skill_vocabulary, 
        save_sample_features, 
        print_feature_summary, 
        FeatureVectorBuilder, 
        build_feature_vectors, 
        print_feature_vector_summary, 
        save_all_artifacts
    )
    
    # Load labeled resumes and domain requirements
    with open('data/labeled_synthetic_resumes.json', 'r') as f:
        labeled_resumes = json.load(f)
    
    with open('data/domain_requirements.json', 'r') as f:
        domain_requirements = json.load(f)
    
    # Build vocabulary
    skill_vocab = build_skill_vocabulary(labeled_resumes, domain_requirements)
    print(f"Skill vocabulary: {len(skill_vocab)} unique skills")
    
    # Extract features
    all_features = extract_features_from_resumes(labeled_resumes, skill_vocab, domain_requirements)
    print(f"Extracted features from {len(all_features)} resumes")
    
    # Scale numeric features
    scaler = scale_numeric_features(all_features)
    
    # Save artifacts
    save_skill_vocabulary(skill_vocab)
    save_sample_features(all_features)
    
    # Print summary
    print_feature_summary(all_features)
    
    # Configure text features
    use_text_features = True  # Set to False to disable text features
    
    # Initialize vector builder
    vector_builder = FeatureVectorBuilder(
        skill_vocab_size=len(skill_vocab), 
        use_text_embeddings=use_text_features
    )
    
    # Fit text vectorizers if using text features
    if use_text_features:
        vector_builder.fit_text_vectorizers(all_features)
    
    # Build feature vectors
    X, y = build_feature_vectors(all_features, vector_builder)
    
    # Get dimensions
    dimensions = vector_builder.get_feature_dimensions()
    
    # Print summary
    print_feature_vector_summary(X, y, dimensions)
    
    # Save all artifacts
    save_all_artifacts(X, y, vector_builder, dimensions)
    
    print("✓ Feature engineering complete\n")
    return skill_vocab, all_features, scaler, vector_builder, X, y, dimensions


# ============================================================================
# STEP 4: MODEL ARCHITECTURE
# ============================================================================

def build_model_architecture():
    """Create and configure the neural network model"""
    print("=== Step 4: Model Architecture ===\n")
    
    from src.model_architecture import (
        set_random_seed, 
        load_all_model_data, 
        validate_feature_dimensions, 
        create_model, 
        print_model_input_shapes
    )
    
    # Set random seed
    set_random_seed(42)
    
    # Load all data
    skill_vocab, dimensions, label_mapping, X = load_all_model_data()
    
    # Validate dimensions
    validate_feature_dimensions(X, skill_vocab, dimensions)
    
    # Create model
    model_classifier = create_model(
        skill_vocab_size=len(skill_vocab),
        dimensions=dimensions,
        num_classes=len(label_mapping['unique_labels'])
    )
    
    # Print model info
    print_model_input_shapes(model_classifier.model)
    print("\n" + "="*70)
    print("Model Summary:")
    print("="*70)
    model_classifier.model.summary()
    
    print("\n✓ Model architecture created\n")
    return model_classifier, skill_vocab, dimensions, label_mapping, X


# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================

def train_model_pipeline(model_classifier, X, y, label_mapping):
    """Train the model with proper data splitting and callbacks"""
    print("=== Step 5: Model Training ===\n")
    
    from src.model_train_test import (
        encode_labels, 
        split_train_val_test, 
        print_split_info, 
        prepare_all_inputs, 
        compute_class_weights, 
        create_training_callbacks, 
        train_model, 
        save_training_history, 
        save_complete_model, 
        save_model_config, 
        save_label_encoder, 
        plot_training_history, 
        print_training_summary
    )
    
    # Encode labels
    label_encoder, y_encoded, y_categorical = encode_labels(y, len(label_mapping['unique_labels']))
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y_categorical, y_encoded
    )
    print_split_info(X_train, X_val, X_test)
    
    # Prepare inputs
    train_inputs, val_inputs, test_inputs = prepare_all_inputs(
        model_classifier, X_train, X_val, X_test
    )
    
    # Compute class weights
    class_weight_dict = compute_class_weights(y_train)
    
    # Training parameters
    batch_size = 32
    epochs = 50
    patience = 5
    
    # Create callbacks
    callbacks = create_training_callbacks(patience=patience)
    
    # Train model
    history = train_model(
        model_classifier.model,
        train_inputs,
        y_train,
        val_inputs,
        y_val,
        class_weight_dict,
        callbacks,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save everything
    save_training_history(history)
    save_complete_model(model_classifier.model)
    save_model_config(model_classifier)
    save_label_encoder(label_encoder)
    
    # Visualize and summarize
    plot_training_history(history)
    print_training_summary(history)
    
    print("\n✓ Training complete → models/best_resume_classifier.h5\n")
    
    return label_encoder, history, X_train, X_val, X_test, y_train, y_val, y_test, test_inputs


# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================

def evaluate_model_pipeline(model_classifier, test_inputs, y_test, label_encoder):
    """Evaluate model performance and generate metrics"""
    print("=== Step 6: Model Evaluation ===\n")
    
    from src.model_evaluation import (
        get_predictions, 
        evaluate_model, 
        calculate_classification_metrics, 
        print_evaluation_results, 
        plot_confusion_matrix, 
        create_calibration_analysis, 
        analyze_uncertainty, 
        save_model_artifacts, 
        save_evaluation_metrics
    )
    
    model = model_classifier.model
    uncertainty_threshold = 0.3  # Threshold for uncertainty analysis
    
    # Get predictions
    y_pred_probs, y_pred, y_true = get_predictions(model, test_inputs, y_test)
    
    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_inputs, y_test)
    
    # Calculate metrics
    target_names = label_encoder.classes_
    report, report_dict, cm, f1_macro, f1_weighted = calculate_classification_metrics(
        y_true, y_pred, target_names
    )
    
    # Print results
    print_evaluation_results(test_loss, test_accuracy, report)
    
    # Visualizations
    plot_confusion_matrix(cm, target_names)
    create_calibration_analysis(y_true, y_pred, y_pred_probs, target_names)
    
    # Uncertainty analysis
    uncertain_mask, uncertain_count = analyze_uncertainty(y_pred_probs, uncertainty_threshold)
    
    # Save artifacts
    save_model_artifacts(model, label_encoder)
    save_evaluation_metrics(test_loss, test_accuracy, f1_macro, f1_weighted, 
                          cm, report_dict)
    
    print("\n✓ Evaluation complete → models/resume_classifier_model.h5, evaluation_metrics.json\n")
    
    result = {
        'y_pred_probs': y_pred_probs,
        'y_pred': y_pred,
        'y_true': y_true,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'report_dict': report_dict,
        'uncertain_mask': uncertain_mask
    }
    
    return result


# ============================================================================
# STEP 7: SAVE ALL ARTIFACTS
# ============================================================================

def save_pipeline_artifacts(model_classifier, scaler, vector_builder, skill_vocab, 
                           domain_requirements, label_encoder, result, 
                           X_train, X_val, X_test):
    """Save all model artifacts for production deployment"""
    print("=== Step 7: Saving Artifacts ===\n")
    
    from src.save_artifacts import save_all_artifacts
    
    save_all_artifacts(
        model_classifier,
        scaler,
        vector_builder,
        skill_vocab,
        domain_requirements,
        label_encoder,
        result,      
        X_train,
        X_val,
        X_test
    )
    
    print("✓ All artifacts saved\n")


# ============================================================================
# STEP 8: PREDICTION PIPELINE
# ============================================================================

def run_prediction_pipeline():
    """Load saved artifacts and run predictions on sample resumes"""
    print("=== Step 8: Prediction Pipeline ===\n")
    
    from src.prediction import (
        get_required_artifact_files, 
        check_required_files, 
        load_classification_pipeline, 
        get_sample_resumes
    )
    
    # Check required files
    required_files = get_required_artifact_files()
    check_required_files(required_files)
    
    # Load pipeline
    classification_pipeline = load_classification_pipeline()
    print("✓ Pipeline loaded from saved artifacts")
    
    # Get sample resumes
    sample_resumes = get_sample_resumes()
    
    # Classify resumes
    print(f"\nGenerating JSON outputs for {len(sample_resumes)} resumes...")
    results = classification_pipeline.batch_classify(
        sample_resumes, 
        output_file='data/sample_json_outputs.json'
    )
    
    # Display example output
    print(f"\nExample output:\n{json.dumps(results[0], indent=2)}")
    print("\n✓ Step 8 Complete - Pipeline ready for production use\n")
    
    return classification_pipeline, results


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Execute complete resume classifier pipeline"""
    print("\n" + "="*70)
    print("RESUME CLASSIFIER - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # Step 0: Setup
    SEED = setup_project()
    
    # Step 1: Generate Dataset
    domain_requirements, synthetic_resumes = generate_dataset()
    
    # Step 2: Label Data
    labeled_resumes = label_resumes()
    
    # Step 3: Feature Engineering
    skill_vocab, all_features, scaler, vector_builder, X, y, dimensions = engineer_features()
    
    # Step 4: Build Model
    model_classifier, skill_vocab, dimensions, label_mapping, X = build_model_architecture()
    
    # Step 5: Train Model
    label_encoder, history, X_train, X_val, X_test, y_train, y_val, y_test, test_inputs = train_model_pipeline(
        model_classifier, X, y, label_mapping
    )
    
    # Step 6: Evaluate Model
    result = evaluate_model_pipeline(model_classifier, test_inputs, y_test, label_encoder)
    
    # Step 7: Save Artifacts
    save_pipeline_artifacts(
        model_classifier, scaler, vector_builder, skill_vocab, 
        domain_requirements, label_encoder, result, 
        X_train, X_val, X_test
    )
    
    # Step 8: Run Predictions
    classification_pipeline, results = run_prediction_pipeline()
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70 + "\n")
    
    return classification_pipeline, results


# ============================================================================
# EXECUTE PIPELINE
# ============================================================================

if __name__ == "__main__":
    classification_pipeline, results = main()