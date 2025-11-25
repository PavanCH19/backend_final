# src/save_artifacts.py
import joblib
import json
from pathlib import Path
from datetime import datetime

def save_all_artifacts(
    model_classifier,
    scaler,
    vector_builder,
    skill_vocab,
    domain_requirements,
    label_encoder,
    results,
    X_train,
    X_val,
    X_test
):
    """Save trained model and all preprocessing artifacts in artifacts/ and models/"""

    print("\n=== Step 8: Save Model & Artifacts ===")

    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('artifacts').mkdir(exist_ok=True)

    # Get the trained model
    trained_model = model_classifier.model

    # Save model weights & architecture
    print("Saving model architecture and weights...")
    trained_model.save('models/resume_classifier_complete.keras')  # modern Keras format
    print("✓ Saved: models/resume_classifier_complete.keras")

    # TensorFlow SavedModel format
    trained_model.export('models/resume_classifier_savedmodel')
    print("✓ Saved: models/resume_classifier_savedmodel/ (TensorFlow SavedModel)")

    # Save preprocessing artifacts
    print("Saving preprocessing artifacts...")
    joblib.dump(scaler, 'artifacts/feature_scaler.pkl')
    print("✓ Saved: artifacts/feature_scaler.pkl")

    joblib.dump(vector_builder, 'artifacts/feature_vector_builder.pkl')
    print("✓ Saved: artifacts/feature_vector_builder.pkl")

    with open('artifacts/skill_vocabulary.json', 'w') as f:
        json.dump(skill_vocab, f, indent=2)
    print("✓ Saved: artifacts/skill_vocabulary.json")

    with open('artifacts/domain_requirements.json', 'w') as f:
        json.dump(domain_requirements, f, indent=2)
    print("✓ Saved: artifacts/domain_requirements.json")

    joblib.dump(label_encoder, 'artifacts/label_encoder.pkl')
    print("✓ Saved: artifacts/label_encoder.pkl")

    # joblib.dump(classification_pipeline, 'artifacts/classification_pipeline.pkl')
    # print("✓ Saved: artifacts/classification_pipeline.pkl")

    # Explanation config
    explanation_config = {
        "score_thresholds": {"excellent": 85, "high": 75, "good": 60, "fair": 50},
        "skill_ratio_thresholds": {"most": 0.8, "many": 0.6, "some": 0.4},
        "experience_thresholds": {"solid": 3, "some": 1},
        "confidence_precision": 3,
        "explanation_template": "template_based_explanation"
    }
    with open('artifacts/explanation_config.json', 'w') as f:
        json.dump(explanation_config, f, indent=2)
    print("✓ Saved: artifacts/explanation_config.json")

    # Manifest/metadata
    model_manifest = {
        "model_name": "resume_classifier",
        "version": "1.0",
        "created_date": datetime.now().isoformat(),
        "model_architecture": "hybrid_neural_network",
        "input_features": {
            "skill_vocabulary_size": len(skill_vocab),
            "numeric_features": 4,
            "text_features": 128 if model_classifier.use_text_branch else 0,
            "total_features": (
                model_classifier.numeric_dim +
                model_classifier.skill_vocab_size +
                (128 if model_classifier.use_text_branch else 0)
            )
        },
        "output_classes": label_encoder.classes_.tolist(),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "test_accuracy": float(results['test_accuracy']),
        "test_loss": float(results['test_loss']),
        "f1_macro": float(results['f1_macro']),
        "f1_weighted": float(results['f1_weighted']),
        "artifacts": {
            "model_weights": "models/resume_classifier_complete.keras",
            "savedmodel": "models/resume_classifier_savedmodel/",
            "feature_scaler": "artifacts/feature_scaler.pkl",
            "skill_vocabulary": "artifacts/skill_vocabulary.json",
            "label_encoder": "artifacts/label_encoder.pkl",
            "feature_builder": "artifacts/feature_vector_builder.pkl",
            "domain_requirements": "artifacts/domain_requirements.json",
            # "pipeline": "artifacts/classification_pipeline.pkl",
            "explanation_config": "artifacts/explanation_config.json"
        }
    }
    with open('artifacts/model_manifest.json', 'w') as f:
        json.dump(model_manifest, f, indent=2)
    print("✓ Saved: artifacts/model_manifest.json")

    # Quick test
    # try:
    #     loaded_pipeline = joblib.load('artifacts/classification_pipeline.pkl')
    #     test_resume = sample_resumes[0]
    #     loaded_result = loaded_pipeline.classify_resume(test_resume)

    #     if 'error' not in loaded_result:
    #         print("✓ Successfully loaded and tested complete pipeline")
    #         print(f"  Test prediction: {loaded_result['label']} ({loaded_result['confidence']})")
    #     else:
    #         print(f"✗ Pipeline test failed: {loaded_result['error']}")
    # except Exception as e:
    #     print(f"✗ Failed to load pipeline: {e}")

    # print("\n" + "="*70)
    # print("All artifacts saved successfully!")
    # print("="*70)
