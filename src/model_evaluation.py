import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix, 
    f1_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Prediction Functions
# ============================================================================

def get_predictions(model, test_inputs, y_test):
    """Get predictions and probabilities from model"""
    y_pred_probs = model.predict(test_inputs, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    return y_pred_probs, y_pred, y_true


def evaluate_model(model, test_inputs, y_test):
    """Evaluate model on test set"""
    test_loss, test_accuracy = model.evaluate(test_inputs, y_test, verbose=0)
    return test_loss, test_accuracy


# ============================================================================
# Metrics Calculation Functions
# ============================================================================

def calculate_classification_metrics(y_true, y_pred, target_names):
    """Calculate comprehensive classification metrics"""
    # Classification report
    report = classification_report(y_true, y_pred, target_names=target_names)
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    return report, report_dict, cm, f1_macro, f1_weighted


def print_evaluation_results(test_loss, test_accuracy, report):
    """Print evaluation results"""
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print(f"\n{report}")


# ============================================================================
# Confusion Matrix Visualization
# ============================================================================

def plot_confusion_matrix(cm, target_names, save_path='data/confusion_matrix.png'):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved → {save_path}")


# ============================================================================
# Calibration Analysis Functions
# ============================================================================

def plot_confidence_distribution(ax, y_pred_probs, uncertainty_threshold=0.55):
    """Plot confidence distribution histogram"""
    max_probs = y_pred_probs.max(axis=1)
    ax.hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=uncertainty_threshold, color='red', linestyle='--', 
               label='Uncertainty Threshold')
    ax.set_xlabel('Max Predicted Probability')
    ax.set_title('Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_class_probability_distribution(ax, y_pred_probs, target_names):
    """Plot probability distribution for each class"""
    for i, class_name in enumerate(target_names):
        ax.hist(y_pred_probs[:, i], bins=20, alpha=0.6, label=class_name, density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_title('Probability Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_calibration_curves(ax, y_true, y_pred_probs, target_names):
    """Plot calibration curves for each class"""
    for i, class_name in enumerate(target_names):
        y_binary = (y_true == i).astype(int)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, y_pred_probs[:, i], n_bins=10, strategy='uniform'
        )
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                label=class_name, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)


def calculate_confidence_accuracy(y_pred_probs, y_pred, y_true, n_bins=10):
    """Calculate accuracy at different confidence levels"""
    max_probs = y_pred_probs.max(axis=1)
    correct = (y_pred == y_true).astype(int)
    confidence_bins = np.linspace(0, 1, n_bins + 1)
    
    bin_accuracies, bin_confidences = [], []
    
    for i in range(len(confidence_bins) - 1):
        bin_mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
        if bin_mask.sum() > 0:
            bin_accuracies.append(correct[bin_mask].mean())
            bin_confidences.append(max_probs[bin_mask].mean())
    
    return bin_confidences, bin_accuracies


def plot_confidence_vs_accuracy(ax, y_pred_probs, y_pred, y_true):
    """Plot confidence vs accuracy"""
    bin_confidences, bin_accuracies = calculate_confidence_accuracy(
        y_pred_probs, y_pred, y_true
    )
    
    ax.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Confidence vs Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_calibration_analysis(y_true, y_pred, y_pred_probs, target_names, 
                                save_path='data/calibration_analysis.png'):
    """Create comprehensive calibration analysis visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot all components
    plot_confidence_distribution(axes[0, 0], y_pred_probs)
    plot_class_probability_distribution(axes[0, 1], y_pred_probs, target_names)
    plot_calibration_curves(axes[1, 0], y_true, y_pred_probs, target_names)
    plot_confidence_vs_accuracy(axes[1, 1], y_pred_probs, y_pred, y_true)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Calibration analysis saved → {save_path}")


# ============================================================================
# Uncertainty Analysis
# ============================================================================

def analyze_uncertainty(y_pred_probs, uncertainty_threshold=0.55):
    """Analyze predictions based on uncertainty threshold"""
    max_probs = y_pred_probs.max(axis=1)
    uncertain_mask = max_probs < uncertainty_threshold
    uncertain_count = uncertain_mask.sum()
    total_count = len(y_pred_probs)
    
    print(f"\nUncertainty Analysis (threshold={uncertainty_threshold}):")
    print(f"  Uncertain: {uncertain_count} ({uncertain_count/total_count*100:.1f}%)")
    print(f"  Confident: {total_count - uncertain_count} "
          f"({(1-uncertain_count/total_count)*100:.1f}%)")
    
    return uncertain_mask, uncertain_count


# ============================================================================
# Save Functions
# ============================================================================

def save_model_artifacts(model, label_encoder, 
                        model_path='models/resume_classifier_model.h5',
                        encoder_path='models/label_encoder.pkl'):
    """Save model and label encoder"""
    import os
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved → {model_path}")
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved → {encoder_path}")


def save_evaluation_metrics(test_loss, test_accuracy, f1_macro, f1_weighted, 
                           cm, report_dict, filepath='data/evaluation_metrics.json'):
    """Save evaluation metrics to JSON file"""
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    evaluation_metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict
    }
    
    with open(filepath, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    print(f"Evaluation metrics saved → {filepath}")


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

# def main(model, test_inputs, y_test, label_encoder, uncertainty_threshold=0.55):
#     """Main function to execute model evaluation pipeline"""
#     print("\n=== Step 7: Model Evaluation ===")
    
#     # Get predictions
#     y_pred_probs, y_pred, y_true = get_predictions(model, test_inputs, y_test)
    
#     # Evaluate model
#     test_loss, test_accuracy = evaluate_model(model, test_inputs, y_test)
    
#     # Calculate metrics
#     target_names = label_encoder.classes_
#     report, report_dict, cm, f1_macro, f1_weighted = calculate_classification_metrics(
#         y_true, y_pred, target_names
#     )
    
#     # Print results
#     print_evaluation_results(test_loss, test_accuracy, report)
    
#     # Visualizations
#     plot_confusion_matrix(cm, target_names)
#     create_calibration_analysis(y_true, y_pred, y_pred_probs, target_names)
    
#     # Uncertainty analysis
#     uncertain_mask, uncertain_count = analyze_uncertainty(y_pred_probs, uncertainty_threshold)
    
#     # Save artifacts
#     save_model_artifacts(model, label_encoder)
#     save_evaluation_metrics(test_loss, test_accuracy, f1_macro, f1_weighted, 
#                           cm, report_dict)
    
#     print("\n✓ Evaluation complete → models/resume_classifier_model.h5, evaluation_metrics.json")
    
#     return {
#         'y_pred_probs': y_pred_probs,
#         'y_pred': y_pred,
#         'y_true': y_true,
#         'test_loss': test_loss,
#         'test_accuracy': test_accuracy,
#         'f1_macro': f1_macro,
#         'f1_weighted': f1_weighted,
#         'confusion_matrix': cm,
#         'report_dict': report_dict,
#         'uncertain_mask': uncertain_mask
#     }


# ============================================================================
# Jupyter Notebook Usage
# ============================================================================

# This module contains evaluation functions to be used after training
# Simply run the cells with function definitions, then call the functions
# 
# In your notebook, after training:
# results = main(model_classifier.model, test_inputs, y_test, label_encoder)
# results = main(model_classifier.model, test_inputs, y_test, label_encoder)