import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path


# ============================================================================
# Label Encoding Functions
# ============================================================================

def encode_labels(y, num_classes):
    """Encode labels to categorical format"""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    return label_encoder, y_encoded, y_categorical


# ============================================================================
# Data Splitting Functions
# ============================================================================

def split_train_val_test(X, y_categorical, y_encoded, test_size=0.3, val_size=0.5, random_state=42):
    """Split data into train/validation/test sets with stratification
    
    Args:
        test_size: Proportion for temp (val+test) split
        val_size: Proportion of temp to use for validation (relative to temp)
        
    Example: test_size=0.3, val_size=0.5 gives 70/15/15 split
    """
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, 
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp.argmax(axis=1)
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_split_info(X_train, X_val, X_test):
    """Print information about data splits"""
    print(f"Split: Train={X_train.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}")
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"Proportions: Train={X_train.shape[0]/total:.1%} | "
          f"Val={X_val.shape[0]/total:.1%} | Test={X_test.shape[0]/total:.1%}")


# ============================================================================
# Input Preparation Functions
# ============================================================================

def prepare_all_inputs(model_classifier, X_train, X_val, X_test):
    """Prepare inputs for all data splits"""
    train_inputs = model_classifier.prepare_inputs(X_train)
    val_inputs = model_classifier.prepare_inputs(X_val)
    test_inputs = model_classifier.prepare_inputs(X_test)
    
    return train_inputs, val_inputs, test_inputs


# ============================================================================
# Class Weight Functions
# ============================================================================

def compute_class_weights(y_train_categorical):
    """Compute balanced class weights"""
    y_train_labels = y_train_categorical.argmax(axis=1)
    
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights_array)}
    
    print(f"\nClass weights: {class_weight_dict}")
    return class_weight_dict


# ============================================================================
# Callback Functions
# ============================================================================

def create_training_callbacks(model_dir='models', patience=5):
    """Create training callbacks for early stopping and checkpointing"""
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            restore_best_weights=True, 
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f'{model_dir}/best_resume_classifier.h5', 
            monitor='val_loss', 
            save_best_only=True, 
            verbose=1
        )
    ]
    
    return callbacks


# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, train_inputs, y_train, val_inputs, y_val, 
                class_weight_dict, callbacks, batch_size=32, epochs=50):
    """Train the model with specified parameters"""
    print(f"\nTraining: {epochs} epochs, batch_size={batch_size}")
    
    history = model.fit(
        train_inputs,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_inputs, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# ============================================================================
# Save Functions
# ============================================================================

def save_training_history(history, filepath='data/training_history.json'):
    """Save training history to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    history_dict = {
        key: [float(v) for v in values] 
        for key, values in history.history.items()
    }
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved → {filepath}")


def save_complete_model(model, filepath='models/resume_classifier_complete.h5'):
    """Save complete Keras model"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Complete model saved → {filepath}")


def save_model_config(model_classifier, filepath='artifacts/model_config.json'):
    """Save model configuration for production"""
    Path(os.path.dirname(filepath)).mkdir(exist_ok=True)
    
    model_config = {
        "use_text_branch": model_classifier.use_text_branch,
        "skill_vocab_size": model_classifier.skill_vocab_size,
        "numeric_dim": model_classifier.numeric_dim,
        "text_dim": model_classifier.text_dim,
        "num_classes": model_classifier.num_classes
    }
    
    with open(filepath, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Model config saved → {filepath}")


def save_label_encoder(label_encoder, filepath='artifacts/label_encoder.pkl'):
    """Save label encoder for production use"""
    import pickle
    
    Path(os.path.dirname(filepath)).mkdir(exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Label encoder saved → {filepath}")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_history(history, save_path='data/training_history.png'):
    """Plot training and validation loss/accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train', marker='o')
    axes[0].plot(history.history['val_loss'], label='Val', marker='s')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Train', marker='o')
    axes[1].plot(history.history['val_accuracy'], label='Val', marker='s')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Training history plot saved → {save_path}")


def print_training_summary(history):
    """Print summary of training results"""
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    
    print(f"Final Epoch:")
    print(f"  Train Loss: {final_train_loss:.4f} | Train Acc: {final_train_acc:.4f}")
    print(f"  Val Loss:   {final_val_loss:.4f} | Val Acc:   {final_val_acc:.4f}")
    print(f"\nBest Epoch: {best_epoch}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val Acc:  {best_val_acc:.4f}")
    print("="*70)


# ============================================================================
# Main Training Pipeline
# ============================================================================

# def main(model_classifier, X, y, label_mapping, batch_size=32, epochs=50, patience=5):
#     """Main function to execute training procedure"""
#     print("\n=== Step 6: Training Procedure ===")
    
#     # Encode labels
#     label_encoder, y_encoded, y_categorical = encode_labels(y, len(label_mapping['unique_labels']))
    
#     # Split data
#     X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
#         X, y_categorical, y_encoded
#     )
#     print_split_info(X_train, X_val, X_test)
    
#     # Prepare inputs
#     train_inputs, val_inputs, test_inputs = prepare_all_inputs(
#         model_classifier, X_train, X_val, X_test
#     )
    
#     # Compute class weights
#     class_weight_dict = compute_class_weights(y_train)
    
#     # Create callbacks
#     callbacks = create_training_callbacks(patience=patience)
    
#     # Train model
#     history = train_model(
#         model_classifier.model,
#         train_inputs,
#         y_train,
#         val_inputs,
#         y_val,
#         class_weight_dict,
#         callbacks,
#         batch_size=batch_size,
#         epochs=epochs
#     )
    
#     # Save everything
#     save_training_history(history)
#     save_complete_model(model_classifier.model)
#     save_model_config(model_classifier)
#     save_label_encoder(label_encoder)
    
#     # Visualize and summarize
#     plot_training_history(history)
#     print_training_summary(history)
    
#     print("\n✓ Training complete → models/best_resume_classifier.h5")
    
#     return history, label_encoder, (X_train, X_val, X_test, y_train, y_val, y_test), \
#            (train_inputs, val_inputs, test_inputs)


# ============================================================================
# Jupyter Notebook Usage
# ============================================================================

# This cell defines training functions - just run it to load the functions
# Then call the training in a separate cell:
#
# history, label_encoder, splits, inputs = main(
#     model_classifier, X, y, label_mapping, 
#     batch_size=32, epochs=50, patience=5
# )
# history, label_encoder, splits, inputs = main(
#     model_classifier, X, y, label_mapping, 
#     batch_size=32, epochs=50, patience=5
# )
# # X_train, X_val, X_test, y_train, y_val, y_test = splits
# train_inputs, val_inputs, test_inputs = inputs