"""
HealthScope - Baseline Model Training
Dataset: Diabetes

Objective: Train a baseline Logistic Regression model for diabetes prediction
and establish performance benchmarks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

print("="*70)
print("HEALTHSCOPE - BASELINE MODEL TRAINING: DIABETES")
print("="*70)

# ============================================================================
# 1. LOAD FINAL DATASET
# ============================================================================
print("\n1. Loading Final Dataset...")
df = pd.read_csv('data/processed/diabetes_final.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Features: {df.shape[1] - 1}")
print(f"  Target: outcome")

# ============================================================================
# 2. PREPARE DATA FOR MODELING
# ============================================================================
print("\n2. Preparing Data for Modeling...")

# Separate features and target
X = df.drop('outcome', axis=1)
y = df['outcome']

print(f"✓ Data prepared")
print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Target distribution:")
print(f"    No Diabetes (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"    Diabetes (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n3. Splitting Data (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Data split completed")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# ============================================================================
# 4. TRAIN BASELINE MODEL (LOGISTIC REGRESSION)
# ============================================================================
print("\n4. Training Baseline Model (Logistic Regression)...")

# Initialize model
model = LogisticRegression(random_state=42, max_iter=1000)

# Train model
model.fit(X_train, y_train)

print("✓ Model trained successfully!")

# ============================================================================
# 5. MAKE PREDICTIONS
# ============================================================================
print("\n5. Making Predictions...")

# Predictions on training set
y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)[:, 1]

# Predictions on test set
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

print("✓ Predictions completed")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n6. Evaluating Model Performance...")

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_proba)

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

print("\n" + "="*70)
print("MODEL PERFORMANCE - TRAINING SET")
print("="*70)
print(f"Accuracy:  {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"F1-Score:  {train_f1:.4f}")
print(f"ROC-AUC:   {train_roc_auc:.4f}")
print("="*70)

print("\n" + "="*70)
print("MODEL PERFORMANCE - TEST SET")
print("="*70)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"ROC-AUC:   {test_roc_auc:.4f}")
print("="*70)

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives:  {cm[1][1]}")

# Classification Report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['No Diabetes', 'Diabetes']))

# ============================================================================
# 7. SAVE MODEL
# ============================================================================
print("\n7. Saving Model...")

# Create models directory if it doesn't exist
os.makedirs('models_saved', exist_ok=True)

# Save model
model_path = 'models_saved/diabetes_baseline_lr.pkl'
joblib.dump(model, model_path)

print(f"✓ Model saved to: {model_path}")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n8. Saving Results...")

# Create results dictionary
results = {
    'model': 'Logistic Regression (Baseline)',
    'dataset': 'Diabetes',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_accuracy': train_accuracy,
    'train_precision': train_precision,
    'train_recall': train_recall,
    'train_f1': train_f1,
    'train_roc_auc': train_roc_auc,
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_roc_auc': test_roc_auc
}

# Save to CSV
results_df = pd.DataFrame([results])
results_path = 'models_saved/diabetes_baseline_results.csv'
results_df.to_csv(results_path, index=False)

print(f"✓ Results saved to: {results_path}")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("BASELINE MODEL SUMMARY - DIABETES")
print("="*70)
print(f"Model: Logistic Regression")
print(f"Features: {X.shape[1]}")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print(f"\nTest Performance:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  ROC-AUC:   {test_roc_auc:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print("="*70)

print("\n✓ Diabetes baseline model training complete!")
