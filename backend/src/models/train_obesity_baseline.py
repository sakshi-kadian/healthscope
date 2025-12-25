"""
HealthScope - Baseline Model Training
Dataset: Obesity

Objective: Train a baseline Logistic Regression model for obesity level prediction
and establish performance benchmarks.

Note: This is a multi-class classification problem (7 obesity categories).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import joblib
import os

print("="*70)
print("HEALTHSCOPE - BASELINE MODEL TRAINING: OBESITY")
print("="*70)

# ============================================================================
# 1. LOAD FINAL DATASET
# ============================================================================
print("\n1. Loading Final Dataset...")
df = pd.read_csv('data/processed/obesity_final.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Features: {df.shape[1] - 1}")
print(f"  Target: nobeyesdad")

# ============================================================================
# 2. PREPARE DATA FOR MODELING
# ============================================================================
print("\n2. Preparing Data for Modeling...")

# Separate features and target
X = df.drop('nobeyesdad', axis=1)
y = df['nobeyesdad']

# Drop original categorical columns (keep only encoded versions)
categorical_cols = ['gender', 'family_history_with_overweight', 'favc', 'caec', 'smoke', 'scc', 'calc', 'mtrans']
X = X.drop(columns=categorical_cols, errors='ignore')

print(f"✓ Data prepared")
print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Target classes: {y.nunique()}")
print(f"\n  Target distribution:")
for category in sorted(y.unique()):
    count = (y == category).sum()
    pct = count / len(y) * 100
    print(f"    {category}: {count} ({pct:.1f}%)")

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
print("\n4. Training Baseline Model (Logistic Regression - Multi-class)...")

# Initialize model for multi-class classification
model = LogisticRegression(
    random_state=42, 
    max_iter=1000
)

# Train model
model.fit(X_train, y_train)

print("✓ Model trained successfully!")

# ============================================================================
# 5. MAKE PREDICTIONS
# ============================================================================
print("\n5. Making Predictions...")

# Predictions on training set
y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)

# Predictions on test set
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

print("✓ Predictions completed")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n6. Evaluating Model Performance...")

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# For multi-class ROC-AUC, we need to binarize labels
y_train_bin = label_binarize(y_train, classes=model.classes_)
train_roc_auc = roc_auc_score(y_train_bin, y_train_proba, average='weighted', multi_class='ovr')

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

y_test_bin = label_binarize(y_test, classes=model.classes_)
test_roc_auc = roc_auc_score(y_test_bin, y_test_proba, average='weighted', multi_class='ovr')

print("\n" + "="*70)
print("MODEL PERFORMANCE - TRAINING SET")
print("="*70)
print(f"Accuracy:  {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f} (weighted)")
print(f"Recall:    {train_recall:.4f} (weighted)")
print(f"F1-Score:  {train_f1:.4f} (weighted)")
print(f"ROC-AUC:   {train_roc_auc:.4f} (weighted, one-vs-rest)")
print("="*70)

print("\n" + "="*70)
print("MODEL PERFORMANCE - TEST SET")
print("="*70)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f} (weighted)")
print(f"Recall:    {test_recall:.4f} (weighted)")
print(f"F1-Score:  {test_f1:.4f} (weighted)")
print(f"ROC-AUC:   {test_roc_auc:.4f} (weighted, one-vs-rest)")
print("="*70)

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Classification Report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# ============================================================================
# 7. SAVE MODEL
# ============================================================================
print("\n7. Saving Model...")

# Create models directory if it doesn't exist
os.makedirs('models_saved', exist_ok=True)

# Save model
model_path = 'models_saved/obesity_baseline_lr.pkl'
joblib.dump(model, model_path)

print(f"✓ Model saved to: {model_path}")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n8. Saving Results...")

# Create results dictionary
results = {
    'model': 'Logistic Regression (Baseline)',
    'dataset': 'Obesity',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'num_classes': y.nunique(),
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
results_path = 'models_saved/obesity_baseline_results.csv'
results_df.to_csv(results_path, index=False)

print(f"✓ Results saved to: {results_path}")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("BASELINE MODEL SUMMARY - OBESITY")
print("="*70)
print(f"Model: Logistic Regression (Multi-class)")
print(f"Features: {X.shape[1]}")
print(f"Classes: {y.nunique()}")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print(f"\nTest Performance:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  ROC-AUC:   {test_roc_auc:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print("="*70)

print("\n✓ Obesity baseline model training complete!")
