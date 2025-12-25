"""
HealthScope - Baseline Heart Disease Model Training
This script trains a Logistic Regression baseline model using corrected labels.

Note: Cleveland Heart Disease dataset has inverted labels (0=disease, 1=no disease).
This script uses the corrected dataset where labels are flipped to standard format.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

print("="*70)
print("HEALTHSCOPE - BASELINE HEART MODEL TRAINING")
print("="*70)

# Load dataset
print("\n1. Loading Dataset...")
df = pd.read_csv('data/processed/heart_final.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Prepare data
print("\n2. Preparing Data...")
X = df.drop('target', axis=1)
y = df['target']

print(f"Original target distribution:")
print(f"  No Disease (0): {(y == 0).sum()}")
print(f"  Disease (1): {(y == 1).sum()}")

# Split data
print("\n3. Splitting Data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training: {len(X_train)}, Test: {len(X_test)}")

# Train model
print("\n4. Training Logistic Regression...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("✓ Model trained!")

# Evaluate
print("\n5. Evaluating...")
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nTest Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  ROC-AUC:  {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Disease', 'Disease']))

# Save model
print("\n6. Saving Model...")
os.makedirs('models_saved', exist_ok=True)
model_path = 'models_saved/heart_baseline_lr.pkl'
joblib.dump(model, model_path)
print(f"✓ Model saved to: {model_path}")

print("\n" + "="*70)
print("✓ BASELINE HEART MODEL TRAINING COMPLETE!")
print("="*70)
