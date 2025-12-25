"""
HealthScope - Advanced Model Training
Dataset: Diabetes

Objective: Train Random Forest and XGBoost models, compare with baseline,
and select the best model based on ROC-AUC score.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

print("="*70)
print("HEALTHSCOPE - ADVANCED MODEL TRAINING: DIABETES")
print("="*70)

# ============================================================================
# 1. LOAD FINAL DATASET
# ============================================================================
print("\n1. Loading Final Dataset...")
df = pd.read_csv('data/processed/diabetes_final.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
print("\n2. Preparing Data...")

X = df.drop('outcome', axis=1)
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Data prepared")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# ============================================================================
# 3. TRAIN RANDOM FOREST MODEL
# ============================================================================
print("\n3. Training Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✓ Random Forest trained!")

# Predictions
rf_train_pred = rf_model.predict(X_train)
rf_train_proba = rf_model.predict_proba(X_train)[:, 1]
rf_test_pred = rf_model.predict(X_test)
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Metrics
rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_train_roc = roc_auc_score(y_train, rf_train_proba)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
rf_test_precision = precision_score(y_test, rf_test_pred)
rf_test_recall = recall_score(y_test, rf_test_pred)
rf_test_f1 = f1_score(y_test, rf_test_pred)
rf_test_roc = roc_auc_score(y_test, rf_test_proba)

print(f"\nRandom Forest Performance:")
print(f"  Train Accuracy: {rf_train_acc:.4f}, ROC-AUC: {rf_train_roc:.4f}")
print(f"  Test Accuracy:  {rf_test_acc:.4f}, ROC-AUC: {rf_test_roc:.4f}")

# ============================================================================
# 4. TRAIN XGBOOST MODEL
# ============================================================================
print("\n4. Training XGBoost Model...")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
print("✓ XGBoost trained!")

# Predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]
xgb_test_pred = xgb_model.predict(X_test)
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]

# Metrics
xgb_train_acc = accuracy_score(y_train, xgb_train_pred)
xgb_train_roc = roc_auc_score(y_train, xgb_train_proba)
xgb_test_acc = accuracy_score(y_test, xgb_test_pred)
xgb_test_precision = precision_score(y_test, xgb_test_pred)
xgb_test_recall = recall_score(y_test, xgb_test_pred)
xgb_test_f1 = f1_score(y_test, xgb_test_pred)
xgb_test_roc = roc_auc_score(y_test, xgb_test_proba)

print(f"\nXGBoost Performance:")
print(f"  Train Accuracy: {xgb_train_acc:.4f}, ROC-AUC: {xgb_train_roc:.4f}")
print(f"  Test Accuracy:  {xgb_test_acc:.4f}, ROC-AUC: {xgb_test_roc:.4f}")

# ============================================================================
# 5. CROSS-VALIDATION
# ============================================================================
print("\n5. Performing 5-Fold Cross-Validation...")

# Random Forest CV
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"\nRandom Forest CV ROC-AUC: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# XGBoost CV
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"XGBoost CV ROC-AUC:       {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std():.4f})")

# ============================================================================
# 6. COMPARE MODELS
# ============================================================================
print("\n6. Comparing All Models...")

# Load baseline results
baseline_results = pd.read_csv('models_saved/diabetes_baseline_results.csv')
baseline_roc = baseline_results['test_roc_auc'].values[0]

print("\n" + "="*70)
print("MODEL COMPARISON - TEST SET")
print("="*70)
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
print("-"*70)
print(f"{'Logistic Regression':<25} {baseline_results['test_accuracy'].values[0]:<12.4f} {baseline_results['test_precision'].values[0]:<12.4f} {baseline_results['test_recall'].values[0]:<12.4f} {baseline_results['test_f1'].values[0]:<12.4f} {baseline_roc:<12.4f}")
print(f"{'Random Forest':<25} {rf_test_acc:<12.4f} {rf_test_precision:<12.4f} {rf_test_recall:<12.4f} {rf_test_f1:<12.4f} {rf_test_roc:<12.4f}")
print(f"{'XGBoost':<25} {xgb_test_acc:<12.4f} {xgb_test_precision:<12.4f} {xgb_test_recall:<12.4f} {xgb_test_f1:<12.4f} {xgb_test_roc:<12.4f}")
print("="*70)

# ============================================================================
# 7. SELECT BEST MODEL
# ============================================================================
print("\n7. Selecting Best Model (Based on ROC-AUC)...")

models = {
    'Logistic Regression': baseline_roc,
    'Random Forest': rf_test_roc,
    'XGBoost': xgb_test_roc
}

best_model_name = max(models, key=models.get)
best_roc_auc = models[best_model_name]

print(f"\n✓ Best Model: {best_model_name}")
print(f"  ROC-AUC: {best_roc_auc:.4f}")

# Select the best model object
if best_model_name == 'Random Forest':
    best_model = rf_model
    best_pred = rf_test_pred
elif best_model_name == 'XGBoost':
    best_model = xgb_model
    best_pred = xgb_test_pred
else:
    # Load baseline model
    best_model = joblib.load('models_saved/diabetes_baseline_lr.pkl')
    best_pred = best_model.predict(X_test)

# ============================================================================
# 8. SAVE BEST MODEL
# ============================================================================
print("\n8. Saving Best Model...")

os.makedirs('models_saved', exist_ok=True)

# Save best model
best_model_path = 'models_saved/diabetes_model.pkl'
joblib.dump(best_model, best_model_path)
print(f"✓ Best model saved to: {best_model_path}")

# Save Random Forest
rf_path = 'models_saved/diabetes_rf.pkl'
joblib.dump(rf_model, rf_path)
print(f"✓ Random Forest saved to: {rf_path}")

# Save XGBoost
xgb_path = 'models_saved/diabetes_xgb.pkl'
joblib.dump(xgb_model, xgb_path)
print(f"✓ XGBoost saved to: {xgb_path}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n9. Saving Results...")

# Create results dataframe
all_results = pd.DataFrame([
    {
        'model': 'Logistic Regression',
        'test_accuracy': baseline_results['test_accuracy'].values[0],
        'test_precision': baseline_results['test_precision'].values[0],
        'test_recall': baseline_results['test_recall'].values[0],
        'test_f1': baseline_results['test_f1'].values[0],
        'test_roc_auc': baseline_roc,
        'cv_roc_auc_mean': None,
        'cv_roc_auc_std': None
    },
    {
        'model': 'Random Forest',
        'test_accuracy': rf_test_acc,
        'test_precision': rf_test_precision,
        'test_recall': rf_test_recall,
        'test_f1': rf_test_f1,
        'test_roc_auc': rf_test_roc,
        'cv_roc_auc_mean': rf_cv_scores.mean(),
        'cv_roc_auc_std': rf_cv_scores.std()
    },
    {
        'model': 'XGBoost',
        'test_accuracy': xgb_test_acc,
        'test_precision': xgb_test_precision,
        'test_recall': xgb_test_recall,
        'test_f1': xgb_test_f1,
        'test_roc_auc': xgb_test_roc,
        'cv_roc_auc_mean': xgb_cv_scores.mean(),
        'cv_roc_auc_std': xgb_cv_scores.std()
    }
])

results_path = 'models_saved/diabetes_all_models_results.csv'
all_results.to_csv(results_path, index=False)
print(f"✓ All results saved to: {results_path}")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ADVANCED MODEL TRAINING SUMMARY - DIABETES")
print("="*70)
print(f"Best Model: {best_model_name}")
print(f"Best ROC-AUC: {best_roc_auc:.4f}")
print(f"\nAll models saved to: models_saved/")
print(f"  - diabetes_model.pkl (best model)")
print(f"  - diabetes_baseline_lr.pkl")
print(f"  - diabetes_rf.pkl")
print(f"  - diabetes_xgb.pkl")
print("="*70)

print("\n✓ Diabetes advanced model training complete!")
