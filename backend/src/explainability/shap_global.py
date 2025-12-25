"""
HealthScope - SHAP Explainability
Objective: Generate global feature importance and explanations for all models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*70)
print("HEALTHSCOPE - SHAP EXPLAINABILITY")
print("="*70)

# Create directories
os.makedirs('reports/figures/shap', exist_ok=True)

# ============================================================================
# 1. DIABETES MODEL - SHAP ANALYSIS
# ============================================================================
print("\n1. Analyzing Diabetes Model...")

# Load model and data
diabetes_model = joblib.load('models_saved/diabetes_model.pkl')
diabetes_data = pd.read_csv('data/processed/diabetes_final.csv')

# Prepare features
categorical_cols = ['gender', 'family_history_with_overweight', 'favc', 'caec', 'smoke', 'scc', 'calc', 'mtrans']
X_diabetes = diabetes_data.drop('outcome', axis=1).drop(columns=categorical_cols, errors='ignore')

# Sample data for SHAP (use 100 samples for speed)
X_diabetes_sample = X_diabetes.sample(min(100, len(X_diabetes)), random_state=42)

print(f"  Features: {X_diabetes.shape[1]}")
print(f"  Sample size: {len(X_diabetes_sample)}")

# Create SHAP explainer
print("  Creating SHAP explainer...")
explainer_diabetes = shap.TreeExplainer(diabetes_model)
shap_values_diabetes = explainer_diabetes.shap_values(X_diabetes_sample)

# Handle binary classification (shap_values might be 3D for RandomForest)
if len(shap_values_diabetes.shape) == 3:
    # For binary classification, use positive class (index 1)
    shap_values_diabetes = shap_values_diabetes[:, :, 1]
elif isinstance(shap_values_diabetes, list):
    shap_values_diabetes = shap_values_diabetes[1]  # Use positive class

print("  ✓ SHAP values calculated")

# Summary plot
print("  Creating summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_diabetes, X_diabetes_sample, show=False)
plt.title('SHAP Feature Importance - Diabetes Model', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/shap/diabetes_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Summary plot saved")

# Bar plot
print("  Creating bar plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_diabetes, X_diabetes_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Mean) - Diabetes Model', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/shap/diabetes_shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Bar plot saved")

# Get feature importance
importance_values = np.abs(shap_values_diabetes).mean(axis=0)
feature_importance_diabetes = pd.DataFrame({
    'feature': list(X_diabetes_sample.columns),
    'importance': importance_values
}).sort_values('importance', ascending=False)

print("\n  Top 5 Features:")
for idx, row in feature_importance_diabetes.head(5).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# 2. HEART DISEASE MODEL - SHAP ANALYSIS
# ============================================================================
print("\n2. Analyzing Heart Disease Model...")

# Load model and data
heart_model = joblib.load('models_saved/heart_model.pkl')
heart_data = pd.read_csv('data/processed/heart_final.csv')

X_heart = heart_data.drop('target', axis=1)
X_heart_sample = X_heart.sample(min(100, len(X_heart)), random_state=42)

print(f"  Features: {X_heart.shape[1]}")
print(f"  Sample size: {len(X_heart_sample)}")

# For Logistic Regression, use LinearExplainer
print("  Creating SHAP explainer...")
explainer_heart = shap.LinearExplainer(heart_model, X_heart_sample)
shap_values_heart = explainer_heart.shap_values(X_heart_sample)

print("  ✓ SHAP values calculated")

# Summary plot
print("  Creating summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_heart, X_heart_sample, show=False)
plt.title('SHAP Feature Importance - Heart Disease Model', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/shap/heart_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Summary plot saved")

# Bar plot
print("  Creating bar plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_heart, X_heart_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Mean) - Heart Disease Model', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/shap/heart_shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Bar plot saved")

# Get feature importance
importance_values_heart = np.abs(shap_values_heart).mean(axis=0)
feature_importance_heart = pd.DataFrame({
    'feature': list(X_heart_sample.columns),
    'importance': importance_values_heart
}).sort_values('importance', ascending=False)

print("\n  Top 5 Features:")
for idx, row in feature_importance_heart.head(5).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# 3. OBESITY MODEL - SHAP ANALYSIS
# ============================================================================
print("\n3. Analyzing Obesity Model...")

# Load model and data
obesity_model = joblib.load('models_saved/obesity_model.pkl')
obesity_data = pd.read_csv('data/processed/obesity_final.csv')

X_obesity = obesity_data.drop('nobeyesdad', axis=1).drop(columns=categorical_cols, errors='ignore')
X_obesity_sample = X_obesity.sample(min(100, len(X_obesity)), random_state=42)

print(f"  Features: {X_obesity.shape[1]}")
print(f"  Sample size: {len(X_obesity_sample)}")

# Create SHAP explainer
print("  Creating SHAP explainer...")
explainer_obesity = shap.TreeExplainer(obesity_model)
shap_values_obesity = explainer_obesity.shap_values(X_obesity_sample)

print("  ✓ SHAP values calculated")

# For multi-class, SHAP returns 3D array (samples, features, classes)
# Use class 0 for visualization
if len(shap_values_obesity.shape) == 3:
    shap_values_obesity_viz = shap_values_obesity[:, :, 0]
elif isinstance(shap_values_obesity, list):
    shap_values_obesity_viz = shap_values_obesity[0]
else:
    shap_values_obesity_viz = shap_values_obesity

# Summary plot
print("  Creating summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_obesity_viz, X_obesity_sample, show=False)
plt.title('SHAP Feature Importance - Obesity Model (Class 0)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/shap/obesity_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Summary plot saved")

# Bar plot
print("  Creating bar plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_obesity_viz, X_obesity_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Mean) - Obesity Model', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/shap/obesity_shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Bar plot saved")

# Get feature importance
# For multi-class, calculate mean absolute SHAP across all classes
if len(shap_values_obesity.shape) == 3:
    # Average absolute SHAP values across all classes
    mean_abs_shap = np.abs(shap_values_obesity).mean(axis=(0, 2))  # Average over samples and classes
elif isinstance(shap_values_obesity, list):
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_obesity], axis=0)
else:
    mean_abs_shap = np.abs(shap_values_obesity_viz).mean(axis=0)

feature_importance_obesity = pd.DataFrame({
    'feature': list(X_obesity_sample.columns),
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False)

print("\n  Top 5 Features:")
for idx, row in feature_importance_obesity.head(5).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# 4. SAVE FEATURE IMPORTANCE RANKINGS
# ============================================================================
print("\n4. Saving Feature Importance Rankings...")

# Save to CSV
feature_importance_diabetes.to_csv('reports/diabetes_feature_importance.csv', index=False)
feature_importance_heart.to_csv('reports/heart_feature_importance.csv', index=False)
feature_importance_obesity.to_csv('reports/obesity_feature_importance.csv', index=False)

print("✓ Feature importance rankings saved")

# ============================================================================
# 5. CREATE COMPARISON VISUALIZATION
# ============================================================================
print("\n5. Creating Feature Importance Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Diabetes
top_diabetes = feature_importance_diabetes.head(10)
axes[0].barh(range(len(top_diabetes)), top_diabetes['importance'], color='steelblue')
axes[0].set_yticks(range(len(top_diabetes)))
axes[0].set_yticklabels(top_diabetes['feature'])
axes[0].invert_yaxis()
axes[0].set_xlabel('Mean |SHAP Value|', fontweight='bold')
axes[0].set_title('Diabetes - Top 10 Features', fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Heart
top_heart = feature_importance_heart.head(10)
axes[1].barh(range(len(top_heart)), top_heart['importance'], color='coral')
axes[1].set_yticks(range(len(top_heart)))
axes[1].set_yticklabels(top_heart['feature'])
axes[1].invert_yaxis()
axes[1].set_xlabel('Mean |SHAP Value|', fontweight='bold')
axes[1].set_title('Heart Disease - Top 10 Features', fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

# Obesity
top_obesity = feature_importance_obesity.head(10)
axes[2].barh(range(len(top_obesity)), top_obesity['importance'], color='mediumseagreen')
axes[2].set_yticks(range(len(top_obesity)))
axes[2].set_yticklabels(top_obesity['feature'])
axes[2].invert_yaxis()
axes[2].set_xlabel('Mean |SHAP Value|', fontweight='bold')
axes[2].set_title('Obesity - Top 10 Features', fontweight='bold')
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/shap/all_models_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Comparison visualization saved")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SHAP EXPLAINABILITY SUMMARY")
print("="*70)
print(f"Models Analyzed: 3")
print(f"Visualizations Created: 7")
print(f"\nTop Contributing Features:")
print(f"\nDiabetes:")
print(f"  1. {feature_importance_diabetes.iloc[0]['feature']}")
print(f"  2. {feature_importance_diabetes.iloc[1]['feature']}")
print(f"  3. {feature_importance_diabetes.iloc[2]['feature']}")
print(f"\nHeart Disease:")
print(f"  1. {feature_importance_heart.iloc[0]['feature']}")
print(f"  2. {feature_importance_heart.iloc[1]['feature']}")
print(f"  3. {feature_importance_heart.iloc[2]['feature']}")
print(f"\nObesity:")
print(f"  1. {feature_importance_obesity.iloc[0]['feature']}")
print(f"  2. {feature_importance_obesity.iloc[1]['feature']}")
print(f"  3. {feature_importance_obesity.iloc[2]['feature']}")
print("="*70)

print("\n✓ SHAP explainability analysis complete!")
print("\nGenerated files:")
print("  - reports/figures/shap/diabetes_shap_summary.png")
print("  - reports/figures/shap/diabetes_shap_bar.png")
print("  - reports/figures/shap/heart_shap_summary.png")
print("  - reports/figures/shap/heart_shap_bar.png")
print("  - reports/figures/shap/obesity_shap_summary.png")
print("  - reports/figures/shap/obesity_shap_bar.png")
print("  - reports/figures/shap/all_models_feature_importance.png")
print("  - reports/diabetes_feature_importance.csv")
print("  - reports/heart_feature_importance.csv")
print("  - reports/obesity_feature_importance.csv")
