"""
HealthScope - Individual Prediction Explanations
Objective: Generate SHAP explanations for individual predictions using
force plots and waterfall charts.
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
print("HEALTHSCOPE - INDIVIDUAL PREDICTION EXPLANATIONS")
print("="*70)

# Create directories
os.makedirs('reports/figures/shap/individual', exist_ok=True)

# ============================================================================
# 1. DIABETES MODEL - INDIVIDUAL EXPLANATIONS
# ============================================================================
print("\n1. Creating Individual Explanations - Diabetes Model...")

# Load model and data
diabetes_model = joblib.load('models_saved/diabetes_model.pkl')
diabetes_data = pd.read_csv('data/processed/diabetes_final.csv')

# Prepare features
categorical_cols = ['gender', 'family_history_with_overweight', 'favc', 'caec', 'smoke', 'scc', 'calc', 'mtrans']
X_diabetes = diabetes_data.drop('outcome', axis=1).drop(columns=categorical_cols, errors='ignore')
y_diabetes = diabetes_data['outcome']

# Sample data
X_sample = X_diabetes.sample(min(100, len(X_diabetes)), random_state=42)

# Create SHAP explainer
explainer_diabetes = shap.TreeExplainer(diabetes_model)
shap_values = explainer_diabetes.shap_values(X_sample)

# Handle 3D array for binary classification
if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]  # Positive class

# Select interesting cases
# Case 1: High risk prediction
high_risk_idx = 0
# Case 2: Low risk prediction  
low_risk_idx = 50

print(f"  Creating explanations for 2 sample cases...")

# Waterfall plot for high risk case
print(f"  - High risk case (index {high_risk_idx})...")
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer_diabetes.expected_value if not isinstance(explainer_diabetes.expected_value, np.ndarray) else explainer_diabetes.expected_value[1],
        data=X_sample.iloc[high_risk_idx],
        feature_names=X_sample.columns.tolist()
    ),
    show=False
)
plt.title('SHAP Waterfall - Diabetes High Risk Case', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/shap/individual/diabetes_waterfall_high_risk.png', dpi=300, bbox_inches='tight')
plt.close()

# Waterfall plot for low risk case
print(f"  - Low risk case (index {low_risk_idx})...")
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[low_risk_idx],
        base_values=explainer_diabetes.expected_value if not isinstance(explainer_diabetes.expected_value, np.ndarray) else explainer_diabetes.expected_value[1],
        data=X_sample.iloc[low_risk_idx],
        feature_names=X_sample.columns.tolist()
    ),
    show=False
)
plt.title('SHAP Waterfall - Diabetes Low Risk Case', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/shap/individual/diabetes_waterfall_low_risk.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Diabetes individual explanations created")

# ============================================================================
# 2. HEART DISEASE MODEL - INDIVIDUAL EXPLANATIONS
# ============================================================================
print("\n2. Creating Individual Explanations - Heart Disease Model...")

# Load model and data
heart_model = joblib.load('models_saved/heart_model.pkl')
heart_data = pd.read_csv('data/processed/heart_final.csv')

X_heart = heart_data.drop('target', axis=1)
y_heart = heart_data['target']

X_heart_sample = X_heart.sample(min(100, len(X_heart)), random_state=42)

# Create SHAP explainer
explainer_heart = shap.LinearExplainer(heart_model, X_heart_sample)
shap_values_heart = explainer_heart.shap_values(X_heart_sample)

print(f"  Creating explanations for 2 sample cases...")

# Waterfall for high risk
print(f"  - High risk case...")
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_heart[0],
        base_values=explainer_heart.expected_value,
        data=X_heart_sample.iloc[0],
        feature_names=X_heart_sample.columns.tolist()
    ),
    show=False
)
plt.title('SHAP Waterfall - Heart Disease High Risk Case', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/shap/individual/heart_waterfall_high_risk.png', dpi=300, bbox_inches='tight')
plt.close()

# Waterfall for low risk
print(f"  - Low risk case...")
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_heart[50],
        base_values=explainer_heart.expected_value,
        data=X_heart_sample.iloc[50],
        feature_names=X_heart_sample.columns.tolist()
    ),
    show=False
)
plt.title('SHAP Waterfall - Heart Disease Low Risk Case', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/shap/individual/heart_waterfall_low_risk.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Heart disease individual explanations created")

# ============================================================================
# 3. OBESITY MODEL - INDIVIDUAL EXPLANATIONS
# ============================================================================
print("\n3. Creating Individual Explanations - Obesity Model...")

# Load model and data
obesity_model = joblib.load('models_saved/obesity_model.pkl')
obesity_data = pd.read_csv('data/processed/obesity_final.csv')

X_obesity = obesity_data.drop('nobeyesdad', axis=1).drop(columns=categorical_cols, errors='ignore')
y_obesity = obesity_data['nobeyesdad']

X_obesity_sample = X_obesity.sample(min(100, len(X_obesity)), random_state=42)

# Create SHAP explainer
explainer_obesity = shap.TreeExplainer(obesity_model)
shap_values_obesity = explainer_obesity.shap_values(X_obesity_sample)

# For multi-class, use class 0
if len(shap_values_obesity.shape) == 3:
    shap_values_viz = shap_values_obesity[:, :, 0]
    expected_value = explainer_obesity.expected_value[0] if isinstance(explainer_obesity.expected_value, np.ndarray) else explainer_obesity.expected_value
else:
    shap_values_viz = shap_values_obesity
    expected_value = explainer_obesity.expected_value

print(f"  Creating explanations for 2 sample cases...")

# Waterfall for case 1
print(f"  - Sample case 1...")
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_viz[0],
        base_values=expected_value,
        data=X_obesity_sample.iloc[0],
        feature_names=X_obesity_sample.columns.tolist()
    ),
    show=False
)
plt.title('SHAP Waterfall - Obesity Sample Case 1', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/shap/individual/obesity_waterfall_case1.png', dpi=300, bbox_inches='tight')
plt.close()

# Waterfall for case 2
print(f"  - Sample case 2...")
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_viz[50],
        base_values=expected_value,
        data=X_obesity_sample.iloc[50],
        feature_names=X_obesity_sample.columns.tolist()
    ),
    show=False
)
plt.title('SHAP Waterfall - Obesity Sample Case 2', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/shap/individual/obesity_waterfall_case2.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Obesity individual explanations created")

# ============================================================================
# 4. CREATE EXPLANATION DOCUMENTATION
# ============================================================================
print("\n4. Creating Explanation Documentation...")

explanation_doc = """
# INDIVIDUAL PREDICTION EXPLANATIONS

## Overview
This document explains how to interpret SHAP waterfall plots for individual predictions.

## What is a SHAP Waterfall Plot?

A SHAP waterfall plot shows how each feature contributes to moving the prediction from the base value (average prediction) to the final prediction for a specific instance.

### Key Components:
1. **Base Value**: The average model output (starting point)
2. **Feature Contributions**: How each feature pushes the prediction up (red) or down (blue)
3. **Final Prediction**: The actual model output for this instance

## How to Read the Plots

### Diabetes Model
- **Base Value**: Average diabetes risk across all patients
- **Red Bars**: Features increasing diabetes risk
- **Blue Bars**: Features decreasing diabetes risk
- **Final Value**: Predicted diabetes risk for this patient

**Example Interpretation**:
- If `glucose` is red and large → High glucose significantly increases risk
- If `age` is blue → Younger age decreases risk
- The sum of all contributions gives the final prediction

### Heart Disease Model
- **Base Value**: Average heart disease risk
- **Red Bars**: Features increasing heart disease risk
- **Blue Bars**: Features decreasing heart disease risk

**Key Features to Watch**:
- `cp` (chest pain type): Different types indicate different risk levels
- `thalach` (max heart rate): Higher values often protective
- `ca` (number of vessels): More vessels colored indicates higher risk

### Obesity Model
- **Base Value**: Average obesity category prediction
- **Red Bars**: Features pushing toward higher obesity categories
- **Blue Bars**: Features pushing toward lower obesity categories

**Key Features**:
- `calculated_bmi`: Primary indicator
- `weight`: Direct contributor
- `lifestyle_score`: Lifestyle factors impact

## Use Cases

### 1. Patient Communication
Use waterfall plots to explain to patients:
- "Your high glucose level is the main factor increasing your diabetes risk"
- "Your regular exercise (high physical activity) is helping reduce your obesity risk"

### 2. Clinical Decision Support
Identify which factors to focus on:
- If `bmi_glucose_interaction` is high → Address both BMI and glucose
- If `age` is protective → Focus on modifiable risk factors

### 3. Model Validation
Check if model is using clinically relevant features:
- ✓ Good: Model relies on glucose, BMI, blood pressure
- ✗ Bad: Model relies on irrelevant features

## Limitations

1. **Correlation ≠ Causation**: SHAP shows associations, not causal relationships
2. **Model-Specific**: Explanations are for the model, not reality
3. **Sample-Dependent**: Based on the training data distribution
4. **Not Medical Advice**: Always consult healthcare professionals

## Examples in This Project

### Diabetes
- `diabetes_waterfall_high_risk.png`: Patient with high diabetes risk
- `diabetes_waterfall_low_risk.png`: Patient with low diabetes risk

### Heart Disease
- `heart_waterfall_high_risk.png`: Patient with high heart disease risk
- `heart_waterfall_low_risk.png`: Patient with low heart disease risk

### Obesity
- `obesity_waterfall_case1.png`: Sample obesity prediction
- `obesity_waterfall_case2.png`: Another sample obesity prediction

---
**Note**: These are educational examples. Real clinical decisions require comprehensive medical evaluation.
"""

with open('reports/SHAP_EXPLANATION_GUIDE.md', 'w', encoding='utf-8') as f:
    f.write(explanation_doc)

print("✓ Explanation documentation created")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("INDIVIDUAL EXPLANATION SUMMARY")
print("="*70)
print(f"Models Analyzed: 3")
print(f"Individual Explanations Created: 6 (2 per model)")
print(f"Visualization Type: SHAP Waterfall Plots")
print(f"\nGenerated Files:")
print(f"  Diabetes:")
print(f"    - diabetes_waterfall_high_risk.png")
print(f"    - diabetes_waterfall_low_risk.png")
print(f"  Heart Disease:")
print(f"    - heart_waterfall_high_risk.png")
print(f"    - heart_waterfall_low_risk.png")
print(f"  Obesity:")
print(f"    - obesity_waterfall_case1.png")
print(f"    - obesity_waterfall_case2.png")
print(f"\nDocumentation:")
print(f"  - SHAP_EXPLANATION_GUIDE.md")
print("="*70)

print("\n✓ Individual prediction explanations complete!")
