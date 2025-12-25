"""
HealthScope - Feature Engineering
Dataset 1: Diabetes

Objective: Create new features from cleaned diabetes dataset to improve
model performance.

Features to engineer:
- BMI categories
- Age groups
- Glucose categories
- Risk score combinations
- Interaction features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("="*70)
print("HEALTHSCOPE - FEATURE ENGINEERING: DIABETES DATASET")
print("="*70)

# ============================================================================
# 1. LOAD CLEANED DATASET
# ============================================================================
print("\n1. Loading Cleaned Dataset...")
df = pd.read_csv('data/processed/diabetes_clean.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# Create a copy for feature engineering
df_features = df.copy()

# ============================================================================
# 2. CREATE BMI CATEGORIES
# ============================================================================
print("\n2. Creating BMI Categories...")

# WHO BMI categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif bmi < 25:
        return 1  # Normal
    elif bmi < 30:
        return 2  # Overweight
    else:
        return 3  # Obese

df_features['bmi_category'] = df_features['bmi'].apply(categorize_bmi)

print("✓ BMI categories created:")
print(df_features['bmi_category'].value_counts().sort_index())

# ============================================================================
# 3. CREATE AGE GROUPS
# ============================================================================
print("\n3. Creating Age Groups...")

# Age groups for diabetes risk
def categorize_age(age):
    if age < 30:
        return 0  # Young
    elif age < 45:
        return 1  # Middle-aged
    elif age < 60:
        return 2  # Senior
    else:
        return 3  # Elderly

df_features['age_group'] = df_features['age'].apply(categorize_age)

print("✓ Age groups created:")
print(df_features['age_group'].value_counts().sort_index())

# ============================================================================
# 4. CREATE GLUCOSE CATEGORIES
# ============================================================================
print("\n4. Creating Glucose Categories...")

# Glucose level categories (mg/dL)
def categorize_glucose(glucose):
    if glucose < 100:
        return 0  # Normal
    elif glucose < 126:
        return 1  # Prediabetes
    else:
        return 2  # Diabetes range

df_features['glucose_category'] = df_features['glucose'].apply(categorize_glucose)

print("✓ Glucose categories created:")
print(df_features['glucose_category'].value_counts().sort_index())

# ============================================================================
# 5. CREATE BLOOD PRESSURE CATEGORIES
# ============================================================================
print("\n5. Creating Blood Pressure Categories...")

# Blood pressure categories (mm Hg)
def categorize_bp(bp):
    if bp < 80:
        return 0  # Low
    elif bp < 90:
        return 1  # Normal
    elif bp < 120:
        return 2  # Elevated
    else:
        return 3  # High

df_features['bp_category'] = df_features['blood_pressure'].apply(categorize_bp)

print("✓ Blood pressure categories created:")
print(df_features['bp_category'].value_counts().sort_index())

# ============================================================================
# 6. CREATE INTERACTION FEATURES
# ============================================================================
print("\n6. Creating Interaction Features...")

# BMI * Glucose interaction (important for diabetes)
df_features['bmi_glucose_interaction'] = df_features['bmi'] * df_features['glucose']

# Age * Glucose interaction
df_features['age_glucose_interaction'] = df_features['age'] * df_features['glucose']

# Pregnancies * Age interaction
df_features['pregnancies_age_interaction'] = df_features['pregnancies'] * df_features['age']

print("✓ Interaction features created:")
print(f"  - bmi_glucose_interaction")
print(f"  - age_glucose_interaction")
print(f"  - pregnancies_age_interaction")

# ============================================================================
# 7. CREATE RISK SCORE
# ============================================================================
print("\n7. Creating Risk Score...")

# Simple risk score based on key factors
df_features['risk_score'] = (
    (df_features['glucose_category'] * 3) +
    (df_features['bmi_category'] * 2) +
    (df_features['age_group'] * 1) +
    (df_features['bp_category'] * 1)
)

print("✓ Risk score created")
print(f"  Range: {df_features['risk_score'].min()} - {df_features['risk_score'].max()}")
print(f"  Mean: {df_features['risk_score'].mean():.2f}")

# ============================================================================
# 8. NORMALIZE NUMERICAL FEATURES
# ============================================================================
print("\n8. Normalizing Numerical Features...")

# Select numerical features to scale (excluding target and categories)
numerical_features = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                     'insulin', 'bmi', 'diabetes_pedigree', 'age',
                     'bmi_glucose_interaction', 'age_glucose_interaction',
                     'pregnancies_age_interaction']

# Create scaler
scaler = StandardScaler()

# Scale features
df_features[numerical_features] = scaler.fit_transform(df_features[numerical_features])

print(f"✓ Normalized {len(numerical_features)} numerical features")

# ============================================================================
# 9. FEATURE SUMMARY
# ============================================================================
print("\n9. Feature Summary...")

print(f"\nOriginal features: {len(df.columns)}")
print(f"Engineered features: {len(df_features.columns) - len(df.columns)}")
print(f"Total features: {len(df_features.columns)}")

print("\nNew features added:")
new_features = [col for col in df_features.columns if col not in df.columns]
for feat in new_features:
    print(f"  - {feat}")

# ============================================================================
# 10. SAVE FINAL DATASET
# ============================================================================
print("\n10. Saving Final Dataset...")

# Create processed directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Save final dataset
output_path = 'data/processed/diabetes_final.csv'
df_features.to_csv(output_path, index=False)

print(f"✓ Final dataset saved to: {output_path}")
print(f"  Shape: {df_features.shape}")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING SUMMARY - DIABETES")
print("="*70)
print(f"Original Features: {len(df.columns)}")
print(f"New Features: {len(df_features.columns) - len(df.columns)}")
print(f"Total Features: {len(df_features.columns)}")
print(f"Total Rows: {len(df_features)}")
print("="*70)

print("\n✓ Diabetes feature engineering complete!")
