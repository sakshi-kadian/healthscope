"""
HealthScope - Feature Engineering
Dataset 2: Heart Disease

Objective: Create new features from cleaned heart disease dataset to improve
model performance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("="*70)
print("HEALTHSCOPE - FEATURE ENGINEERING: HEART DISEASE DATASET")
print("="*70)

# ============================================================================
# 1. LOAD CLEANED DATASET
# ============================================================================
print("\n1. Loading Cleaned Dataset...")
df = pd.read_csv('data/processed/heart_clean.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# Create a copy for feature engineering
df_features = df.copy()

# ============================================================================
# 2. CREATE AGE GROUPS
# ============================================================================
print("\n2. Creating Age Groups...")

def categorize_age(age):
    if age < 40:
        return 0  # Young
    elif age < 55:
        return 1  # Middle-aged
    elif age < 70:
        return 2  # Senior
    else:
        return 3  # Elderly

df_features['age_group'] = df_features['age'].apply(categorize_age)

print("✓ Age groups created:")
print(df_features['age_group'].value_counts().sort_index())

# ============================================================================
# 3. CREATE CHOLESTEROL CATEGORIES
# ============================================================================
print("\n3. Creating Cholesterol Categories...")

def categorize_chol(chol):
    if chol < 200:
        return 0  # Desirable
    elif chol < 240:
        return 1  # Borderline high
    else:
        return 2  # High

df_features['chol_category'] = df_features['chol'].apply(categorize_chol)

print("✓ Cholesterol categories created:")
print(df_features['chol_category'].value_counts().sort_index())

# ============================================================================
# 4. CREATE BLOOD PRESSURE CATEGORIES
# ============================================================================
print("\n4. Creating Blood Pressure Categories...")

def categorize_bp(bp):
    if bp < 120:
        return 0  # Normal
    elif bp < 140:
        return 1  # Elevated
    else:
        return 2  # High

df_features['bp_category'] = df_features['trestbps'].apply(categorize_bp)

print("✓ Blood pressure categories created:")
print(df_features['bp_category'].value_counts().sort_index())

# ============================================================================
# 5. CREATE HEART RATE CATEGORIES
# ============================================================================
print("\n5. Creating Heart Rate Categories...")

def categorize_hr(hr):
    if hr < 100:
        return 0  # Low
    elif hr < 140:
        return 1  # Normal
    else:
        return 2  # High

df_features['hr_category'] = df_features['thalach'].apply(categorize_hr)

print("✓ Heart rate categories created:")
print(df_features['hr_category'].value_counts().sort_index())

# ============================================================================
# 6. CREATE INTERACTION FEATURES
# ============================================================================
print("\n6. Creating Interaction Features...")

# Age * Cholesterol interaction
df_features['age_chol_interaction'] = df_features['age'] * df_features['chol']

# Age * Max heart rate interaction
df_features['age_hr_interaction'] = df_features['age'] * df_features['thalach']

# Chest pain * ST depression interaction
df_features['cp_oldpeak_interaction'] = df_features['cp'] * df_features['oldpeak']

print("✓ Interaction features created:")
print(f"  - age_chol_interaction")
print(f"  - age_hr_interaction")
print(f"  - cp_oldpeak_interaction")

# ============================================================================
# 7. CREATE RISK SCORE
# ============================================================================
print("\n7. Creating Risk Score...")

# Risk score based on key factors
df_features['risk_score'] = (
    (df_features['age_group'] * 2) +
    (df_features['chol_category'] * 2) +
    (df_features['bp_category'] * 1) +
    (df_features['cp'] * 1) +
    (df_features['exang'] * 2)  # Exercise induced angina
)

print("✓ Risk score created")
print(f"  Range: {df_features['risk_score'].min()} - {df_features['risk_score'].max()}")
print(f"  Mean: {df_features['risk_score'].mean():.2f}")

# ============================================================================
# 8. NORMALIZE NUMERICAL FEATURES
# ============================================================================
print("\n8. Normalizing Numerical Features...")

# Select numerical features to scale
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                     'age_chol_interaction', 'age_hr_interaction',
                     'cp_oldpeak_interaction']

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

# Save final dataset
output_path = 'data/processed/heart_final.csv'
df_features.to_csv(output_path, index=False)

print(f"✓ Final dataset saved to: {output_path}")
print(f"  Shape: {df_features.shape}")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING SUMMARY - HEART DISEASE")
print("="*70)
print(f"Original Features: {len(df.columns)}")
print(f"New Features: {len(df_features.columns) - len(df.columns)}")
print(f"Total Features: {len(df_features.columns)}")
print(f"Total Rows: {len(df_features)}")
print("="*70)

print("\n✓ Heart disease feature engineering complete!")
