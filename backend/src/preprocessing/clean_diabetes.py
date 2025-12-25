"""
HealthScope - Data Cleaning & Preprocessing
Dataset 1: Diabetes (Pima Indians)

Objective: Clean the diabetes dataset by handling missing values, outliers,
and standardizing features.

Issues identified in EDA:
- Zero values in medical features (likely missing data)
- Potential outliers in some features
- Need to standardize feature names
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("HEALTHSCOPE - DATA CLEANING: DIABETES DATASET")
print("="*70)

# ============================================================================
# 1. LOAD RAW DATASET
# ============================================================================
print("\n1. Loading Raw Dataset...")
df = pd.read_csv('data/raw/diabetes.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Original Shape: {df.shape}")
print(f"  Original Columns: {df.columns.tolist()}")

# Create a copy for cleaning
df_clean = df.copy()

# ============================================================================
# 2. HANDLE MISSING VALUES (ZEROS IN MEDICAL FEATURES)
# ============================================================================
print("\n2. Handling Missing Values...")

# Medical features that cannot be zero
zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print(f"\nFeatures where zero is invalid: {zero_not_allowed}")

# Count zeros before cleaning
print("\nZero values before cleaning:")
for col in zero_not_allowed:
    zero_count = (df_clean[col] == 0).sum()
    zero_pct = (zero_count / len(df_clean)) * 100
    print(f"  {col}: {zero_count} ({zero_pct:.2f}%)")

# Replace zeros with NaN for these features
for col in zero_not_allowed:
    df_clean[col] = df_clean[col].replace(0, np.nan)

# Impute missing values with median (more robust to outliers)
print("\nImputing missing values with median...")
for col in zero_not_allowed:
    if df_clean[col].isnull().sum() > 0:
        median_value = df_clean[col].median()
        df_clean[col].fillna(median_value, inplace=True)
        print(f"  {col}: Imputed {(df[col] == 0).sum()} values with median = {median_value:.2f}")

# Verify no missing values remain
print(f"\n✓ Missing values after imputation: {df_clean.isnull().sum().sum()}")

# ============================================================================
# 3. HANDLE OUTLIERS
# ============================================================================
print("\n3. Handling Outliers...")

# Define clinical thresholds for outlier detection
clinical_thresholds = {
    'Glucose': (0, 300),  # mg/dL
    'BloodPressure': (40, 200),  # mm Hg
    'SkinThickness': (0, 100),  # mm
    'Insulin': (0, 850),  # mu U/ml
    'BMI': (10, 70),  # kg/m^2
    'Age': (18, 100)  # years
}

print("\nApplying clinical thresholds:")
outliers_removed = 0

for col, (min_val, max_val) in clinical_thresholds.items():
    if col in df_clean.columns:
        # Count outliers
        outliers = ((df_clean[col] < min_val) | (df_clean[col] > max_val)).sum()
        
        if outliers > 0:
            print(f"  {col}: {outliers} outliers found (range: {min_val}-{max_val})")
            
            # Cap outliers at threshold values
            df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
            outliers_removed += outliers

print(f"\n✓ Total outliers capped: {outliers_removed}")

# ============================================================================
# 4. STANDARDIZE FEATURE NAMES
# ============================================================================
print("\n4. Standardizing Feature Names...")

# Rename columns to lowercase with underscores
column_mapping = {
    'Pregnancies': 'pregnancies',
    'Glucose': 'glucose',
    'BloodPressure': 'blood_pressure',
    'SkinThickness': 'skin_thickness',
    'Insulin': 'insulin',
    'BMI': 'bmi',
    'DiabetesPedigreeFunction': 'diabetes_pedigree',
    'Age': 'age',
    'Outcome': 'outcome'
}

df_clean.rename(columns=column_mapping, inplace=True)

print("✓ Column names standardized:")
print(f"  {df_clean.columns.tolist()}")

# ============================================================================
# 5. DATA VALIDATION
# ============================================================================
print("\n5. Data Validation...")

# Check for any remaining issues
validation_checks = {
    'Missing Values': df_clean.isnull().sum().sum(),
    'Duplicate Rows': df_clean.duplicated().sum(),
    'Negative Values': (df_clean < 0).sum().sum(),
    'Total Rows': len(df_clean),
    'Total Columns': len(df_clean.columns)
}

print("\nValidation Results:")
for check, value in validation_checks.items():
    status = "✓" if value == 0 or check in ['Total Rows', 'Total Columns'] else "⚠"
    print(f"  {status} {check}: {value}")

# ============================================================================
# 6. SAVE CLEANED DATASET
# ============================================================================
print("\n6. Saving Cleaned Dataset...")

# Create processed directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Save cleaned dataset
output_path = 'data/processed/diabetes_clean.csv'
df_clean.to_csv(output_path, index=False)

print(f"✓ Cleaned dataset saved to: {output_path}")
print(f"  Shape: {df_clean.shape}")

# ============================================================================
# 7. CLEANING SUMMARY
# ============================================================================
print("\n7. Cleaning Summary...")

summary = {
    'Original Rows': len(df),
    'Cleaned Rows': len(df_clean),
    'Rows Removed': len(df) - len(df_clean),
    'Original Columns': len(df.columns),
    'Cleaned Columns': len(df_clean.columns),
    'Missing Values Imputed': (df[zero_not_allowed] == 0).sum().sum(),
    'Outliers Capped': outliers_removed,
    'Duplicate Rows': df_clean.duplicated().sum()
}

print("\n" + "="*70)
print("CLEANING SUMMARY - DIABETES")
print("="*70)
for key, value in summary.items():
    print(f"{key}: {value}")
print("="*70)

print("\n✓ Diabetes dataset cleaning complete!")
