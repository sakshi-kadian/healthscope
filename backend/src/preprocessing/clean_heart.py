"""
HealthScope - Data Cleaning & Preprocessing
Dataset 2: Heart Disease (Cleveland)

Objective: Clean the heart disease dataset by handling missing values, outliers,
and standardizing features.
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("HEALTHSCOPE - DATA CLEANING: HEART DISEASE DATASET")
print("="*70)

# ============================================================================
# 1. LOAD RAW DATASET
# ============================================================================
print("\n1. Loading Raw Dataset...")
df = pd.read_csv('data/raw/heart.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Original Shape: {df.shape}")
print(f"  Original Columns: {df.columns.tolist()}")

# Create a copy for cleaning
df_clean = df.copy()

# ============================================================================
# 2. HANDLE MISSING VALUES
# ============================================================================
print("\n2. Handling Missing Values...")

# Check for missing values
missing_count = df_clean.isnull().sum().sum()
print(f"Missing values found: {missing_count}")

if missing_count > 0:
    print("\nImputing missing values with median...")
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                median_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_value)
                print(f"  {col}: Imputed with median = {median_value:.2f}")
else:
    print("✓ No missing values found!")

# ============================================================================
# 3. HANDLE OUTLIERS
# ============================================================================
print("\n3. Handling Outliers...")

# Define clinical thresholds for heart disease features
clinical_thresholds = {
    'age': (18, 100),
    'trestbps': (80, 220),  # Resting blood pressure (mm Hg)
    'chol': (100, 600),  # Cholesterol (mg/dl)
    'thalach': (60, 220),  # Max heart rate
    'oldpeak': (0, 7)  # ST depression
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

if outliers_removed == 0:
    print("  ✓ No outliers found within clinical thresholds")
else:
    print(f"\n✓ Total outliers capped: {outliers_removed}")

# ============================================================================
# 4. HANDLE DUPLICATES
# ============================================================================
print("\n4. Handling Duplicate Rows...")

duplicates_before = df_clean.duplicated().sum()
print(f"Duplicate rows found: {duplicates_before}")

if duplicates_before > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"✓ Removed {duplicates_before} duplicate rows")
    print(f"  New shape: {df_clean.shape}")
else:
    print("✓ No duplicate rows found!")

# ============================================================================
# 5. STANDARDIZE FEATURE NAMES
# ============================================================================
print("\n5. Standardizing Feature Names...")

# Keep names consistent (already in lowercase)
print("✓ Column names already standardized:")
print(f"  {df_clean.columns.tolist()}")

# ============================================================================
# 6. DATA VALIDATION
# ============================================================================
print("\n6. Data Validation...")

# Check for any remaining issues
validation_checks = {
    'Missing Values': df_clean.isnull().sum().sum(),
    'Duplicate Rows': df_clean.duplicated().sum(),
    'Negative Values (excluding oldpeak)': (df_clean.drop(columns=['oldpeak'], errors='ignore') < 0).sum().sum(),
    'Total Rows': len(df_clean),
    'Total Columns': len(df_clean.columns)
}

print("\nValidation Results:")
for check, value in validation_checks.items():
    status = "✓" if value == 0 or check in ['Total Rows', 'Total Columns'] else "⚠"
    print(f"  {status} {check}: {value}")

# ============================================================================
# 7. SAVE CLEANED DATASET
# ============================================================================
print("\n7. Saving Cleaned Dataset...")

# Create processed directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Save cleaned dataset
output_path = 'data/processed/heart_clean.csv'
df_clean.to_csv(output_path, index=False)

print(f"✓ Cleaned dataset saved to: {output_path}")
print(f"  Shape: {df_clean.shape}")

# ============================================================================
# 8. CLEANING SUMMARY
# ============================================================================
print("\n8. Cleaning Summary...")

summary = {
    'Original Rows': len(df),
    'Cleaned Rows': len(df_clean),
    'Rows Removed': len(df) - len(df_clean),
    'Original Columns': len(df.columns),
    'Cleaned Columns': len(df_clean.columns),
    'Missing Values Imputed': missing_count,
    'Outliers Capped': outliers_removed,
    'Duplicates Removed': duplicates_before
}

print("\n" + "="*70)
print("CLEANING SUMMARY - HEART DISEASE")
print("="*70)
for key, value in summary.items():
    print(f"{key}: {value}")
print("="*70)

print("\n✓ Heart disease dataset cleaning complete!")
