"""
HealthScope - Feature Engineering
Dataset 3: Obesity

Objective: Create new features from cleaned obesity dataset and encode
categorical variables.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

print("="*70)
print("HEALTHSCOPE - FEATURE ENGINEERING: OBESITY DATASET")
print("="*70)

# ============================================================================
# 1. LOAD CLEANED DATASET
# ============================================================================
print("\n1. Loading Cleaned Dataset...")
df = pd.read_csv('data/processed/obesity_clean.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# Create a copy for feature engineering
df_features = df.copy()

# ============================================================================
# 2. CALCULATE BMI (if not present)
# ============================================================================
print("\n2. Calculating BMI...")

# BMI = weight (kg) / height^2 (m^2)
df_features['calculated_bmi'] = df_features['weight'] / (df_features['height'] ** 2)

print(f"✓ BMI calculated")
print(f"  Range: {df_features['calculated_bmi'].min():.2f} - {df_features['calculated_bmi'].max():.2f}")
print(f"  Mean: {df_features['calculated_bmi'].mean():.2f}")

# ============================================================================
# 3. CREATE BMI CATEGORIES
# ============================================================================
print("\n3. Creating BMI Categories...")

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif bmi < 25:
        return 1  # Normal
    elif bmi < 30:
        return 2  # Overweight
    else:
        return 3  # Obese

df_features['bmi_category'] = df_features['calculated_bmi'].apply(categorize_bmi)

print("✓ BMI categories created:")
print(df_features['bmi_category'].value_counts().sort_index())

# ============================================================================
# 4. CREATE AGE GROUPS
# ============================================================================
print("\n4. Creating Age Groups...")

def categorize_age(age):
    if age < 20:
        return 0  # Teen
    elif age < 30:
        return 1  # Young adult
    elif age < 40:
        return 2  # Adult
    else:
        return 3  # Middle-aged+

df_features['age_group'] = df_features['age'].apply(categorize_age)

print("✓ Age groups created:")
print(df_features['age_group'].value_counts().sort_index())

# ============================================================================
# 5. CREATE LIFESTYLE SCORE
# ============================================================================
print("\n5. Creating Lifestyle Score...")

# Healthy lifestyle score (higher = healthier)
df_features['lifestyle_score'] = (
    df_features['fcvc'] +  # Vegetable consumption
    df_features['faf'] +   # Physical activity
    df_features['ch2o'] -  # Water consumption
    df_features['tue']     # Technology use (negative impact)
)

print("✓ Lifestyle score created")
print(f"  Range: {df_features['lifestyle_score'].min():.2f} - {df_features['lifestyle_score'].max():.2f}")
print(f"  Mean: {df_features['lifestyle_score'].mean():.2f}")

# ============================================================================
# 6. CREATE INTERACTION FEATURES
# ============================================================================
print("\n6. Creating Interaction Features...")

# BMI * Physical activity interaction
df_features['bmi_activity_interaction'] = df_features['calculated_bmi'] * df_features['faf']

# Age * BMI interaction
df_features['age_bmi_interaction'] = df_features['age'] * df_features['calculated_bmi']

# Vegetable * Water interaction (healthy habits)
df_features['fcvc_ch2o_interaction'] = df_features['fcvc'] * df_features['ch2o']

print("✓ Interaction features created:")
print(f"  - bmi_activity_interaction")
print(f"  - age_bmi_interaction")
print(f"  - fcvc_ch2o_interaction")

# ============================================================================
# 7. ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n7. Encoding Categorical Variables...")

# Identify categorical columns
categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()

# Remove target column from encoding
if 'nobeyesdad' in categorical_cols:
    categorical_cols.remove('nobeyesdad')

print(f"\nCategorical columns to encode: {categorical_cols}")

# Label encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_features[col + '_encoded'] = le.fit_transform(df_features[col])
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}: {len(le.classes_)} categories")

# Drop original categorical columns (keep only encoded versions)
print(f"\nDropping original categorical columns...")
df_features = df_features.drop(columns=categorical_cols)
print(f"✓ Dropped {len(categorical_cols)} original categorical columns")

# ============================================================================
# 8. NORMALIZE NUMERICAL FEATURES
# ============================================================================
print("\n8. Normalizing Numerical Features...")

# Select numerical features to scale
numerical_features = ['age', 'height', 'weight', 'fcvc', 'ncp', 'ch2o', 'faf', 'tue',
                     'calculated_bmi', 'lifestyle_score',
                     'bmi_activity_interaction', 'age_bmi_interaction',
                     'fcvc_ch2o_interaction']

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
output_path = 'data/processed/obesity_final.csv'
df_features.to_csv(output_path, index=False)

print(f"✓ Final dataset saved to: {output_path}")
print(f"  Shape: {df_features.shape}")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING SUMMARY - OBESITY")
print("="*70)
print(f"Original Features: {len(df.columns)}")
print(f"New Features: {len(df_features.columns) - len(df.columns)}")
print(f"Total Features: {len(df_features.columns)}")
print(f"Total Rows: {len(df_features)}")
print(f"Categorical Features Encoded: {len(categorical_cols)}")
print("="*70)

print("\n✓ Obesity feature engineering complete!")
