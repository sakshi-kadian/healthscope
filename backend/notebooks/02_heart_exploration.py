"""
HealthScope - Exploratory Data Analysis
Dataset 2: Heart Disease (Cleveland)

Objective: Understand the heart disease dataset structure, identify patterns, 
and document data quality issues.

Dataset: Cleveland Heart Disease Dataset from UCI ML Repository
- Rows: 1025
- Columns: 14 (13 features + 1 target)
- Target: target (0 = No disease, 1 = Disease)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("="*70)
print("HEALTHSCOPE - EXPLORATORY DATA ANALYSIS: HEART DISEASE DATASET")
print("="*70)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n1. Loading Dataset...")
df = pd.read_csv('data/raw/heart.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ============================================================================
# 2. INITIAL DATA INSPECTION
# ============================================================================
print("\n2. Initial Data Inspection...")
print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nRandom 10 rows:")
print(df.sample(10))

# ============================================================================
# 3. DATA TYPES AND INFO
# ============================================================================
print("\n3. Data Types and Info...")
print("\nDataset Info:")
df.info()

print("\nColumn Names:")
print(df.columns.tolist())

# ============================================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================================
print("\n4. Missing Values Analysis...")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})

print("\nMissing Values:")
print(missing_df[missing_df['Missing Count'] > 0])

if missing.sum() == 0:
    print("\n✓ No missing values found!")
else:
    print(f"\n⚠ Total missing values: {missing.sum()}")

# Check for zero values
print("\nZero Values:")
zero_counts = (df == 0).sum()
zero_pct = ((df == 0).sum() / len(df)) * 100

zero_df = pd.DataFrame({
    'Zero Count': zero_counts,
    'Percentage': zero_pct
})

print(zero_df[zero_df['Zero Count'] > 0])

# ============================================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n5. Descriptive Statistics...")
print("\nDescriptive Statistics:")
print(df.describe())

print("\nTarget Variable (target) Distribution:")
print(df['target'].value_counts())
print("\nPercentage:")
print(df['target'].value_counts(normalize=True) * 100)

# ============================================================================
# 6. DISTRIBUTION PLOTS
# ============================================================================
print("\n6. Creating Distribution Plots...")

# Create reports/figures directory if it doesn't exist
os.makedirs('reports/figures', exist_ok=True)

# Plot distributions for all features
n_cols = len(df.columns)
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
fig.suptitle('Distribution of Features - Heart Disease', fontsize=16, y=1.00)
axes = axes.flatten()

for idx, col in enumerate(df.columns):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(col)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')

# Hide extra subplots
for idx in range(len(df.columns), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('reports/figures/heart_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Distribution plots saved to reports/figures/heart_distributions.png")
plt.close()

# Box plots to identify outliers
fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
fig.suptitle('Box Plots - Outlier Detection (Heart Disease)', fontsize=16, y=1.00)
axes = axes.flatten()

for idx, col in enumerate(df.columns):
    axes[idx].boxplot(df[col].dropna())
    axes[idx].set_title(col)
    axes[idx].set_ylabel('Value')

# Hide extra subplots
for idx in range(len(df.columns), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('reports/figures/heart_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Box plots saved to reports/figures/heart_boxplots.png")
plt.close()

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================
print("\n7. Correlation Analysis...")

# Calculate correlation matrix
correlation_matrix = df.corr()

# Display correlation with target
print("\nCorrelation with Target (target):")
target_corr = correlation_matrix['target'].sort_values(ascending=False)
print(target_corr)

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Correlation Matrix - Heart Disease Dataset', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('reports/figures/heart_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved to reports/figures/heart_correlation.png")
plt.close()

# ============================================================================
# 8. FEATURE DISTRIBUTIONS BY TARGET
# ============================================================================
print("\n8. Creating Feature Distributions by Target...")

# Compare feature distributions for disease vs no disease
features = [col for col in df.columns if col != 'target']

n_features = len(features)
n_rows = (n_features + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 4))
fig.suptitle('Feature Distributions by Target - Heart Disease', fontsize=16, y=1.00)
axes = axes.flatten()

for idx, feature in enumerate(features):
    # Plot for no disease (0)
    axes[idx].hist(df[df['target'] == 0][feature], bins=20, alpha=0.5, 
                   label='No Disease', color='green', edgecolor='black')
    # Plot for disease (1)
    axes[idx].hist(df[df['target'] == 1][feature], bins=20, alpha=0.5, 
                   label='Disease', color='red', edgecolor='black')
    
    axes[idx].set_title(feature)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

# Hide extra subplot if odd number of features
if n_features % 2 != 0:
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig('reports/figures/heart_by_target.png', dpi=300, bbox_inches='tight')
print("✓ Target comparison plots saved to reports/figures/heart_by_target.png")
plt.close()

# ============================================================================
# 9. SUMMARY STATISTICS BY TARGET
# ============================================================================
print("\n9. Summary Statistics by Target...")

print("\nNo Disease (target = 0):")
print(df[df['target'] == 0].describe())

print("\nDisease (target = 1):")
print(df[df['target'] == 1].describe())

# ============================================================================
# 10. EXPLORATION SUMMARY
# ============================================================================
print("\n10. Creating Exploration Summary...")

summary = {
    'Dataset': 'Heart Disease (Cleveland)',
    'Total Rows': len(df),
    'Total Columns': len(df.columns),
    'Missing Values': df.isnull().sum().sum(),
    'Duplicate Rows': df.duplicated().sum(),
    'Target Distribution': df['target'].value_counts().to_dict(),
    'Top Correlated Features': target_corr.head(4).to_dict()
}

print("\n" + "="*70)
print("EXPLORATION SUMMARY - HEART DISEASE")
print("="*70)
for key, value in summary.items():
    print(f"{key}: {value}")
print("="*70)
