"""
HealthScope - Exploratory Data Analysis
Dataset 1: Diabetes (Pima Indians)

Objective: Understand the diabetes dataset structure, identify patterns, 
and document data quality issues.

Dataset: Pima Indians Diabetes Dataset from UCI ML Repository
- Rows: 768
- Columns: 9 (8 features + 1 target)
- Target: Outcome (0 = No diabetes, 1 = Diabetes)
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
print("HEALTHSCOPE - EXPLORATORY DATA ANALYSIS: DIABETES DATASET")
print("="*70)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n1. Loading Dataset...")
df = pd.read_csv('data/raw/diabetes.csv')

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

# Check for zero values (potential missing data in medical datasets)
print("\nZero Values (potential missing data):")
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

print("\nTarget Variable (Outcome) Distribution:")
print(df['Outcome'].value_counts())
print("\nPercentage:")
print(df['Outcome'].value_counts(normalize=True) * 100)

# ============================================================================
# 6. DISTRIBUTION PLOTS
# ============================================================================
print("\n6. Creating Distribution Plots...")

# Create reports/figures directory if it doesn't exist
os.makedirs('reports/figures', exist_ok=True)

# Plot distributions for all features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Distribution of Features', fontsize=16, y=1.00)

for idx, col in enumerate(df.columns):
    row = idx // 3
    col_idx = idx % 3
    
    axes[row, col_idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[row, col_idx].set_title(col)
    axes[row, col_idx].set_xlabel('Value')
    axes[row, col_idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('reports/figures/diabetes_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Distribution plots saved to reports/figures/diabetes_distributions.png")
plt.close()

# Box plots to identify outliers
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.00)

for idx, col in enumerate(df.columns):
    row = idx // 3
    col_idx = idx % 3
    
    axes[row, col_idx].boxplot(df[col].dropna())
    axes[row, col_idx].set_title(col)
    axes[row, col_idx].set_ylabel('Value')

plt.tight_layout()
plt.savefig('reports/figures/diabetes_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Box plots saved to reports/figures/diabetes_boxplots.png")
plt.close()

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================
print("\n7. Correlation Analysis...")

# Calculate correlation matrix
correlation_matrix = df.corr()

# Display correlation with target
print("\nCorrelation with Target (Outcome):")
target_corr = correlation_matrix['Outcome'].sort_values(ascending=False)
print(target_corr)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Diabetes Dataset', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('reports/figures/diabetes_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved to reports/figures/diabetes_correlation.png")
plt.close()

# ============================================================================
# 8. FEATURE DISTRIBUTIONS BY TARGET
# ============================================================================
print("\n8. Creating Feature Distributions by Target...")

# Compare feature distributions for diabetes vs no diabetes
features = [col for col in df.columns if col != 'Outcome']

fig, axes = plt.subplots(4, 2, figsize=(15, 16))
fig.suptitle('Feature Distributions by Outcome', fontsize=16, y=1.00)

for idx, feature in enumerate(features):
    row = idx // 2
    col = idx % 2
    
    # Plot for no diabetes (0)
    axes[row, col].hist(df[df['Outcome'] == 0][feature], bins=20, alpha=0.5, 
                        label='No Diabetes', color='green', edgecolor='black')
    # Plot for diabetes (1)
    axes[row, col].hist(df[df['Outcome'] == 1][feature], bins=20, alpha=0.5, 
                        label='Diabetes', color='red', edgecolor='black')
    
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].legend()

plt.tight_layout()
plt.savefig('reports/figures/diabetes_by_outcome.png', dpi=300, bbox_inches='tight')
print("✓ Outcome comparison plots saved to reports/figures/diabetes_by_outcome.png")
plt.close()

# ============================================================================
# 9. SUMMARY STATISTICS BY OUTCOME
# ============================================================================
print("\n9. Summary Statistics by Outcome...")

print("\nNo Diabetes (Outcome = 0):")
print(df[df['Outcome'] == 0].describe())

print("\nDiabetes (Outcome = 1):")
print(df[df['Outcome'] == 1].describe())

# ============================================================================
# 10. EXPLORATION SUMMARY
# ============================================================================
print("\n10. Creating Exploration Summary...")

summary = {
    'Dataset': 'Diabetes (Pima Indians)',
    'Total Rows': len(df),
    'Total Columns': len(df.columns),
    'Missing Values': df.isnull().sum().sum(),
    'Duplicate Rows': df.duplicated().sum(),
    'Target Distribution': df['Outcome'].value_counts().to_dict(),
    'Top Correlated Features': target_corr.head(4).to_dict()
}

print("\n" + "="*70)
print("EXPLORATION SUMMARY")
print("="*70)
for key, value in summary.items():
    print(f"{key}: {value}")
print("="*70)
