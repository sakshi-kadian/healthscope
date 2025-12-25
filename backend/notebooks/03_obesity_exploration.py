"""
HealthScope - Exploratory Data Analysis
Dataset 3: Obesity Levels

Objective: Understand the obesity dataset structure, identify patterns, 
and document data quality issues.

Dataset: Obesity Levels Dataset from UCI ML Repository
- Rows: 2111
- Columns: 17 (16 features + 1 target)
- Target: NObeyesdad (Obesity level categories)
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
print("HEALTHSCOPE - EXPLORATORY DATA ANALYSIS: OBESITY DATASET")
print("="*70)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n1. Loading Dataset...")
df = pd.read_csv('data/raw/obesity.csv')

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

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

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

# ============================================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n5. Descriptive Statistics...")
print("\nNumerical Features Statistics:")
print(df[numerical_cols].describe())

print("\nCategorical Features Value Counts:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ============================================================================
# 6. DISTRIBUTION PLOTS - NUMERICAL FEATURES
# ============================================================================
print("\n6. Creating Distribution Plots...")

# Create reports/figures directory if it doesn't exist
os.makedirs('reports/figures', exist_ok=True)

# Plot distributions for numerical features
if len(numerical_cols) > 0:
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    fig.suptitle('Distribution of Numerical Features - Obesity', fontsize=16, y=1.00)
    
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(col)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    # Hide extra subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/figures/obesity_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Distribution plots saved to reports/figures/obesity_distributions.png")
    plt.close()

# Box plots for numerical features
if len(numerical_cols) > 0:
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    fig.suptitle('Box Plots - Outlier Detection (Obesity)', fontsize=16, y=1.00)
    
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].boxplot(df[col].dropna())
        axes[idx].set_title(col)
        axes[idx].set_ylabel('Value')
    
    # Hide extra subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/figures/obesity_boxplots.png', dpi=300, bbox_inches='tight')
    print("✓ Box plots saved to reports/figures/obesity_boxplots.png")
    plt.close()

# ============================================================================
# 7. CATEGORICAL FEATURES VISUALIZATION
# ============================================================================
print("\n7. Creating Categorical Features Visualization...")

# Plot categorical features
if len(categorical_cols) > 0:
    n_cat = len(categorical_cols)
    n_rows_cat = (n_cat + 1) // 2
    fig, axes = plt.subplots(n_rows_cat, 2, figsize=(15, n_rows_cat * 4))
    fig.suptitle('Categorical Features Distribution - Obesity', fontsize=16, y=1.00)
    
    if n_rows_cat == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    for idx, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts()
        axes[idx].bar(range(len(value_counts)), value_counts.values, edgecolor='black', alpha=0.7)
        axes[idx].set_title(col)
        axes[idx].set_xlabel('Category')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    # Hide extra subplot if odd number
    if n_cat % 2 != 0:
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/figures/obesity_categorical.png', dpi=300, bbox_inches='tight')
    print("✓ Categorical plots saved to reports/figures/obesity_categorical.png")
    plt.close()

# ============================================================================
# 8. CORRELATION ANALYSIS (NUMERICAL FEATURES ONLY)
# ============================================================================
print("\n8. Correlation Analysis...")

if len(numerical_cols) > 1:
    # Calculate correlation matrix for numerical features
    correlation_matrix = df[numerical_cols].corr()
    
    print("\nCorrelation Matrix (Numerical Features):")
    print(correlation_matrix)
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Correlation Matrix - Obesity Dataset (Numerical Features)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('reports/figures/obesity_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation heatmap saved to reports/figures/obesity_correlation.png")
    plt.close()

# ============================================================================
# 9. EXPLORATION SUMMARY
# ============================================================================
print("\n9. Creating Exploration Summary...")

# Get target column (assuming it's the last column or named NObeyesdad)
target_col = 'NObeyesdad' if 'NObeyesdad' in df.columns else df.columns[-1]

summary = {
    'Dataset': 'Obesity Levels',
    'Total Rows': len(df),
    'Total Columns': len(df.columns),
    'Numerical Features': len(numerical_cols),
    'Categorical Features': len(categorical_cols),
    'Missing Values': df.isnull().sum().sum(),
    'Duplicate Rows': df.duplicated().sum(),
    'Target Column': target_col,
    'Target Categories': df[target_col].nunique() if target_col in df.columns else 'N/A'
}

print("\n" + "="*70)
print("EXPLORATION SUMMARY - OBESITY")
print("="*70)
for key, value in summary.items():
    print(f"{key}: {value}")

if target_col in df.columns:
    print(f"\nTarget Distribution ({target_col}):")
    print(df[target_col].value_counts())

print("="*70)
