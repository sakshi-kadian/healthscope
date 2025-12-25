"""
HealthScope - Explainability Visualization Refinement
Objective: Create polished, publication-ready SHAP visualizations and
a comprehensive explainability dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for publication-quality plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

print("="*70)
print("HEALTHSCOPE - EXPLAINABILITY VISUALIZATION REFINEMENT")
print("="*70)

# Create directories
os.makedirs('reports/figures/shap/refined', exist_ok=True)

# ============================================================================
# 1. CREATE COMPREHENSIVE FEATURE IMPORTANCE DASHBOARD
# ============================================================================
print("\n1. Creating Comprehensive Feature Importance Dashboard...")

# Load feature importance data
diabetes_fi = pd.read_csv('reports/diabetes_feature_importance.csv')
heart_fi = pd.read_csv('reports/heart_feature_importance.csv')
obesity_fi = pd.read_csv('reports/obesity_feature_importance.csv')

# Create dashboard with 3 subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01']

# Diabetes - Top 10 features
ax1 = fig.add_subplot(gs[0, :])
top_diabetes = diabetes_fi.head(10)
bars1 = ax1.barh(range(len(top_diabetes)), top_diabetes['importance'], color=colors[0], alpha=0.8)
ax1.set_yticks(range(len(top_diabetes)))
ax1.set_yticklabels(top_diabetes['feature'])
ax1.invert_yaxis()
ax1.set_xlabel('Mean |SHAP Value| (Impact on Model Output)', fontweight='bold')
ax1.set_title('Diabetes Model - Top 10 Most Important Features', fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, top_diabetes['importance'])):
    ax1.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

# Heart Disease - Top 10 features
ax2 = fig.add_subplot(gs[1, :])
top_heart = heart_fi.head(10)
bars2 = ax2.barh(range(len(top_heart)), top_heart['importance'], color=colors[1], alpha=0.8)
ax2.set_yticks(range(len(top_heart)))
ax2.set_yticklabels(top_heart['feature'])
ax2.invert_yaxis()
ax2.set_xlabel('Mean |SHAP Value| (Impact on Model Output)', fontweight='bold')
ax2.set_title('Heart Disease Model - Top 10 Most Important Features', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, top_heart['importance'])):
    ax2.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=9)

# Obesity - Top 10 features
ax3 = fig.add_subplot(gs[2, :])
top_obesity = obesity_fi.head(10)
bars3 = ax3.barh(range(len(top_obesity)), top_obesity['importance'], color=colors[2], alpha=0.8)
ax3.set_yticks(range(len(top_obesity)))
ax3.set_yticklabels(top_obesity['feature'])
ax3.invert_yaxis()
ax3.set_xlabel('Mean |SHAP Value| (Impact on Model Output)', fontweight='bold')
ax3.set_title('Obesity Model - Top 10 Most Important Features', fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='x', alpha=0.3, linestyle='--')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add value labels
for i, (bar, val) in enumerate(zip(bars3, top_obesity['importance'])):
    ax3.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

# Add overall title
fig.suptitle('HealthScope - SHAP Feature Importance Dashboard\nAll Models', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('reports/figures/shap/refined/feature_importance_dashboard.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Feature importance dashboard created")

# ============================================================================
# 2. CREATE SIDE-BY-SIDE COMPARISON
# ============================================================================
print("\n2. Creating Side-by-Side Model Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Diabetes
top5_diabetes = diabetes_fi.head(5)
axes[0].barh(range(len(top5_diabetes)), top5_diabetes['importance'], 
             color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(top5_diabetes)))
axes[0].set_yticklabels(top5_diabetes['feature'], fontsize=11)
axes[0].invert_yaxis()
axes[0].set_xlabel('SHAP Importance', fontweight='bold', fontsize=12)
axes[0].set_title('Diabetes\nTop 5 Features', fontsize=13, fontweight='bold', pad=15)
axes[0].grid(axis='x', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Heart
top5_heart = heart_fi.head(5)
axes[1].barh(range(len(top5_heart)), top5_heart['importance'], 
             color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_yticks(range(len(top5_heart)))
axes[1].set_yticklabels(top5_heart['feature'], fontsize=11)
axes[1].invert_yaxis()
axes[1].set_xlabel('SHAP Importance', fontweight='bold', fontsize=12)
axes[1].set_title('Heart Disease\nTop 5 Features', fontsize=13, fontweight='bold', pad=15)
axes[1].grid(axis='x', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Obesity
top5_obesity = obesity_fi.head(5)
axes[2].barh(range(len(top5_obesity)), top5_obesity['importance'], 
             color=colors[2], alpha=0.8, edgecolor='black', linewidth=1.5)
axes[2].set_yticks(range(len(top5_obesity)))
axes[2].set_yticklabels(top5_obesity['feature'], fontsize=11)
axes[2].invert_yaxis()
axes[2].set_xlabel('SHAP Importance', fontweight='bold', fontsize=12)
axes[2].set_title('Obesity\nTop 5 Features', fontsize=13, fontweight='bold', pad=15)
axes[2].grid(axis='x', alpha=0.3, linestyle='--')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

fig.suptitle('HealthScope - Top 5 Most Important Features by Model', 
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('reports/figures/shap/refined/top5_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Side-by-side comparison created")

# ============================================================================
# 3. CREATE FEATURE IMPORTANCE SUMMARY TABLE
# ============================================================================
print("\n3. Creating Feature Importance Summary Table...")

# Create summary table
summary_data = []

for i in range(5):
    summary_data.append({
        'Rank': i + 1,
        'Diabetes': diabetes_fi.iloc[i]['feature'],
        'Diabetes_Importance': f"{diabetes_fi.iloc[i]['importance']:.4f}",
        'Heart': heart_fi.iloc[i]['feature'],
        'Heart_Importance': f"{heart_fi.iloc[i]['importance']:.4f}",
        'Obesity': obesity_fi.iloc[i]['feature'],
        'Obesity_Importance': f"{obesity_fi.iloc[i]['importance']:.4f}"
    })

summary_df = pd.DataFrame(summary_data)

# Create table visualization
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# Create table
table_data = []
table_data.append(['Rank', 'Diabetes Feature', 'Importance', 'Heart Feature', 'Importance', 'Obesity Feature', 'Importance'])

for _, row in summary_df.iterrows():
    table_data.append([
        row['Rank'],
        row['Diabetes'],
        row['Diabetes_Importance'],
        row['Heart'],
        row['Heart_Importance'],
        row['Obesity'],
        row['Obesity_Importance']
    ])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.08, 0.28, 0.12, 0.28, 0.12, 0.28, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(7):
    cell = table[(0, i)]
    cell.set_facecolor('#2E86AB')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(7):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#F0F0F0')
        else:
            cell.set_facecolor('white')

plt.title('HealthScope - Top 5 Feature Importance Summary\nAll Models', 
          fontsize=14, fontweight='bold', pad=20)

plt.savefig('reports/figures/shap/refined/feature_importance_table.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Feature importance table created")

# Save as CSV too
summary_df.to_csv('reports/feature_importance_summary.csv', index=False)
print("✓ Summary table saved to CSV")

# ============================================================================
# 4. CREATE EXPLAINABILITY SUMMARY INFOGRAPHIC
# ============================================================================
print("\n4. Creating Explainability Summary Infographic...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('HealthScope - Model Explainability Summary', 
             fontsize=18, fontweight='bold', y=0.98)

# Diabetes section
ax1 = fig.add_subplot(gs[0, :])
ax1.text(0.5, 0.9, 'DIABETES PREDICTION MODEL', ha='center', va='top',
         fontsize=14, fontweight='bold', color=colors[0],
         transform=ax1.transAxes)
ax1.text(0.5, 0.7, 'Key Insight: Glucose-BMI interaction is the strongest predictor', 
         ha='center', va='top', fontsize=11, transform=ax1.transAxes)
ax1.text(0.5, 0.5, f'Top Feature: {diabetes_fi.iloc[0]["feature"]}', 
         ha='center', va='top', fontsize=10, transform=ax1.transAxes)
ax1.text(0.5, 0.3, f'Importance: {diabetes_fi.iloc[0]["importance"]:.4f}', 
         ha='center', va='top', fontsize=10, transform=ax1.transAxes, style='italic')
ax1.axis('off')

# Heart section
ax2 = fig.add_subplot(gs[1, :])
ax2.text(0.5, 0.9, 'HEART DISEASE PREDICTION MODEL', ha='center', va='top',
         fontsize=14, fontweight='bold', color=colors[1],
         transform=ax2.transAxes)
ax2.text(0.5, 0.7, 'Key Insight: Chest pain type is the most critical indicator', 
         ha='center', va='top', fontsize=11, transform=ax2.transAxes)
ax2.text(0.5, 0.5, f'Top Feature: {heart_fi.iloc[0]["feature"]}', 
         ha='center', va='top', fontsize=10, transform=ax2.transAxes)
ax2.text(0.5, 0.3, f'Importance: {heart_fi.iloc[0]["importance"]:.4f}', 
         ha='center', va='top', fontsize=10, transform=ax2.transAxes, style='italic')
ax2.axis('off')

# Obesity section
ax3 = fig.add_subplot(gs[2, :])
ax3.text(0.5, 0.9, 'OBESITY LEVEL PREDICTION MODEL', ha='center', va='top',
         fontsize=14, fontweight='bold', color=colors[2],
         transform=ax3.transAxes)
ax3.text(0.5, 0.7, 'Key Insight: Calculated BMI dominates obesity classification', 
         ha='center', va='top', fontsize=11, transform=ax3.transAxes)
ax3.text(0.5, 0.5, f'Top Feature: {obesity_fi.iloc[0]["feature"]}', 
         ha='center', va='top', fontsize=10, transform=ax3.transAxes)
ax3.text(0.5, 0.3, f'Importance: {obesity_fi.iloc[0]["importance"]:.4f}', 
         ha='center', va='top', fontsize=10, transform=ax3.transAxes, style='italic')
ax3.axis('off')

plt.savefig('reports/figures/shap/refined/explainability_summary.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Explainability summary infographic created")

# ============================================================================
# 5. CREATE VISUALIZATION INDEX
# ============================================================================
print("\n5. Creating Visualization Index...")

index_doc = """
# HEALTHSCOPE - EXPLAINABILITY VISUALIZATION INDEX

## Overview
This document provides an index of all SHAP explainability visualizations created for the HealthScope project.

## Refined Visualizations (Publication-Ready)

### 1. Feature Importance Dashboard
**File**: `reports/figures/shap/refined/feature_importance_dashboard.png`
**Description**: Comprehensive dashboard showing top 10 features for all three models
**Use Case**: Project presentations, documentation, reports
**Key Insights**:
- Diabetes: Interaction features dominate
- Heart: Clinical symptoms are key
- Obesity: BMI-related features are primary

### 2. Top 5 Comparison
**File**: `reports/figures/shap/refined/top5_comparison.png`
**Description**: Side-by-side comparison of top 5 features across models
**Use Case**: Quick reference, executive summaries
**Format**: Horizontal bar charts, color-coded by model

### 3. Feature Importance Table
**File**: `reports/figures/shap/refined/feature_importance_table.png`
**Description**: Tabular summary of top 5 features with importance scores
**Use Case**: Reports, academic papers
**Also Available**: CSV format (`reports/feature_importance_summary.csv`)

### 4. Explainability Summary
**File**: `reports/figures/shap/refined/explainability_summary.png`
**Description**: High-level infographic summarizing key insights
**Use Case**: Presentations, posters, quick overviews

## Global Explanations

### Diabetes Model
- **Summary Plot**: `reports/figures/shap/diabetes_shap_summary.png`
- **Bar Plot**: `reports/figures/shap/diabetes_shap_bar.png`
- **Feature Rankings**: `reports/diabetes_feature_importance.csv`

### Heart Disease Model
- **Summary Plot**: `reports/figures/shap/heart_shap_summary.png`
- **Bar Plot**: `reports/figures/shap/heart_shap_bar.png`
- **Feature Rankings**: `reports/heart_feature_importance.csv`

### Obesity Model
- **Summary Plot**: `reports/figures/shap/obesity_shap_summary.png`
- **Bar Plot**: `reports/figures/shap/obesity_shap_bar.png`
- **Feature Rankings**: `reports/obesity_feature_importance.csv`

## Individual Explanations

### Diabetes Cases
- **High Risk**: `reports/figures/shap/individual/diabetes_waterfall_high_risk.png`
- **Low Risk**: `reports/figures/shap/individual/diabetes_waterfall_low_risk.png`

### Heart Disease Cases
- **High Risk**: `reports/figures/shap/individual/heart_waterfall_high_risk.png`
- **Low Risk**: `reports/figures/shap/individual/heart_waterfall_low_risk.png`

### Obesity Cases
- **Case 1**: `reports/figures/shap/individual/obesity_waterfall_case1.png`
- **Case 2**: `reports/figures/shap/individual/obesity_waterfall_case2.png`

## Comparison Visualizations

### All Models
- **Combined Importance**: `reports/figures/shap/all_models_feature_importance.png`

## Documentation

- **Explanation Guide**: `reports/SHAP_EXPLANATION_GUIDE.md`
- **Visualization Index**: `reports/VISUALIZATION_INDEX.md` (this file)

## Usage Guidelines

### For Presentations
1. Use refined visualizations (publication-ready)
2. Start with explainability summary for overview
3. Use dashboard for detailed discussion
4. Show individual cases for real-world examples

### For Reports
1. Include feature importance table
2. Reference CSV files for exact values
3. Use summary plots for technical details
4. Cite individual explanations as examples

### For Documentation
1. Link to visualization index
2. Include explanation guide
3. Provide context for each visualization
4. Explain interpretation methods

## Technical Details

- **Format**: PNG (300 DPI)
- **Color Palette**: Colorblind-friendly
- **Fonts**: System defaults, scalable
- **Size**: Optimized for both screen and print

## Key Findings

### Diabetes
- **Most Important**: BMI-Glucose interaction
- **Insight**: Combined metabolic factors matter most
- **Actionable**: Focus on both weight and glucose control

### Heart Disease
- **Most Important**: Chest pain type (cp)
- **Insight**: Symptoms are highly predictive
- **Actionable**: Symptom assessment is critical

### Obesity
- **Most Important**: Calculated BMI
- **Insight**: Direct body composition measures dominate
- **Actionable**: BMI is the primary indicator

---
**Last Updated**: December 2025
**Version**: 1.0
**Project**: HealthScope - Chronic Disease Risk Prediction
"""

with open('reports/VISUALIZATION_INDEX.md', 'w', encoding='utf-8') as f:
    f.write(index_doc)

print("✓ Visualization index created")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION REFINEMENT SUMMARY")
print("="*70)
print(f"Refined Visualizations Created: 4")
print(f"  1. Feature Importance Dashboard (comprehensive)")
print(f"  2. Top 5 Comparison (side-by-side)")
print(f"  3. Feature Importance Table (publication-ready)")
print(f"  4. Explainability Summary (infographic)")
print(f"\nDocumentation:")
print(f"  - Visualization Index (VISUALIZATION_INDEX.md)")
print(f"  - Feature Importance Summary (CSV)")
print(f"\nAll visualizations are:")
print(f"  ✓ Publication-ready (300 DPI)")
print(f"  ✓ Colorblind-friendly palette")
print(f"  ✓ Professional styling")
print(f"  ✓ Properly labeled and titled")
print("="*70)

print("\n✓ Explainability visualization refinement complete!")
