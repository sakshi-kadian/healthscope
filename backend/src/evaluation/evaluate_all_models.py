"""
HealthScope - Model Evaluation & Comparison
Objective: Evaluate and compare all trained models with comprehensive metrics
and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import joblib
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*70)
print("HEALTHSCOPE - MODEL EVALUATION & COMPARISON")
print("="*70)

# Create reports directory
os.makedirs('reports/figures', exist_ok=True)

# ============================================================================
# 1. LOAD ALL RESULTS
# ============================================================================
print("\n1. Loading All Model Results...")

# Load results
diabetes_results = pd.read_csv('models_saved/diabetes_all_models_results.csv')
heart_results = pd.read_csv('models_saved/heart_all_models_results.csv')
obesity_results = pd.read_csv('models_saved/obesity_all_models_results.csv')

# Add dataset column
diabetes_results['dataset'] = 'Diabetes'
heart_results['dataset'] = 'Heart Disease'
obesity_results['dataset'] = 'Obesity'

# Combine all results
all_results = pd.concat([diabetes_results, heart_results, obesity_results], ignore_index=True)

print(f"✓ Loaded results for {len(all_results)} models")
print(f"  - Diabetes: {len(diabetes_results)} models")
print(f"  - Heart Disease: {len(heart_results)} models")
print(f"  - Obesity: {len(obesity_results)} models")

# ============================================================================
# 2. SUMMARY TABLE
# ============================================================================
print("\n2. Creating Summary Table...")

print("\n" + "="*90)
print("MODEL PERFORMANCE SUMMARY - ALL DATASETS")
print("="*90)
print(f"{'Dataset':<20} {'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
print("-"*90)

for _, row in all_results.iterrows():
    print(f"{row['dataset']:<20} {row['model']:<25} {row['test_accuracy']:<12.4f} {row['test_precision']:<12.4f} {row['test_recall']:<12.4f} {row['test_f1']:<12.4f} {row['test_roc_auc']:<12.4f}")

print("="*90)

# ============================================================================
# 3. BEST MODELS SUMMARY
# ============================================================================
print("\n3. Best Models by Dataset...")

best_models = all_results.loc[all_results.groupby('dataset')['test_roc_auc'].idxmax()]

print("\n" + "="*70)
print("BEST MODELS (by ROC-AUC)")
print("="*70)
for _, row in best_models.iterrows():
    print(f"{row['dataset']:<20} {row['model']:<25} ROC-AUC: {row['test_roc_auc']:.4f}")
print("="*70)

# ============================================================================
# 4. COMPARISON VISUALIZATIONS
# ============================================================================
print("\n4. Creating Comparison Visualizations...")

# 4.1 ROC-AUC Comparison
fig, ax = plt.subplots(figsize=(14, 6))
datasets = all_results['dataset'].unique()
x = np.arange(len(datasets))
width = 0.25

for i, model_type in enumerate(['Logistic Regression', 'Random Forest', 'XGBoost']):
    model_data = all_results[all_results['model'] == model_type]
    roc_scores = [model_data[model_data['dataset'] == d]['test_roc_auc'].values[0] 
                  if len(model_data[model_data['dataset'] == d]) > 0 else 0 
                  for d in datasets]
    ax.bar(x + i*width, roc_scores, width, label=model_type)

ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison - ROC-AUC Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.8, color='r', linestyle='--', label='Target (0.80)', alpha=0.5)

plt.tight_layout()
plt.savefig('reports/figures/model_comparison_roc_auc.png', dpi=300, bbox_inches='tight')
print("✓ ROC-AUC comparison saved")
plt.close()

# 4.2 Accuracy Comparison
fig, ax = plt.subplots(figsize=(14, 6))

for i, model_type in enumerate(['Logistic Regression', 'Random Forest', 'XGBoost']):
    model_data = all_results[all_results['model'] == model_type]
    acc_scores = [model_data[model_data['dataset'] == d]['test_accuracy'].values[0] 
                  if len(model_data[model_data['dataset'] == d]) > 0 else 0 
                  for d in datasets]
    ax.bar(x + i*width, acc_scores, width, label=model_type)

ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison - Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/model_comparison_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Accuracy comparison saved")
plt.close()

# 4.3 F1-Score Comparison
fig, ax = plt.subplots(figsize=(14, 6))

for i, model_type in enumerate(['Logistic Regression', 'Random Forest', 'XGBoost']):
    model_data = all_results[all_results['model'] == model_type]
    f1_scores = [model_data[model_data['dataset'] == d]['test_f1'].values[0] 
                 if len(model_data[model_data['dataset'] == d]) > 0 else 0 
                 for d in datasets]
    ax.bar(x + i*width, f1_scores, width, label=model_type)

ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison - F1-Score', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/model_comparison_f1.png', dpi=300, bbox_inches='tight')
print("✓ F1-Score comparison saved")
plt.close()

# ============================================================================
# 5. HEATMAP OF ALL METRICS
# ============================================================================
print("\n5. Creating Performance Heatmap...")

# Create pivot table for heatmap
metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, dataset in enumerate(datasets):
    dataset_data = all_results[all_results['dataset'] == dataset][['model'] + metrics]
    dataset_data = dataset_data.set_index('model')
    dataset_data.columns = metric_names
    
    sns.heatmap(dataset_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=axes[idx], cbar_kws={'label': 'Score'})
    axes[idx].set_title(f'{dataset}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Model' if idx == 0 else '')

plt.tight_layout()
plt.savefig('reports/figures/model_performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Performance heatmap saved")
plt.close()

# ============================================================================
# 6. SAVE COMPREHENSIVE RESULTS
# ============================================================================
print("\n6. Saving Comprehensive Results...")

# Save combined results
all_results.to_csv('reports/all_models_evaluation.csv', index=False)
print("✓ All results saved to: reports/all_models_evaluation.csv")

# Save best models summary
best_models.to_csv('reports/best_models_summary.csv', index=False)
print("✓ Best models summary saved to: reports/best_models_summary.csv")

# ============================================================================
# 7. FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)
print(f"Total Models Evaluated: {len(all_results)}")
print(f"Datasets: {len(datasets)}")
print(f"Model Types: 3 (Logistic Regression, Random Forest, XGBoost)")
print(f"\nBest Overall Performance:")
best_overall = all_results.loc[all_results['test_roc_auc'].idxmax()]
print(f"  Dataset: {best_overall['dataset']}")
print(f"  Model: {best_overall['model']}")
print(f"  ROC-AUC: {best_overall['test_roc_auc']:.4f}")
print(f"\nAll models meet target (ROC-AUC > 0.80): {(all_results['test_roc_auc'] > 0.80).all()}")
print("="*70)

print("\n✓ Model evaluation and comparison complete!")
print("\nGenerated visualizations:")
print("  - reports/figures/model_comparison_roc_auc.png")
print("  - reports/figures/model_comparison_accuracy.png")
print("  - reports/figures/model_comparison_f1.png")
print("  - reports/figures/model_performance_heatmap.png")
