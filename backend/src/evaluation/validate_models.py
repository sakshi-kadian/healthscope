"""
HealthScope - Model Validation & Testing
Objective: Validate model serialization, test edge cases, and document limitations.
"""

import pandas as pd
import numpy as np
import joblib
import os

print("="*70)
print("HEALTHSCOPE - MODEL VALIDATION & TESTING")
print("="*70)

# ============================================================================
# 1. TEST MODEL SERIALIZATION/DESERIALIZATION
# ============================================================================
print("\n1. Testing Model Serialization/Deserialization...")

models_to_test = {
    'Diabetes': 'models_saved/diabetes_model.pkl',
    'Heart Disease': 'models_saved/heart_model.pkl',
    'Obesity': 'models_saved/obesity_model.pkl'
}

serialization_results = []

for dataset, model_path in models_to_test.items():
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Check if model has required methods
        has_predict = hasattr(model, 'predict')
        has_predict_proba = hasattr(model, 'predict_proba')
        
        serialization_results.append({
            'dataset': dataset,
            'model_path': model_path,
            'loaded': True,
            'has_predict': has_predict,
            'has_predict_proba': has_predict_proba,
            'status': 'PASS' if (has_predict and has_predict_proba) else 'FAIL'
        })
        
        print(f"✓ {dataset}: Model loaded successfully")
        
    except Exception as e:
        serialization_results.append({
            'dataset': dataset,
            'model_path': model_path,
            'loaded': False,
            'has_predict': False,
            'has_predict_proba': False,
            'status': 'FAIL',
            'error': str(e)
        })
        print(f"✗ {dataset}: Failed to load - {str(e)}")

# Save serialization test results
ser_df = pd.DataFrame(serialization_results)
ser_df.to_csv('reports/model_serialization_test.csv', index=False)
print(f"\n✓ Serialization test results saved")

# ============================================================================
# 2. TEST EDGE CASES
# ============================================================================
print("\n2. Testing Edge Cases...")

edge_case_results = []

# Test Diabetes model with edge cases
print("\n  Testing Diabetes Model...")
diabetes_model = joblib.load('models_saved/diabetes_model.pkl')
diabetes_data = pd.read_csv('data/processed/diabetes_final.csv')

# Drop categorical columns if present
categorical_cols = ['gender', 'family_history_with_overweight', 'favc', 'caec', 'smoke', 'scc', 'calc', 'mtrans']
diabetes_features = diabetes_data.drop('outcome', axis=1).drop(columns=categorical_cols, errors='ignore')

# Edge case 1: All zeros (normalized)
edge_zeros = pd.DataFrame(np.zeros((1, diabetes_features.shape[1])), columns=diabetes_features.columns)
try:
    pred = diabetes_model.predict(edge_zeros)
    edge_case_results.append({'dataset': 'Diabetes', 'case': 'All Zeros', 'status': 'PASS', 'prediction': pred[0]})
    print(f"    ✓ All zeros: Prediction = {pred[0]}")
except Exception as e:
    edge_case_results.append({'dataset': 'Diabetes', 'case': 'All Zeros', 'status': 'FAIL', 'error': str(e)})
    print(f"    ✗ All zeros: Failed - {str(e)}")

# Edge case 2: All ones (normalized)
edge_ones = pd.DataFrame(np.ones((1, diabetes_features.shape[1])), columns=diabetes_features.columns)
try:
    pred = diabetes_model.predict(edge_ones)
    edge_case_results.append({'dataset': 'Diabetes', 'case': 'All Ones', 'status': 'PASS', 'prediction': pred[0]})
    print(f"    ✓ All ones: Prediction = {pred[0]}")
except Exception as e:
    edge_case_results.append({'dataset': 'Diabetes', 'case': 'All Ones', 'status': 'FAIL', 'error': str(e)})
    print(f"    ✗ All ones: Failed - {str(e)}")

# Test Heart model
print("\n  Testing Heart Disease Model...")
heart_model = joblib.load('models_saved/heart_model.pkl')
heart_data = pd.read_csv('data/processed/heart_final.csv')
heart_features = heart_data.drop('target', axis=1)

edge_zeros_heart = pd.DataFrame(np.zeros((1, heart_features.shape[1])), columns=heart_features.columns)
try:
    pred = heart_model.predict(edge_zeros_heart)
    edge_case_results.append({'dataset': 'Heart', 'case': 'All Zeros', 'status': 'PASS', 'prediction': pred[0]})
    print(f"    ✓ All zeros: Prediction = {pred[0]}")
except Exception as e:
    edge_case_results.append({'dataset': 'Heart', 'case': 'All Zeros', 'status': 'FAIL', 'error': str(e)})

# Test Obesity model
print("\n  Testing Obesity Model...")
obesity_model = joblib.load('models_saved/obesity_model.pkl')
obesity_data = pd.read_csv('data/processed/obesity_final.csv')
obesity_features = obesity_data.drop('nobeyesdad', axis=1).drop(columns=categorical_cols, errors='ignore')

edge_zeros_obesity = pd.DataFrame(np.zeros((1, obesity_features.shape[1])), columns=obesity_features.columns)
try:
    pred = obesity_model.predict(edge_zeros_obesity)
    edge_case_results.append({'dataset': 'Obesity', 'case': 'All Zeros', 'status': 'PASS', 'prediction': pred[0]})
    print(f"    ✓ All zeros: Prediction = {pred[0]}")
except Exception as e:
    edge_case_results.append({'dataset': 'Obesity', 'case': 'All Zeros', 'status': 'FAIL', 'error': str(e)})

# Save edge case results
edge_df = pd.DataFrame(edge_case_results)
edge_df.to_csv('reports/edge_case_test_results.csv', index=False)
print(f"\n✓ Edge case test results saved")

# ============================================================================
# 3. CREATE MODEL CARDS
# ============================================================================
print("\n3. Creating Model Cards...")

os.makedirs('reports/model_cards', exist_ok=True)

# Diabetes Model Card
diabetes_card = """
# MODEL CARD: DIABETES PREDICTION

## Model Details
- **Model Type**: Random Forest Classifier
- **Purpose**: Predict diabetes risk based on health metrics
- **Version**: 1.0
- **Date**: December 2025

## Performance Metrics
- **ROC-AUC**: 0.8365
- **Accuracy**: 76.62%
- **Precision**: 0.6667
- **Recall**: 0.6667
- **F1-Score**: 0.6667

## Training Data
- **Dataset**: Pima Indians Diabetes Database
- **Samples**: 768 (614 train, 154 test)
- **Features**: 16 (after feature engineering)
- **Target**: Binary (0: No Diabetes, 1: Diabetes)

## Intended Use
- **Primary Use**: Educational/Portfolio demonstration
- **NOT for**: Clinical diagnosis or medical decision-making
- **Users**: Data science students, researchers

## Limitations
- Trained on specific population (Pima Indians)
- May not generalize to other demographics
- Requires normalized input features
- Should not replace medical professional judgment

## Ethical Considerations
- Model predictions are probabilistic, not definitive
- Always consult healthcare professionals
- Privacy: No patient data stored
- Bias: Limited to training population demographics
"""

with open('reports/model_cards/diabetes_model_card.md', 'w', encoding='utf-8') as f:
    f.write(diabetes_card)
print("✓ Diabetes model card created")

# Heart Disease Model Card
heart_card = """
# MODEL CARD: HEART DISEASE PREDICTION

## Model Details
- **Model Type**: Logistic Regression
- **Purpose**: Predict heart disease risk
- **Version**: 1.0
- **Date**: December 2025

## Performance Metrics
- **ROC-AUC**: 0.8626
- **Accuracy**: 77.05%
- **Precision**: 0.7879
- **Recall**: 0.7879
- **F1-Score**: 0.7879

## Training Data
- **Dataset**: Cleveland Heart Disease Database
- **Samples**: 302 (241 train, 61 test)
- **Features**: 21 (after feature engineering)
- **Target**: Binary (0: No Disease, 1: Disease)

## Intended Use
- **Primary Use**: Educational/Portfolio demonstration
- **NOT for**: Clinical diagnosis or medical decision-making
- **Users**: Data science students, researchers

## Limitations
- Small dataset (302 samples after cleaning)
- Requires normalized input features
- Should not replace medical professional judgment
- May not generalize to all populations

## Ethical Considerations
- Model predictions are probabilistic, not definitive
- Always consult healthcare professionals
- Privacy: No patient data stored
- Bias: Limited to training population
"""

with open('reports/model_cards/heart_model_card.md', 'w', encoding='utf-8') as f:
    f.write(heart_card)
print("✓ Heart disease model card created")

# Obesity Model Card
obesity_card = """
# MODEL CARD: OBESITY LEVEL PREDICTION

## Model Details
- **Model Type**: Random Forest Classifier
- **Purpose**: Predict obesity level category
- **Version**: 1.0
- **Date**: December 2025

## Performance Metrics
- **ROC-AUC**: 0.9998 (Multi-class, weighted)
- **Accuracy**: 98.09%
- **Precision**: 0.9815 (weighted)
- **Recall**: 0.9809 (weighted)
- **F1-Score**: 0.9808 (weighted)

## Training Data
- **Dataset**: Obesity Levels Dataset
- **Samples**: 2087 (1669 train, 418 test)
- **Features**: 23 (after feature engineering)
- **Target**: Multi-class (7 obesity categories)
- **Classes**: Insufficient Weight, Normal Weight, Overweight Level I & II, 
              Obesity Type I, II, & III

## Intended Use
- **Primary Use**: Educational/Portfolio demonstration
- **NOT for**: Clinical diagnosis or medical decision-making
- **Users**: Data science students, researchers

## Limitations
- Requires normalized input features
- Should not replace medical professional judgment
- BMI-based categories may not account for muscle mass
- Lifestyle factors are self-reported

## Ethical Considerations
- Model predictions are probabilistic, not definitive
- Always consult healthcare professionals
- Privacy: No patient data stored
- Bias: May not generalize to all populations
- Sensitivity: Weight-related predictions require careful communication
"""

with open('reports/model_cards/obesity_model_card.md', 'w', encoding='utf-8') as f:
    f.write(obesity_card)
print("✓ Obesity model card created")

# ============================================================================
# 4. DOCUMENT MODEL LIMITATIONS
# ============================================================================
print("\n4. Documenting Model Limitations...")

limitations_doc = """
# MODEL LIMITATIONS - HEALTHSCOPE PROJECT

## General Limitations (All Models)

### 1. Educational Purpose Only
- **NOT for clinical use**: These models are for educational/portfolio purposes
- **NOT medical devices**: Do not use for actual medical diagnosis
- **Require professional validation**: Any real-world use requires clinical validation

### 2. Data Limitations
- **Small datasets**: Limited training samples (302-2087 samples)
- **Specific populations**: May not generalize to all demographics
- **Historical data**: Training data may not reflect current medical knowledge

### 3. Technical Limitations
- **Normalized inputs required**: Features must be preprocessed correctly
- **No missing value handling**: Requires complete feature sets
- **Static models**: Do not adapt to new data without retraining

### 4. Performance Limitations
- **Not 100% accurate**: All models have error rates
- **Class imbalance**: Some models trained on imbalanced datasets
- **Overfitting risk**: High training accuracy may not generalize

## Model-Specific Limitations

### Diabetes Model
- **Population bias**: Trained on Pima Indians population
- **Limited features**: Only 8 original features
- **Recall**: 66.67% - misses 1/3 of diabetes cases
- **Age range**: May not work well for very young/old patients

### Heart Disease Model
- **Small dataset**: Only 302 unique samples (after removing 723 duplicates)
- **Feature engineering**: Relies heavily on engineered features
- **Binary only**: Cannot predict disease severity
- **Missing features**: Lacks some important cardiac markers

### Obesity Model
- **Self-reported data**: Lifestyle factors may be inaccurate
- **BMI limitations**: Doesn't account for muscle mass, body composition
- **Category boundaries**: Rigid classification may not reflect health nuances
- **Multi-class complexity**: 7 classes may be too granular for some uses

## Recommendations for Use

### DO:
✓ Use for learning and demonstration
✓ Understand the preprocessing requirements
✓ Validate predictions with domain experts
✓ Document all assumptions and limitations
✓ Test thoroughly before any deployment

### DON'T:
✗ Use for actual medical diagnosis
✗ Deploy without clinical validation
✗ Assume 100% accuracy
✗ Ignore ethical considerations
✗ Use on populations different from training data

## Future Improvements

1. **Larger datasets**: Collect more diverse training data
2. **External validation**: Test on independent datasets
3. **Feature expansion**: Add more relevant health markers
4. **Ensemble methods**: Combine multiple models
5. **Continuous learning**: Update models with new data
6. **Explainability**: Add SHAP values for transparency
7. **Calibration**: Improve probability estimates
8. **Fairness testing**: Evaluate across demographic groups

## Ethical Considerations

- **Informed consent**: Users must understand limitations
- **Privacy**: No patient data should be stored
- **Bias**: Be aware of population-specific training
- **Transparency**: Always disclose model limitations
- **Accountability**: Clear responsibility for predictions
- **Beneficence**: Ensure models do more good than harm

---
**Last Updated**: December 2025
**Version**: 1.0
"""

with open('reports/MODEL_LIMITATIONS.md', 'w', encoding='utf-8') as f:
    f.write(limitations_doc)
print("✓ Model limitations documented")

# ============================================================================
# 5. FINAL VALIDATION SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

print(f"\nSerialization Tests:")
for result in serialization_results:
    status_symbol = "✓" if result['status'] == 'PASS' else "✗"
    print(f"  {status_symbol} {result['dataset']}: {result['status']}")

print(f"\nEdge Case Tests:")
passed = sum(1 for r in edge_case_results if r['status'] == 'PASS')
total = len(edge_case_results)
print(f"  Passed: {passed}/{total}")

print(f"\nDocumentation Created:")
print(f"  ✓ 3 Model cards")
print(f"  ✓ Limitations document")
print(f"  ✓ Test results saved")

print("="*70)

print("\n✓ Model validation and testing complete!")
print("\nGenerated files:")
print("  - reports/model_serialization_test.csv")
print("  - reports/edge_case_test_results.csv")
print("  - reports/model_cards/diabetes_model_card.md")
print("  - reports/model_cards/heart_model_card.md")
print("  - reports/model_cards/obesity_model_card.md")
print("  - reports/MODEL_LIMITATIONS.md")
