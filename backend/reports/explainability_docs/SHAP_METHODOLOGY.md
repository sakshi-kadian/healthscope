
# SHAP METHODOLOGY - HEALTHSCOPE PROJECT

## What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to explain the output of machine learning models. It connects optimal credit allocation with local explanations using Shapley values from cooperative game theory.

## Why SHAP?

### Advantages
1. **Theoretically Sound**: Based on Shapley values from game theory
2. **Model-Agnostic**: Works with any machine learning model
3. **Consistent**: Same feature importance regardless of other features
4. **Accurate**: Provides exact feature attributions
5. **Interpretable**: Easy to understand visualizations

### Use Cases in HealthScope
- Explain individual patient risk predictions
- Identify most important health factors
- Validate model decisions
- Build trust with healthcare providers
- Support clinical decision-making

## SHAP in HealthScope

### Models Explained
1. **Diabetes Prediction** (Random Forest)
2. **Heart Disease Prediction** (Logistic Regression)
3. **Obesity Classification** (Random Forest)

### Explainer Types Used

#### TreeExplainer
- **Used For**: Random Forest models (Diabetes, Obesity)
- **Speed**: Fast (optimized for tree-based models)
- **Accuracy**: Exact SHAP values
- **Method**: Uses tree structure for efficient computation

#### LinearExplainer
- **Used For**: Logistic Regression (Heart Disease)
- **Speed**: Very fast
- **Accuracy**: Exact for linear models
- **Method**: Analytical solution for linear models

## SHAP Values Interpretation

### What is a SHAP Value?
A SHAP value represents the contribution of a feature to the prediction, compared to the average prediction.

### Formula
```
Prediction = Base Value + sum(SHAP values for all features)
```

### Example (Diabetes)
```
Base Value (average risk): 0.35
+ glucose SHAP value: +0.15
+ BMI SHAP value: +0.10
+ age SHAP value: -0.05
= Final Prediction: 0.55 (55% diabetes risk)
```

### Positive vs Negative SHAP Values
- **Positive (+)**: Feature increases the prediction
  - For diabetes: Higher glucose → Higher risk
- **Negative (-)**: Feature decreases the prediction
  - For diabetes: Younger age → Lower risk

## Visualization Types

### 1. Summary Plot (Beeswarm)
**Purpose**: Show feature importance and effects across all samples
**Interpretation**:
- Features sorted by importance (top to bottom)
- Each dot is one patient
- Color shows feature value (red=high, blue=low)
- Position shows SHAP value (right=increases risk, left=decreases)

**Example**: 
- Red dots on right = High feature value increases risk
- Blue dots on left = Low feature value decreases risk

### 2. Bar Plot
**Purpose**: Show average feature importance
**Interpretation**:
- Longer bars = More important features
- Shows mean absolute SHAP value
- Simple ranking of features

### 3. Waterfall Plot
**Purpose**: Explain individual predictions
**Interpretation**:
- Starts from base value (average)
- Each bar shows one feature's contribution
- Red bars push prediction up
- Blue bars push prediction down
- Ends at final prediction

### 4. Force Plot
**Purpose**: Visualize prediction for one instance
**Interpretation**:
- Red features push higher
- Blue features push lower
- Width shows magnitude of effect

## Mathematical Foundation

### Shapley Values
From cooperative game theory, Shapley values provide a fair distribution of "payout" among "players" (features).

**Properties**:
1. **Efficiency**: Sum of SHAP values = Prediction - Base value
2. **Symmetry**: Equal features get equal values
3. **Dummy**: Zero-effect features get zero value
4. **Additivity**: Values add up correctly

### Computation
For a prediction f(x) and feature i:
```
φᵢ = Σ [|S|! (|F|-|S|-1)! / |F|!] × [f(S∪{i}) - f(S)]
```
Where:
- S = subset of features
- F = all features
- φᵢ = SHAP value for feature i

## Implementation in HealthScope

### Code Structure
```python
# 1. Load model and data
model = joblib.load('models_saved/diabetes_model.pkl')
X = data.drop('target', axis=1)

# 2. Create explainer
explainer = shap.TreeExplainer(model)

# 3. Calculate SHAP values
shap_values = explainer.shap_values(X)

# 4. Visualize
shap.summary_plot(shap_values, X)
```

### Performance Optimization
- **Sample Size**: Use 100-200 samples for visualization
- **Background Data**: Use training data subset
- **Caching**: Save computed SHAP values
- **Parallel**: Use n_jobs=-1 for speed

## Validation

### Consistency Checks
1. **Sum Check**: SHAP values sum to prediction difference
2. **Symmetry**: Identical features have identical values
3. **Monotonicity**: Higher feature values → consistent direction

### Quality Metrics
- **Computation Time**: < 1 minute per model
- **Memory Usage**: < 500MB
- **Accuracy**: Exact for tree/linear models

## Limitations

### Technical Limitations
1. **Computational Cost**: Expensive for large datasets
2. **Correlation**: Doesn't handle correlated features perfectly
3. **Causality**: Shows association, not causation
4. **Model-Specific**: Explains model, not reality

### Practical Limitations
1. **Interpretation**: Requires domain knowledge
2. **Complexity**: Can be hard to explain to non-technical users
3. **Assumptions**: Assumes feature independence
4. **Scope**: Limited to model's training distribution

## Best Practices

### Do's ✓
- Use appropriate explainer for model type
- Sample data for faster computation
- Validate SHAP values sum correctly
- Combine with domain expertise
- Use multiple visualization types
- Document assumptions clearly

### Don'ts ✗
- Don't use for causal inference
- Don't ignore feature correlations
- Don't over-interpret small values
- Don't use without domain validation
- Don't assume model = reality
- Don't skip consistency checks

## References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS.
2. Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games.
3. SHAP Documentation: https://shap.readthedocs.io/

## HealthScope-Specific Notes

### Diabetes Model
- **Explainer**: TreeExplainer (Random Forest)
- **Key Finding**: Glucose-BMI interaction most important
- **Validation**: Clinically consistent

### Heart Disease Model
- **Explainer**: LinearExplainer (Logistic Regression)
- **Key Finding**: Chest pain type dominates
- **Validation**: Aligns with medical knowledge

### Obesity Model
- **Explainer**: TreeExplainer (Random Forest)
- **Key Finding**: BMI is primary predictor
- **Validation**: Expected and validated

---
**Version**: 1.0  
**Last Updated**: December 2025  
**Author**: HealthScope Team
