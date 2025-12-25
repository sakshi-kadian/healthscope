
# EXAMPLE EXPLANATIONS - HEALTHSCOPE

## Diabetes Prediction Examples

### Example 1: High Risk Patient

**Patient Profile**:
- Age: 55 years
- BMI: 32 (Obese)
- Glucose: 180 mg/dL (High)
- Blood Pressure: 140/90 (Elevated)
- Family History: Yes

**Prediction**: 78% Diabetes Risk (High)

**SHAP Explanation**:
```
Base Risk (Average): 35%
Your Risk: 78%
Difference: +43%

Contributing Factors:
+ Glucose (180): +25% ← Primary concern
+ BMI (32): +15% ← Secondary concern
+ Family History: +8%
+ Age (55): +5%
- Blood Pressure (normal range): -5%
- Regular Exercise: -5%
```

**Plain English**:
"Your diabetes risk is significantly higher than average (78% vs 35%). The main reason is your elevated blood glucose level, which alone increases your risk by 25 percentage points. Your BMI in the obese range adds another 15%. However, your regular exercise routine is helping to reduce your risk by 5%."

**Recommendations**:
1. **Priority**: Work with your doctor on glucose control
2. **Important**: Gradual weight loss (target BMI < 30)
3. **Maintain**: Keep up regular exercise
4. **Monitor**: Regular blood sugar checks

---

### Example 2: Low Risk Patient

**Patient Profile**:
- Age: 28 years
- BMI: 22 (Normal)
- Glucose: 95 mg/dL (Normal)
- Blood Pressure: 115/75 (Normal)
- Family History: No

**Prediction**: 12% Diabetes Risk (Low)

**SHAP Explanation**:
```
Base Risk (Average): 35%
Your Risk: 12%
Difference: -23%

Contributing Factors:
- Young Age (28): -12% ← Protective
- Normal BMI (22): -8% ← Protective
- Normal Glucose (95): -5%
- No Family History: -3%
+ Sedentary Lifestyle: +5%
```

**Plain English**:
"Your diabetes risk is much lower than average (12% vs 35%). Your young age and healthy weight are your biggest protective factors. However, increasing your physical activity could reduce your risk even further."

**Recommendations**:
1. **Maintain**: Healthy weight and diet
2. **Improve**: Add 30 minutes of exercise daily
3. **Monitor**: Annual check-ups
4. **Prevent**: Maintain healthy lifestyle

---

## Heart Disease Prediction Examples

### Example 1: High Risk Patient

**Patient Profile**:
- Age: 62 years
- Chest Pain: Typical Angina
- Max Heart Rate: 110 bpm (Low)
- Cholesterol: 280 mg/dL (High)
- Resting BP: 150/95 (High)

**Prediction**: 85% Heart Disease Risk (High)

**SHAP Explanation**:
```
Base Risk: 45%
Your Risk: 85%
Difference: +40%

Contributing Factors:
+ Chest Pain (Typical Angina): +30% ← Critical
+ Low Max Heart Rate (110): +15%
+ High Cholesterol (280): +10%
+ Age (62): +5%
- No Smoking: -10%
- Normal Blood Sugar: -10%
```

**Plain English**:
"Your heart disease risk is very high (85%). The most significant factor is your chest pain pattern, which is typical of angina. Combined with your low maximum heart rate during exercise and high cholesterol, these factors substantially increase your risk."

**Recommendations**:
1. **Urgent**: Consult cardiologist immediately
2. **Testing**: Stress test, ECG, possible angiogram
3. **Medication**: Likely need cholesterol medication
4. **Lifestyle**: Diet changes, monitored exercise
5. **Follow-up**: Regular cardiology appointments

---

## Obesity Classification Examples

### Example 1: Obesity Type II

**Patient Profile**:
- BMI: 37 (Obesity Type II)
- Weight: 105 kg
- Height: 1.68 m
- Physical Activity: Low
- Vegetable Consumption: Low

**Prediction**: Obesity Type II (98% confidence)

**SHAP Explanation**:
```
Base Category: Normal Weight
Your Category: Obesity Type II

Contributing Factors:
+ High BMI (37): +45% ← Primary
+ High Weight (105kg): +25%
+ Low Physical Activity: +15%
+ Low Vegetable Intake: +8%
+ High Calorie Intake: +5%
```

**Plain English**:
"Your BMI of 37 places you in the Obesity Type II category. The model is 98% confident in this classification. Your low physical activity and diet patterns are contributing factors that can be modified."

**Recommendations**:
1. **Medical**: Consult doctor for health assessment
2. **Nutrition**: Work with dietitian
3. **Exercise**: Start gradual activity program
4. **Support**: Consider support groups
5. **Goal**: Aim for 5-10% weight loss initially

---

## Dashboard Snippets

### Quick Explanation Template
```python
def generate_quick_explanation(prediction, top_features):
    risk_level = "High" if prediction > 0.7 else "Medium" if prediction > 0.3 else "Low"
    
    explanation = f"Your risk is {risk_level} ({prediction*100:.0f}%).\n\n"
    explanation += "Main factors:\n"
    
    for feature, value in top_features[:3]:
        direction = "increases" if value > 0 else "decreases"
        explanation += f"• {feature}: {direction} risk\n"
    
    return explanation
```

### Detailed Explanation Template
```python
def generate_detailed_explanation(model_name, prediction, shap_values, features):
    template = f"""
    {model_name} Risk Assessment
    
    Your Risk: {prediction*100:.1f}%
    Average Risk: {base_value*100:.1f}%
    
    Key Factors Affecting Your Risk:
    
    Increasing Risk:
    {format_positive_factors(shap_values, features)}
    
    Decreasing Risk:
    {format_negative_factors(shap_values, features)}
    
    Recommendations:
    {generate_recommendations(shap_values, features)}
    """
    return template
```

---
**Version**: 1.0  
**Last Updated**: December 2025  
**Purpose**: Reference Examples
