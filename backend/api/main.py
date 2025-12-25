"""
HealthScope API - Simplified Main Application
Using exact training data format for predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os

from schemas import (
    DiabetesInput, DiabetesOutput,
    HeartDiseaseInput, HeartDiseaseOutput,
    ObesityInput, ObesityOutput,
    HealthCheck
)

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="HealthScope API",
    description="Chronic Disease Risk Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

models = {}
MODEL_DIR = "models_saved"

# Load sample data to get exact feature format
sample_diabetes = pd.read_csv('data/processed/diabetes_final.csv').drop('outcome', axis=1).iloc[0:1]
sample_heart = pd.read_csv('data/processed/heart_final.csv').drop('target', axis=1).iloc[0:1]
sample_obesity = pd.read_csv('data/processed/obesity_final.csv')
categorical_cols = ['gender', 'family_history_with_overweight', 'favc', 'caec', 'smoke', 'scc', 'calc', 'mtrans']
sample_obesity = sample_obesity.drop('nobeyesdad', axis=1).drop(columns=categorical_cols, errors='ignore').iloc[0:1]

# ============================================================================
# STARTUP - LOAD MODELS
# ============================================================================

@app.on_event("startup")
async def load_models():
    global models
    try:
        models['diabetes'] = joblib.load(os.path.join(MODEL_DIR, 'diabetes_model.pkl'))
        models['heart'] = joblib.load(os.path.join(MODEL_DIR, 'heart_baseline_lr.pkl'))
        # Compatibility fix: Handle sklearn version mismatch for LogisticRegression
        if not hasattr(models['heart'], 'multi_class'):
            setattr(models['heart'], 'multi_class', 'auto')
            
        models['obesity'] = joblib.load(os.path.join(MODEL_DIR, 'obesity_model.pkl'))
        models['obesity_encoder'] = joblib.load(os.path.join(MODEL_DIR, 'obesity_label_encoder.pkl'))
        print("✓ All models loaded!")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_level(prob: float) -> str:
    return "Low" if prob < 0.3 else "Medium" if prob < 0.7 else "High"

def prepare_diabetes(data: DiabetesInput) -> pd.DataFrame:
    """Prepare diabetes features matching training format"""
    df = sample_diabetes.copy()
    
    # Set basic features
    df['pregnancies'] = data.pregnancies
    df['glucose'] = data.glucose
    df['blood_pressure'] = data.blood_pressure
    df['skin_thickness'] = data.skin_thickness
    df['insulin'] = data.insulin
    df['bmi'] = data.bmi
    df['diabetes_pedigree'] = data.diabetes_pedigree
    df['age'] = data.age
    
    # Recalculate engineered features
    df['bmi_category'] = 1 if data.bmi < 25 else 2 if data.bmi < 30 else 3
    df['age_group'] = 1 if data.age < 30 else 2 if data.age < 50 else 3
    df['glucose_category'] = 1 if data.glucose < 100 else 2 if data.glucose < 126 else 3
    df['bp_category'] = 1 if data.blood_pressure < 80 else 2 if data.blood_pressure < 90 else 3
    df['bmi_glucose_interaction'] = data.bmi * data.glucose / 100
    df['age_glucose_interaction'] = data.age * data.glucose / 100
    df['pregnancies_age_interaction'] = data.pregnancies * data.age / 10
    df['risk_score'] = (df['glucose_category'].values[0] + df['bmi_category'].values[0] + 
                       df['bp_category'].values[0] + df['age_group'].values[0]) / 4
    
    return df

def prepare_heart(data: HeartDiseaseInput) -> pd.DataFrame:
    """Prepare heart features matching training format"""
    df = sample_heart.copy()
    
    # Set basic features
    df['age'] = data.age
    df['sex'] = data.sex
    df['cp'] = data.cp
    df['trestbps'] = data.trestbps
    df['chol'] = data.chol
    df['fbs'] = data.fbs
    df['restecg'] = data.restecg
    df['thalach'] = data.thalach
    df['exang'] = data.exang
    df['oldpeak'] = data.oldpeak
    df['slope'] = data.slope
    df['ca'] = data.ca
    df['thal'] = data.thal
    
    # Recalculate engineered features
    df['age_group'] = 1 if data.age < 50 else 2 if data.age < 60 else 3
    df['chol_category'] = 1 if data.chol < 200 else 2 if data.chol < 240 else 3
    df['bp_category'] = 1 if data.trestbps < 120 else 2 if data.trestbps < 140 else 3
    df['hr_category'] = 1 if data.thalach < 120 else 2 if data.thalach < 150 else 3
    df['age_chol_interaction'] = data.age * data.chol / 100
    df['age_hr_interaction'] = data.age * data.thalach / 100
    df['cp_oldpeak_interaction'] = data.cp * data.oldpeak
    df['risk_score'] = (df['age_group'].values[0] + df['chol_category'].values[0] + 
                       df['bp_category'].values[0]) / 3
    
    return df

def prepare_obesity(data: ObesityInput) -> pd.DataFrame:
    """Prepare obesity features matching training format"""
    df = sample_obesity.copy()
    
    bmi = data.weight / (data.height ** 2)
    
    # Set basic features
    df['age'] = data.age
    df['height'] = data.height
    df['weight'] = data.weight
    df['fcvc'] = data.fcvc
    df['ncp'] = data.ncp
    df['ch2o'] = data.ch2o
    df['faf'] = data.faf
    df['tue'] = data.tue
    df['calculated_bmi'] = bmi
    
    # Recalculate engineered features
    df['bmi_category'] = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
    df['age_group'] = 0 if data.age < 20 else 1 if data.age < 30 else 2 if data.age < 40 else 3
    df['lifestyle_score'] = data.fcvc + data.faf + data.ch2o - data.tue
    df['bmi_activity_interaction'] = bmi * data.faf
    df['age_bmi_interaction'] = data.age * bmi
    df['fcvc_ch2o_interaction'] = data.fcvc * data.ch2o
    
    # Set encoded categorical features to defaults (0 or 1)
    # Set encoded categorical features based on inputs
    df['gender_encoded'] = 1 if data.gender == 'Male' else 0
    df['family_history_with_overweight_encoded'] = 1 if data.family_history_with_overweight == 'yes' else 0
    df['favc_encoded'] = 1 if data.favc == 'yes' else 0
    df['caec_encoded'] = 2 # Default 'Sometimes'
    df['smoke_encoded'] = 1 if data.smoke == 'yes' else 0
    df['scc_encoded'] = 1 if data.scc == 'yes' else 0
    
    # CALC encoding: 0=Always, 1=Frequently, 2=Sometimes, 3=no
    if data.calc == 'Frequently':
        df['calc_encoded'] = 1
    elif data.calc == 'Always':
        df['calc_encoded'] = 0
    elif data.calc == 'no':
        df['calc_encoded'] = 3
    else:
        df['calc_encoded'] = 2
        
    df['mtrans_encoded'] = 1 # Default Public_Transportation
    
    return df

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"message": "HealthScope API", "docs": "/docs"}

@app.get("/health", response_model=HealthCheck)
async def health():
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "diabetes": "diabetes" in models,
            "heart": "heart" in models,
            "obesity": "obesity" in models
        },
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict/diabetes", response_model=DiabetesOutput)
async def predict_diabetes(data: DiabetesInput):
    try:
        features = prepare_diabetes(data)
        pred = models['diabetes'].predict(features)[0]
        prob = models['diabetes'].predict_proba(features)[0][1]
        
        return DiabetesOutput(
            prediction=int(pred),
            probability=float(prob),
            risk_level=get_risk_level(prob),
            top_features=[
                {"glucose": float(data.glucose)},
                {"bmi": float(data.bmi)},
                {"age": float(data.age)}
            ]
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/predict/heart", response_model=HeartDiseaseOutput)
async def predict_heart(data: HeartDiseaseInput):
    try:
        features = prepare_heart(data)
        pred = models['heart'].predict(features)[0]
        prob = models['heart'].predict_proba(features)[0][1]
        
        return HeartDiseaseOutput(
            prediction=int(pred),
            probability=float(prob),
            risk_level=get_risk_level(prob),
            top_features=[
                {"chest_pain": float(data.cp)},
                {"max_hr": float(data.thalach)},
                {"age": float(data.age)}
            ]
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/predict/obesity", response_model=ObesityOutput)
async def predict_obesity(data: ObesityInput):
    try:
        features = prepare_obesity(data)
        bmi = data.weight / (data.height ** 2)
        
        pred_encoded = models['obesity'].predict(features)[0]
        prob = models['obesity'].predict_proba(features)[0].max()
        pred = models['obesity_encoder'].inverse_transform([pred_encoded])[0]
        
        return ObesityOutput(
            prediction=str(pred),
            probability=float(prob),
            bmi=float(bmi),
            top_features=[
                {"bmi": float(bmi)},
                {"weight": float(data.weight)},
                {"activity": float(data.faf)}
            ]
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

# ============================================================================
# COMBINED PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict/all")
async def predict_all(data: dict):
    """
    Combined endpoint for all predictions using actual ML models
    """
    print(f"DEBUG: Received prediction request with data: {data}")
    try:
        # Calculate BMI if not provided
        bmi = data.get('BMI')
        if not bmi:
            weight = data.get('weight', 70)
            height_cm = data.get('height', 170)
            height_m = height_cm / 100
            bmi = weight / (height_m ** 2)
        
        # ===== DIABETES PREDICTION =====
        # Map form toggles to estimated biometrics for the PIMA model
        # HighBP -> Higher Blood Pressure (140 vs 72)
        # HighChol -> Higher Glucose assumption (130 vs 95)
        est_bp = 140 if data.get('HighBP') == 1 else 72
        est_glucose = 135 if data.get('HighChol') == 1 else 95
        
        print(f"DEBUG: Predicting Diabetes with est_bp={est_bp}, est_glucose={est_glucose}")
        try:
            diabetes_in = DiabetesInput(
                pregnancies=0,
                glucose=est_glucose,
                blood_pressure=est_bp,
                skin_thickness=20,
                insulin=79,
                bmi=float(bmi),
                diabetes_pedigree=0.5,
                age=float(data.get('Age', 30))
            )
            diabetes_df = prepare_diabetes(diabetes_in)
            
            # Predict diabetes
            diabetes_pred = models['diabetes'].predict(diabetes_df)[0]
            diabetes_proba_both = models['diabetes'].predict_proba(diabetes_df)[0]
            diabetes_prob = diabetes_proba_both[1]  # Probability of class 1 (diabetes)
            
            print(f"DEBUG: Diabetes prediction={diabetes_pred}, proba_class_0={diabetes_proba_both[0]:.3f}, proba_class_1={diabetes_proba_both[1]:.3f}")
            print("DEBUG: Diabetes Prediction Success")
        except Exception as e:
            print(f"CRITICAL ERROR in DIABETES: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # ===== HEART DISEASE PREDICTION =====
        # Intelligent mapping: form toggles → estimated medical parameters
        
        # Blood Pressure & Cholesterol
        est_trestbps = 150 if data.get('HighBP') == 1 else 120
        est_chol = 250 if data.get('HighChol') == 1 else 200
        
        # Exercise-induced angina (difficulty walking)
        est_exang = 1 if data.get('DiffWalk') == 1 else 0
        
        # Fasting blood sugar
        est_fbs = 1 if est_glucose > 120 else 0
        
        # Max Heart Rate (based on age and physical activity)
        # Active people have better cardiovascular fitness
        base_max_hr = 220 - float(data.get('Age', 30))  # Age-predicted max HR
        if data.get('PhysActivity', 0) == 1:
            est_thalach = base_max_hr * 0.85  # Active: can reach 85% of max
        else:
            est_thalach = base_max_hr * 0.65  # Sedentary: only 65% of max
        
        # Chest Pain Type (based on health history)
        if data.get('HeartDiseaseorAttack', 0) == 1:
            est_cp = 3  # Asymptomatic (previous heart issues)
        elif data.get('HighBP', 0) == 1 or data.get('HighChol', 0) == 1:
            est_cp = 2  # Non-anginal pain
        else:
            est_cp = 0  # Typical angina (healthy)
        
        # ST Depression (oldpeak) - based on overall cardiovascular stress
        cardiovascular_stress = sum([
            data.get('HighBP', 0),
            data.get('HighChol', 0),
            data.get('Smoker', 0),
            data.get('HeartDiseaseorAttack', 0)
        ])
        est_oldpeak = min(cardiovascular_stress * 0.5, 3.0)  # 0-3 range
        
        print(f"DEBUG: Heart params - BP:{est_trestbps}, Chol:{est_chol}, MaxHR:{est_thalach:.0f}, CP:{est_cp}, Oldpeak:{est_oldpeak}")
        try:
            heart_in = HeartDiseaseInput(
                age=float(data.get('Age', 30)),
                sex=int(data.get('Sex', 1)),
                cp=est_cp,
                trestbps=est_trestbps,
                chol=est_chol,
                fbs=est_fbs,
                restecg=0,
                thalach=est_thalach,
                exang=est_exang,
                oldpeak=est_oldpeak,
                slope=1,
                ca=0,
                thal=2
            )
            heart_df = prepare_heart(heart_in)
            
            # DEBUG: Print actual features being sent to model
            print(f"DEBUG: Heart features shape: {heart_df.shape}")
            print(f"DEBUG: Heart features (first 10 cols): {heart_df.iloc[0, :10].to_dict()}")
            
            # Predict heart disease
            heart_pred = models['heart'].predict(heart_df)[0]
            heart_proba_both = models['heart'].predict_proba(heart_df)[0]
            heart_prob = heart_proba_both[1]  # Probability of class 1 (disease)
            
            print(f"DEBUG: Heart prediction={heart_pred}, proba_class_0={heart_proba_both[0]:.3f}, proba_class_1={heart_proba_both[1]:.3f}")
            print("DEBUG: Heart Prediction Success")
        except Exception as e:
            print(f"CRITICAL ERROR in HEART: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # ===== OBESITY PREDICTION =====
        # Comprehensive mapping of lifestyle factors
        
        # FCVC (Vegetables) - 1.0 to 3.0 scale
        has_veggies = data.get('Veggies', 0) == 1
        has_fruits = data.get('Fruits', 0) == 1
        est_fcvc = 3.0 if (has_veggies and has_fruits) else 2.0 if (has_veggies or has_fruits) else 1.0
        
        # FAF (Physical Activity Frequency) - 0.0 to 3.0 scale
        est_faf = 2.0 if data.get('PhysActivity', 0) == 1 else 0.0
        
        # SMOKE
        est_smoke = 'yes' if data.get('Smoker', 0) == 1 else 'no'
        
        # CALC (Alcohol) - 'no', 'Sometimes', 'Frequently', 'Always'
        est_calc = 'Frequently' if data.get('HvyAlcoholConsump', 0) == 1 else 'no'

        print(f"DEBUG: Constructing ObesityInput with: smoke={est_smoke}, calc={est_calc}, faf={est_faf}, fcvc={est_fcvc}")
        
        try:
            obesity_in = ObesityInput(
                age=float(data.get('Age', 30)),
                height=float(data.get('height', 170)) / 100,
                weight=float(data.get('weight', 70)),
                fcvc=est_fcvc,
                ncp=3.0,
                ch2o=2.0,
                faf=est_faf,
                tue=1.0,
                calc=est_calc,
                mtrans='Public_Transportation',
                family_history_with_overweight='yes',
                favc='yes',
                smoke=est_smoke,
                scc='no',
                gender='Female' if int(data.get('Sex', 1)) == 0 else 'Male'
            )
        except Exception as validation_error:
            print(f"VALIDATION ERROR: {validation_error}")
            raise validation_error
        
        obesity_df = prepare_obesity(obesity_in)
        
        # Predict obesity
        obesity_pred_encoded = models['obesity'].predict(obesity_df)[0]
        obesity_prob = models['obesity'].predict_proba(obesity_df)[0].max()
        obesity_pred = models['obesity_encoder'].inverse_transform([obesity_pred_encoded])[0]
        
        # Return combined results
        # NOTE: Heart disease dataset has inverted labels, we flip its probability
        # Diabetes model has been retrained with corrected labels, use directly
        corrected_heart_prob = 1.0 - float(heart_prob)
        
        return {
            "diabetes_risk": float(diabetes_prob),
            "diabetes_prediction": "At Risk" if diabetes_pred == 1 else "Healthy",
            "heart_risk": corrected_heart_prob,
            "heart_prediction": "At Risk" if corrected_heart_prob > 0.5 else "Healthy",
            "obesity_risk": float(obesity_prob),
            "obesity_level": str(obesity_pred),
            "bmi": float(bmi)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error in combined prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
