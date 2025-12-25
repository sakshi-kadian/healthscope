"""
HealthScope API - Pydantic Schemas
Request and response models for all endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# ============================================================================
# DIABETES SCHEMAS
# ============================================================================

class DiabetesInput(BaseModel):
    """Input schema for diabetes prediction"""
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    blood_pressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, le=70, description="Body mass index")
    diabetes_pedigree: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 2,
                "glucose": 120,
                "blood_pressure": 70,
                "skin_thickness": 20,
                "insulin": 80,
                "bmi": 25.5,
                "diabetes_pedigree": 0.5,
                "age": 35
            }
        }

class DiabetesOutput(BaseModel):
    """Output schema for diabetes prediction"""
    prediction: int = Field(..., description="Prediction: 0 (No Diabetes) or 1 (Diabetes)")
    probability: float = Field(..., ge=0, le=1, description="Probability of diabetes")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    top_features: List[Dict[str, float]] = Field(..., description="Top contributing features")
    
# ============================================================================
# HEART DISEASE SCHEMAS
# ============================================================================

class HeartDiseaseInput(BaseModel):
    """Input schema for heart disease prediction"""
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex: 0 (Female), 1 (Male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.5,
                "slope": 1,
                "ca": 0,
                "thal": 2
            }
        }

class HeartDiseaseOutput(BaseModel):
    """Output schema for heart disease prediction"""
    prediction: int = Field(..., description="Prediction: 0 (No Disease) or 1 (Disease)")
    probability: float = Field(..., ge=0, le=1, description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    top_features: List[Dict[str, float]] = Field(..., description="Top contributing features")

# ============================================================================
# OBESITY SCHEMAS
# ============================================================================

class ObesityInput(BaseModel):
    """Input schema for obesity prediction"""
    age: int = Field(..., ge=0, le=120, description="Age in years")
    height: float = Field(..., ge=0.5, le=2.5, description="Height in meters")
    weight: float = Field(..., ge=20, le=300, description="Weight in kilograms")
    fcvc: float = Field(..., ge=1, le=3, description="Frequency of vegetable consumption (1-3)")
    ncp: float = Field(..., ge=1, le=4, description="Number of main meals (1-4)")
    ch2o: float = Field(..., ge=1, le=3, description="Daily water consumption (1-3 liters)")
    faf: float = Field(..., ge=0, le=3, description="Physical activity frequency (0-3)")
    tue: float = Field(..., ge=0, le=2, description="Time using technology (0-2 hours)")
    
    # Extended fields for full model feature mapping
    gender: Optional[str] = "Male"
    family_history_with_overweight: Optional[str] = "yes"
    favc: Optional[str] = "yes"
    smoke: Optional[str] = "no"
    scc: Optional[str] = "no"
    calc: Optional[str] = "Sometimes"
    mtrans: Optional[str] = "Public_Transportation"
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 30,
                "height": 1.75,
                "weight": 75,
                "fcvc": 2.5,
                "ncp": 3,
                "ch2o": 2,
                "faf": 2,
                "tue": 1
            }
        }

class ObesityOutput(BaseModel):
    """Output schema for obesity prediction"""
    prediction: str = Field(..., description="Predicted obesity category")
    probability: float = Field(..., ge=0, le=1, description="Confidence of prediction")
    bmi: float = Field(..., description="Calculated BMI")
    top_features: List[Dict[str, float]] = Field(..., description="Top contributing features")

# ============================================================================
# COMBINED PREDICTION SCHEMA
# ============================================================================

class CombinedInput(BaseModel):
    """Input schema for combined health assessment"""
    # Demographics
    age: int = Field(..., ge=0, le=120)
    sex: Optional[int] = Field(None, ge=0, le=1)
    height: Optional[float] = Field(None, ge=0.5, le=2.5)
    weight: Optional[float] = Field(None, ge=20, le=300)
    
    # Diabetes-specific
    glucose: Optional[float] = Field(None, ge=0, le=300)
    blood_pressure: Optional[float] = Field(None, ge=0, le=200)
    insulin: Optional[float] = Field(None, ge=0, le=900)
    
    # Heart-specific
    chest_pain: Optional[int] = Field(None, ge=0, le=3)
    cholesterol: Optional[float] = Field(None, ge=0, le=600)
    max_heart_rate: Optional[float] = Field(None, ge=0, le=250)
    
    # Lifestyle
    physical_activity: Optional[float] = Field(None, ge=0, le=3)
    vegetable_consumption: Optional[float] = Field(None, ge=1, le=3)

class CombinedOutput(BaseModel):
    """Output schema for combined health assessment"""
    diabetes_risk: Optional[Dict] = None
    heart_disease_risk: Optional[Dict] = None
    obesity_assessment: Optional[Dict] = None
    overall_health_score: float = Field(..., ge=0, le=100)
    recommendations: List[str] = Field(..., description="Health recommendations")

# ============================================================================
# HEALTH CHECK SCHEMA
# ============================================================================

class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    timestamp: str = Field(..., description="Current timestamp")

# ============================================================================
# ERROR SCHEMA
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional error details")
