# HealthScope API

FastAPI-based REST API for chronic disease risk prediction.

## Features

- **Diabetes Prediction**: Predict diabetes risk based on health metrics
- **Heart Disease Prediction**: Assess heart disease risk
- **Obesity Classification**: Classify obesity levels
- **SHAP Explainability**: Get feature importance for predictions
- **Interactive Docs**: Auto-generated API documentation

## Installation

```bash
# Install dependencies
pip install fastapi uvicorn pydantic python-multipart requests
```

## Running the API

### Start the Server

```bash
# From project root
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### Stop the Server

Press `CTRL+C` in the terminal where the server is running.

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Health Check
```
GET /health
```

Returns API status and loaded models.

### Diabetes Prediction
```
POST /predict/diabetes
```

**Request Body**:
```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 80,
  "bmi": 25.5,
  "diabetes_pedigree": 0.5,
  "age": 35
}
```

**Response**:
```json
{
  "prediction": 0,
  "probability": 0.25,
  "risk_level": "Low",
  "top_features": [...]
}
```

### Heart Disease Prediction
```
POST /predict/heart
```

**Request Body**:
```json
{
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
```

### Obesity Prediction
```
POST /predict/obesity
```

**Request Body**:
```json
{
  "age": 30,
  "height": 1.75,
  "weight": 75,
  "fcvc": 2.5,
  "ncp": 3,
  "ch2o": 2,
  "faf": 2,
  "tue": 1
}
```

## Testing

Run the test script:

```bash
python api/test_api.py
```

## Important Notes

### After Code Changes

If you modify `api/main.py` or `api/schemas.py`, you need to **restart the server**:

1. Stop the server (CTRL+C)
2. Start it again: `python api/main.py`

### Feature Engineering

The API applies the same feature engineering as the training phase:
- Diabetes: BMI-glucose interactions, risk scores
- Heart: Engineered cardiac features
- Obesity: BMI calculations, lifestyle scores

### CORS

CORS is enabled for all origins (`*`) for development. In production, specify exact origins in `api/main.py`.

## Troubleshooting

### Models Not Loading

Ensure models are in `models_saved/` directory:
- `diabetes_model.pkl`
- `heart_model.pkl`
- `obesity_model.pkl`
- `obesity_label_encoder.pkl`

### Port Already in Use

If port 8000 is busy, change the port in `api/main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port
```

### Feature Mismatch Errors

The API expects features in the exact same format as training. If you get errors:
1. Check the feature engineering logic matches training
2. Verify all required features are present
3. Ensure data types are correct

## Development

### Project Structure

```
api/
├── __init__.py
├── main.py          # Main FastAPI application
├── schemas.py       # Pydantic models
├── test_api.py      # API tests
└── README.md        # This file
```

### Adding New Endpoints

1. Define schema in `schemas.py`
2. Add endpoint in `main.py`
3. Test with `test_api.py`

## Version

**Current Version**: 1.0.0

## License

Part of the HealthScope project.
