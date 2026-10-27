import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# Initialize FastAPI application
app = FastAPI(
    title="Intelligent Loan Risk Prediction API",
    description="Production-ready RESTful API for real-time bank loan risk classification using an Artificial Neural Network (ANN).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessors
MODEL_PATH = "loan_risk_ann.keras"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "label_encoders.pkl"
DATASET_PATH = "Loan Prediction.csv"

model = None
scaler = None
label_encoders = None
metadata_cache = {}


def load_artifacts():
    global model, scaler, label_encoders, metadata_cache
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
        model = load_model(MODEL_PATH)
    if scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file '{SCALER_PATH}' not found.")
        scaler = joblib.load(SCALER_PATH)
    if label_encoders is None:
        if not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError(f"Label encoders file '{ENCODERS_PATH}' not found.")
        label_encoders = joblib.load(ENCODERS_PATH)

    # Cache metadata for dropdowns and validation
    if os.path.exists(DATASET_PATH) and not metadata_cache:
        df = pd.read_csv(DATASET_PATH, encoding="latin1")
        metadata_cache = {
            "professions": sorted(df["Profession"].unique().tolist()),
            "cities": sorted(df["CITY"].unique().tolist()),
            "states": sorted(df["STATE"].unique().tolist()),
            "house_ownership": sorted(df["House_Ownership"].unique().tolist()),
            "car_ownership": sorted(df["Car_Ownership"].unique().tolist()),
            "marital_status": sorted(df["Married/Single"].unique().tolist()),
        }


@app.on_event("startup")
def startup_event():
    load_artifacts()


# --- Pydantic Data Models ---
class LoanApplicant(BaseModel):
    Income: int = Field(..., ge=0, description="Annual income of the applicant in local currency (e.g., INR)", example=1303834)
    Age: int = Field(..., ge=18, le=100, description="Applicant age in years", example=23)
    Experience: int = Field(..., ge=0, le=60, description="Total professional work experience in years", example=3)
    Married_Single: Literal["single", "married"] = Field(..., alias="Married/Single", description="Marital status ('single' or 'married')", example="single")
    House_Ownership: Literal["owned", "rented", "norent_noown"] = Field(..., description="House ownership status", example="rented")
    Car_Ownership: Literal["yes", "no"] = Field(..., description="Car ownership status ('yes' or 'no')", example="no")
    Profession: str = Field(..., description="Applicant profession", example="Mechanical_engineer")
    CITY: str = Field(..., description="City of residence", example="Rewa")
    STATE: str = Field(..., description="State of residence", example="Madhya_Pradesh")
    CURRENT_JOB_YRS: int = Field(..., ge=0, le=50, description="Years in current job", example=3)
    CURRENT_HOUSE_YRS: int = Field(..., ge=0, le=50, description="Years in current house", example=13)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Income": 1303834,
                "Age": 23,
                "Experience": 3,
                "Married/Single": "single",
                "House_Ownership": "rented",
                "Car_Ownership": "no",
                "Profession": "Mechanical_engineer",
                "CITY": "Rewa",
                "STATE": "Madhya_Pradesh",
                "CURRENT_JOB_YRS": 3,
                "CURRENT_HOUSE_YRS": 13
            }
        }


class PredictionResult(BaseModel):
    prediction: str = Field(..., description="'Low Risk' or 'High Risk'")
    risk_flag: int = Field(..., description="0 for Low Risk, 1 for High Risk")
    probability: float = Field(..., description="Raw default probability from ANN sigmoid output (0.0 to 1.0)")
    risk_percentage: float = Field(..., description="Default risk percentage (0% to 100%)")
    confidence: float = Field(..., description="Confidence percentage in predicted classification")
    risk_tier: str = Field(..., description="Tiered category: Ultra Low, Low, Moderate, High, Critical")
    recommendation: str = Field(..., description="Automated business policy recommendation")


class BatchLoanApplicants(BaseModel):
    applicants: List[LoanApplicant]


class BatchPredictionResult(BaseModel):
    total_processed: int
    high_risk_count: int
    low_risk_count: int
    results: List[PredictionResult]


def process_applicant(applicant: LoanApplicant) -> PredictionResult:
    load_artifacts()

    # Build single-row DataFrame
    data_dict = {
        "Income": applicant.Income,
        "Age": applicant.Age,
        "Experience": applicant.Experience,
        "Married/Single": applicant.Married_Single,
        "House_Ownership": applicant.House_Ownership,
        "Car_Ownership": applicant.Car_Ownership,
        "Profession": applicant.Profession,
        "CITY": applicant.CITY,
        "STATE": applicant.STATE,
        "CURRENT_JOB_YRS": applicant.CURRENT_JOB_YRS,
        "CURRENT_HOUSE_YRS": applicant.CURRENT_HOUSE_YRS
    }
    df_row = pd.DataFrame([data_dict])

    # Validate and encode categorical features
    categorical_cols = [
        "Married/Single",
        "House_Ownership",
        "Car_Ownership",
        "Profession",
        "CITY",
        "STATE"
    ]

    for col in categorical_cols:
        val = df_row.at[0, col]
        encoder = label_encoders[col]
        if val not in encoder.classes_:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown category '{val}' for feature '{col}'. Valid values are available at /metadata endpoint."
            )
        df_row[col] = encoder.transform(df_row[col])

    # Feature Scaling
    scaled_data = scaler.transform(df_row)

    # Inference with ANN Model
    prob = float(model.predict(scaled_data, verbose=0)[0][0])

    is_high_risk = prob >= 0.5
    prediction = "High Risk" if is_high_risk else "Low Risk"
    risk_flag = 1 if is_high_risk else 0
    risk_pct = round(prob * 100, 2)
    confidence = round((prob if is_high_risk else (1.0 - prob)) * 100, 2)

    # Risk tiering & strategic credit policy recommendation
    if prob < 0.15:
        risk_tier = "Ultra Low Risk"
        recommendation = "Fast-track approval with prime interest rate and standard loan covenants."
    elif prob < 0.35:
        risk_tier = "Low Risk"
        recommendation = "Standard automated approval. Standard interest rates apply."
    elif prob < 0.50:
        risk_tier = "Moderate Risk"
        recommendation = "Manual review recommended. Require collateral proof or secondary guarantor."
    elif prob < 0.70:
        risk_tier = "High Risk"
        recommendation = "High risk of default. Cap maximum loan amount and request additional income verification."
    else:
        risk_tier = "Critical Risk"
        recommendation = "Reject application or require strict cash collateral / senior co-signer."

    return PredictionResult(
        prediction=prediction,
        risk_flag=risk_flag,
        probability=round(prob, 4),
        risk_percentage=risk_pct,
        confidence=confidence,
        risk_tier=risk_tier,
        recommendation=recommendation
    )


# --- API Routes ---

@app.get("/", tags=["System"])
def root():
    return {
        "project": "Loan Risk Prediction System using Deep Learning (ANN)",
        "framework": "FastAPI + TensorFlow/Keras",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "metadata": "/metadata",
            "predict_single": "/predict",
            "predict_batch": "/predict/batch"
        }
    }


@app.get("/health", tags=["System"])
def health_check():
    load_artifacts()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": label_encoders is not None
    }


@app.get("/metadata", tags=["Metadata"])
def get_metadata():
    load_artifacts()
    return metadata_cache


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_loan_risk(applicant: LoanApplicant):
    """
    Accepts applicant attributes, performs categorical encoding and standard feature scaling,
    runs inference through the trained Artificial Neural Network, and returns default probability
    and credit risk recommendation.
    """
    return process_applicant(applicant)


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Inference"])
def predict_batch_loan_risk(batch: BatchLoanApplicants):
    """
    Batch prediction endpoint for scoring multiple loan applications simultaneously.
    """
    results = [process_applicant(app_item) for app_item in batch.applicants]
    high_count = sum(1 for r in results if r.risk_flag == 1)
    low_count = len(results) - high_count

    return BatchPredictionResult(
        total_processed=len(results),
        high_risk_count=high_count,
        low_risk_count=low_count,
        results=results
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)
