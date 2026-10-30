
import json
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    CustomerData,
    PredictionResponse,
    ApprovalRequest
)

from approval import (
    load_candidate_metrics,
    load_production_metrics,
    make_approval_decision
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "production" / "production_model.joblib"
)

REGISTRY_PATH = (
    PROJECT_ROOT / "models" / "production_registry.json"
)


# --------------------------------------------------
# 2. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="Predict customer churn using the approved production model.",
    version="1.0.0",
)


# --------------------------------------------------
# 3. Load production model
# --------------------------------------------------

def load_production_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production model not found. Approve a model first."
        )

    return joblib.load(MODEL_PATH)


def load_production_registry():

    if not REGISTRY_PATH.exists():
        return {}

    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# 4. API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Telecom Churn Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "production_model_available": MODEL_PATH.exists(),
    }


@app.get("/model-info")
def model_info():

    registry = load_production_registry()

    return {
        "production_model": registry.get("production_model"),
        "model_version": registry.get("model_version"),
        "approved_by": registry.get("approved_by"),
        "approval_status": registry.get("approval_status"),
    }


@app.get("/model-comparison")
def model_comparison():
    try:
        candidate_metrics = load_candidate_metrics()
        production_metrics = load_production_metrics()

        return {
            "production": json.loads(
                production_metrics.to_json(orient="records")
            ),
            "candidates": json.loads(
                candidate_metrics.to_json(orient="records")
            )
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/approval")
def approve_model(request: ApprovalRequest):
    try:
        result = make_approval_decision(
            decision=request.decision,
            reviewer=request.reviewer,
            confirm_lower_f1=request.confirm_lower_f1
        )

        return result

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):

    try:
        model = load_production_model()

        customer_data = pd.DataFrame(
            [customer.model_dump()]
        )

        prediction = int(model.predict(customer_data)[0])

        probabilities = model.predict_proba(customer_data)[0]

        churn_probability = float(probabilities[1])

        return PredictionResponse(
            churn_prediction=prediction,
            churn_label="Churn" if prediction == 1 else "No Churn",
            churn_probability=round(churn_probability, 4),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
