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
    make_approval_decision,
    get_pending_approval
)

from candidate_training import run_candidate_training


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
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# Pending approval request endpoint
# --------------------------------------------------

@app.get("/pending-approval")
def pending_approval():

    request = get_pending_approval()

    if request is None:
        return {
            "status": "no_pending_request",
            "message": "There are no pending approval requests."
        }

    return request


# --------------------------------------------------
# Human approval endpoint
# --------------------------------------------------

@app.post("/approval")
def approve_model(request: ApprovalRequest):

    try:

        # --------------------------------------------------
        # Check what type of approval is currently pending.
        #
        # There are two approval stages:
        #
        # 1. Retraining approval
        #    Drift was detected and human approval is required
        #    before candidate models can be trained.
        #
        # 2. Production approval
        #    Candidate models have already been trained and
        #    a human must choose which model goes to production.
        # --------------------------------------------------

        pending_request = get_pending_approval()

        if pending_request is None:
            raise ValueError(
                "There is no pending approval request."
            )

        approval_type = pending_request.get(
            "approval_type",
            "production"
        )

        # --------------------------------------------------
        # RETRAINING APPROVAL
        # --------------------------------------------------

        if approval_type == "retraining":

            # First update the retraining approval request.
            result = make_approval_decision(
                decision=request.decision,
                reviewer=request.reviewer,
                confirm_lower_f1=request.confirm_lower_f1
            )

            # If retraining was rejected, stop here.
            if result.get("status") == "retraining_rejected":
                return result

            # Only continue when retraining was approved.
            if result.get("status") == "retraining_approved":

                training_data_path = (
                    pending_request.get("training_data_path")
                )

                if not training_data_path:
                    raise ValueError(
                        "Training data path is missing from "
                        "the retraining approval request."
                    )

                print("\n" + "=" * 60)
                print("RETRAINING APPROVED")
                print("=" * 60)
                print(
                    f"Training data : {training_data_path}"
                )
                print(
                    f"Reviewer      : {request.reviewer}"
                )
                print("Starting candidate training...")

                # --------------------------------------------------
                # Start candidate training only after human approval.
                # --------------------------------------------------

                training_result = run_candidate_training(
                    training_data_path=training_data_path,
                    trigger_source=pending_request.get(
                        "trigger_source",
                        "Unknown"
                    ),
                    retraining_approved=True
                )

                # --------------------------------------------------
                # Candidate training should create a new production
                # approval request after successful training.
                # --------------------------------------------------

                if training_result is None:

                    return {
                        **result,
                        "message": (
                            "Retraining approved, but candidate "
                            "training did not produce new results."
                        )
                    }

                return {
                    **result,
                    "training_status": "completed",
                    "message": (
                        "Retraining approved and candidate "
                        "models trained successfully. "
                        "Production approval is now required."
                    )
                }

            return result

        # --------------------------------------------------
        # PRODUCTION APPROVAL
        # --------------------------------------------------

        result = make_approval_decision(
            decision=request.decision,
            reviewer=request.reviewer,
            confirm_lower_f1=request.confirm_lower_f1
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):

    try:

        model = load_production_model()

        customer_data = pd.DataFrame(
            [customer.model_dump()]
        )

        prediction = int(
            model.predict(customer_data)[0]
        )

        probabilities = model.predict_proba(
            customer_data
        )[0]

        churn_probability = float(
            probabilities[1]
        )

        return PredictionResponse(
            churn_prediction=prediction,
            churn_label=(
                "Churn"
                if prediction == 1
                else "No Churn"
            ),
            churn_probability=round(
                churn_probability,
                4
            ),
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


# ==========================================
# SCRIPT SUMMARY
# ==========================================
# This FastAPI application provides the Telecom Churn
# Prediction API and manages the human approval workflow.
#
# The approval workflow now has two separate stages:
#
# 1. RETRAINING APPROVAL
#    When Event Watcher detects data drift, a retraining
#    approval request is created. Candidate training does
#    not start until a human approves the retraining request.
#
# 2. PRODUCTION APPROVAL
#    After candidate training completes, a separate
#    production approval request is created. A human can
#    approve Logistic Regression, approve Decision Tree,
#    or reject both candidates.
#
# The API also provides:
# - Health checking
# - Production model information
# - Candidate/production model comparison
# - Pending approval status
# - Customer churn prediction
#
# Candidate training is triggered only after explicit
# human approval for retraining.
# Production promotion remains a separate approval step.