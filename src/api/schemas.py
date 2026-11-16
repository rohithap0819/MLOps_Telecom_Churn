from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 1. Customer Data Schema
# ============================================================

class CustomerData(BaseModel):
    """
    Input schema for customer churn prediction.

    This class defines the data that the /predict API endpoint
    expects when a user wants to predict churn for a customer.

    Pydantic automatically validates the incoming API request.
    If a required field is missing or contains an invalid value,
    FastAPI returns a validation error instead of passing
    incorrect data to the ML model.
    """

    # Categorical customer information.
    # These values are expected to be strings because the
    # trained model's preprocessing pipeline expects categorical
    # values in this form.
    gender: str
    senior_citizen: str
    partner: str
    dependents: str

    # Number of months the customer has stayed with the company.
    #
    # ge=0 means the value cannot be negative.
    # le=100 means the value cannot be greater than 100.
    #
    # Field() allows us to add validation rules directly
    # to the API input.
    tenure_months: int = Field(ge=0, le=100)

    # Customer service-related categorical features.
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str

    # Billing and contract information.
    contract: str
    paperless_billing: str
    payment_method: str

    # Monthly amount charged to the customer.
    #
    # ge=0 prevents negative charges from being accepted.
    monthly_charges: float = Field(ge=0)

    # Total amount charged to the customer.
    #
    # Optional means the value can be omitted from the request.
    #
    # default=None means that if the field is not provided,
    # its value will be None.
    #
    # ge=0 still prevents a negative value when a value is provided.
    total_charges: Optional[float] = Field(default=None, ge=0)


# ============================================================
# 2. Prediction Response Schema
# ============================================================

class PredictionResponse(BaseModel):
    """
    Response returned by the prediction API.

    This defines the structure of the response returned by
    the /predict endpoint after the ML model makes a prediction.
    """

    # Numeric prediction returned by the ML model.
    #
    # Typically:
    # 0 -> No Churn
    # 1 -> Churn
    churn_prediction: int

    # Human-readable version of the prediction.
    #
    # Example:
    # "Churn"
    # "No Churn"
    churn_label: str

    # Probability that the customer will churn.
    #
    # Example:
    # 0.82 means an estimated 82% probability of churn.
    churn_probability: float


# ============================================================
# 3. Approval Request Schema
# ============================================================

class ApprovalRequest(BaseModel):
    """
    Request schema used when a human submits an approval decision.

    The Streamlit dashboard sends this information to the
    FastAPI /approval endpoint.

    The same schema supports both stages of the approval workflow:

    1. Retraining approval
    2. Production model approval
    """

    # Unique ID of the approval request currently shown
    # in the dashboard.
    #
    # This is important because multiple approval requests
    # can exist during the retraining workflow.
    request_id: str = Field(
        ...,
        description="Unique ID of the pending approval request."
    )

    # Human decision submitted through the dashboard.
    #
    # The exact accepted values depend on which approval stage
    # is currently active.
    decision: str = Field(
        ...,
        description=(
            "Approval decision. Use approve_retraining or "
            "reject_retraining for retraining approval, and "
            "logistic_regression, decision_tree, or reject_all "
            "for production approval."
        )
    )

    # Name of the person who is making the approval decision.
    reviewer: str = Field(
        ...,
        description="Name of the human reviewer."
    )

    # Used when the approval workflow requires explicit
    # confirmation for a candidate model with a lower F1 score.
    confirm_lower_f1: bool = False


# ============================================================
# SCRIPT SUMMARY
# ============================================================
# Purpose:
#     Defines the Pydantic data schemas used by the FastAPI
#     application.
#
# What this script does:
#     1. Defines the expected customer input fields.
#     2. Validates numeric values such as tenure and charges.
#     3. Defines the structure of prediction responses.
#     4. Defines the structure of human approval requests.
#
# Input:
#     JSON data received by the FastAPI endpoints.
#
# Output:
#     Validated Python objects that FastAPI can safely pass
#     to the prediction and approval workflows.
#
# Role in the MLOps project:
#     This file forms the validation layer between the API
#     users/dashboard and the underlying ML workflow.
#
# Main schemas:
#
#     CustomerData
#          |
#          v
#     Validates customer information
#          |
#          v
#     ML Prediction
#
#     ApprovalRequest
#          |
#          v
#     Validates human approval decision
#          |
#          v
#     Approval Workflow
#
#     PredictionResponse
#          |
#          v
#     Defines the structure of the API prediction result.
# ============================================================