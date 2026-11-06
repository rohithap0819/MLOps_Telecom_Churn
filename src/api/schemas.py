
from typing import Optional

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Input schema for customer churn prediction."""

    gender: str
    senior_citizen: str
    partner: str
    dependents: str

    tenure_months: int = Field(ge=0, le=100)

    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str

    contract: str
    paperless_billing: str
    payment_method: str

    monthly_charges: float = Field(ge=0)
    total_charges: Optional[float] = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    """Response returned by the prediction API."""

    churn_prediction: int
    churn_label: str
    churn_probability: float


class ApprovalRequest(BaseModel):
    decision: str = Field(
        ...,
        description=(
            "Approval decision. Use approve_retraining or "
            "reject_retraining for retraining approval, and "
            "logistic_regression, decision_tree, or reject_all "
            "for production approval."
        )
    )
    reviewer: str = Field(
        ...,
        description="Name of the human reviewer."
    )
    confirm_lower_f1: bool = False