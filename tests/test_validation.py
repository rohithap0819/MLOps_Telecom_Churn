
import pandas as pd
import pytest

from pandera.errors import SchemaErrors
from src.data.validate import validate_data


@pytest.fixture
def valid_data():
    """Create a small valid telecom churn dataset for testing."""

    return pd.DataFrame({
        "customer_id": [f"TEST{i:04d}" for i in range(1, 11)],
        "gender": ["Male", "Female"] * 5,
        "senior_citizen": ["Yes", "No"] * 5,
        "partner": ["Yes", "No"] * 5,
        "dependents": ["No", "Yes"] * 5,
        "tenure_months": [12, 24, 36, 48, 60] * 2,
        "phone_service": ["Yes"] * 10,
        "multiple_lines": ["No", "Yes"] * 5,
        "internet_service": ["DSL", "Fiber optic"] * 5,
        "online_security": ["Yes", "No"] * 5,
        "online_backup": ["No", "Yes"] * 5,
        "device_protection": ["Yes", "No"] * 5,
        "tech_support": ["No", "Yes"] * 5,
        "streaming_tv": ["Yes", "No"] * 5,
        "streaming_movies": ["No", "Yes"] * 5,
        "contract": ["Month-to-month", "One year"] * 5,
        "paperless_billing": ["Yes", "No"] * 5,
        "payment_method": ["Electronic check", "Mailed check"] * 5,
        "monthly_charges": [50.0, 70.0] * 5,
        "total_charges": [600.0, 1680.0] * 5,
        "churn": [0, 1] * 5,
    })


def test_valid_data_passes_validation(valid_data):
    """Valid data should pass Pandera validation."""

    validated_data = validate_data(valid_data)

    assert len(validated_data) == 10
    assert len(validated_data.columns) == 21


def test_duplicate_customer_id_fails(valid_data):
    """Duplicate customer IDs should fail validation."""

    valid_data.loc[1, "customer_id"] = valid_data.loc[0, "customer_id"]

    with pytest.raises(SchemaErrors):
        validate_data(valid_data)


def test_invalid_churn_value_fails(valid_data):
    """Churn must contain only 0 or 1."""

    valid_data.loc[0, "churn"] = 5

    with pytest.raises(SchemaErrors):
        validate_data(valid_data)


def test_missing_column_fails(valid_data):
    """Missing required columns should fail validation."""

    valid_data = valid_data.drop(columns=["monthly_charges"])

    with pytest.raises(SchemaErrors):
        validate_data(valid_data)


# SCRIPT SUMMARY:
# This test module validates the telecom churn data validation pipeline.
# It uses a self-contained sample dataset instead of the local training CSV.
# It tests valid data, duplicate customer IDs, invalid churn values,
# and missing required columns.
