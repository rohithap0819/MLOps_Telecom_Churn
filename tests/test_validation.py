
import pandas as pd
import pytest

from pandera.errors import SchemaErrors
from src.data.validate import validate_data


@pytest.fixture
def valid_data():
    """Load a small sample of valid telecom churn data."""

    return pd.read_csv("data/train.csv").head(10)


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