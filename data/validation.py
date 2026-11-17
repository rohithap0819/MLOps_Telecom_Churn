import pandas as pd
import pandera.pandas as pa

from pandera.errors import SchemaErrors


# ============================================================
# 1. DEFINE THE EXPECTED DATASET SCHEMA
# ============================================================

# Pandera schema used to verify that the telecom churn dataset
# has the expected columns, data types, allowed values, and
# basic data-quality rules.
#
# This validation is intentionally kept separate from the
# modelling code so that the same validation logic can be
# reused by tests and the MLOps workflow.

telecom_schema = pa.DataFrameSchema(
    columns={
        # Customer identifier must be present and unique.
        "customer_id": pa.Column(
            str,
            nullable=False,
            unique=True,
        ),

        # Customer demographic and service-related fields.
        "gender": pa.Column(
            str,
            nullable=False,
        ),
        "senior_citizen": pa.Column(
            str,
            checks=pa.Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "partner": pa.Column(
            str,
            nullable=False,
        ),
        "dependents": pa.Column(
            str,
            nullable=False,
        ),

        # Customer tenure must be a non-negative value
        # within the expected range.
        "tenure_months": pa.Column(
            int,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),

        # Telecom service-related columns.
        "phone_service": pa.Column(
            str,
            nullable=False,
        ),
        "multiple_lines": pa.Column(
            str,
            nullable=False,
        ),
        "internet_service": pa.Column(
            str,
            nullable=False,
        ),
        "online_security": pa.Column(
            str,
            nullable=False,
        ),
        "online_backup": pa.Column(
            str,
            nullable=False,
        ),
        "device_protection": pa.Column(
            str,
            nullable=False,
        ),
        "tech_support": pa.Column(
            str,
            nullable=False,
        ),
        "streaming_tv": pa.Column(
            str,
            nullable=False,
        ),
        "streaming_movies": pa.Column(
            str,
            nullable=False,
        ),
        "contract": pa.Column(
            str,
            nullable=False,
        ),
        "paperless_billing": pa.Column(
            str,
            nullable=False,
        ),
        "payment_method": pa.Column(
            str,
            nullable=False,
        ),

        # Billing-related numeric columns.
        # Charges must not contain negative values.
        "monthly_charges": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "total_charges": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=True,
        ),

        # Target column.
        # Churn must be represented as either 0 or 1.
        "churn": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        ),
    },

    # strict=True ensures that unexpected columns are rejected.
    #
    # coerce=True allows Pandera to convert compatible values
    # into the expected data types during validation.
    strict=True,
    coerce=True,
)


# ============================================================
# 2. VALIDATE A DATAFRAME
# ============================================================

def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a telecom churn DataFrame using Pandera.

    The function checks the DataFrame against the predefined
    telecom_schema.

    Args:
        data: DataFrame containing telecom customer data.

    Returns:
        The validated DataFrame.

    Raises:
        SchemaErrors: If one or more schema validation checks fail.
        SchemaError: If a schema-level validation error occurs.
    """

    print("\nStarting data validation...")

    try:
        # lazy=True collects all validation failures before
        # raising the error. This makes it easier to identify
        # multiple data-quality issues at once.
        validated_data = telecom_schema.validate(
            data,
            lazy=True,
        )

        print("Data validation successful!")
        print(f"Rows validated: {len(validated_data)}")
        print(f"Columns validated: {len(validated_data.columns)}")

        return validated_data

    except SchemaErrors as error:
        # SchemaErrors contains detailed information about
        # individual row/column validation failures.
        print("\nDATA VALIDATION FAILED")
        print(error.failure_cases.to_string(index=False))
        raise

    except pa.errors.SchemaError as error:
        # Handles other schema-level validation errors that
        # are not returned as a collection of failure cases.
        print("\nDATA VALIDATION FAILED")
        print(error)
        raise


# ============================================================
# 3. VALIDATE A CSV FILE
# ============================================================

def validate_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and validate its contents.

    Args:
        file_path: Path to the CSV file that should be validated.

    Returns:
        The validated DataFrame.

    Raises:
        Any validation error raised by validate_data().
    """

    # Load the CSV file into a pandas DataFrame.
    print(f"\nLoading dataset: {file_path}")

    data = pd.read_csv(file_path)

    # Display the original shape before validation so that
    # learners can see how much data is being validated.
    print(f"Original dataset shape: {data.shape}")

    # Pass the loaded DataFrame through the same validation
    # function used for DataFrame-level validation.
    return validate_data(data)


# ============================================================
# FILE SUMMARY
# ============================================================

# Purpose:
#     Validate the structure and data quality of the telecom
#     customer churn dataset using Pandera.
#
# Main responsibilities:
#     1. Define the expected telecom churn dataset schema.
#     2. Validate column names and data types.
#     3. Validate required and unique fields.
#     4. Validate allowed categorical and target values.
#     5. Validate numeric ranges such as tenure and charges.
#     6. Reject unexpected columns using strict schema validation.
#     7. Validate pandas DataFrames using validate_data().
#     8. Load and validate CSV files using validate_csv().
#
# Used by:
#     - tests/test_pipeline.py
#     - MLOps monitoring and data-processing workflows
#
# Important:
#     This file contains data validation logic only.
#     It does not train models or perform model selection.
#
# Data source location:
#     CSV datasets used by the project are stored under:
#
#         data/data_sources/
#
# The validation functions receive the specific file path
# when a CSV needs to be validated.
# ============================================================