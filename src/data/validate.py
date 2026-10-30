
import pandas as pd
import pandera.pandas as pa

from pandera.errors import SchemaErrors


# --------------------------------------------------
# 1. Define the expected dataset schema
# --------------------------------------------------

telecom_schema = pa.DataFrameSchema(
    columns={
        "customer_id": pa.Column(
            str,
            nullable=False,
            unique=True,
        ),
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
        "tenure_months": pa.Column(
            int,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),
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
        "churn": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        ),
    },
    strict=True,
    coerce=True,
)


# --------------------------------------------------
# 2. Validate the dataset
# --------------------------------------------------

def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a telecom churn DataFrame using Pandera.

    Returns the validated DataFrame.
    Raises an error if the dataset violates the schema.
    """

    print("\nStarting data validation...")

    try:
        validated_data = telecom_schema.validate(
            data,
            lazy=True,
        )

        print("Data validation successful!")
        print(f"Rows validated: {len(validated_data)}")
        print(f"Columns validated: {len(validated_data.columns)}")

        return validated_data

    except SchemaErrors as error:
        print("\nDATA VALIDATION FAILED")
        print(error.failure_cases.to_string(index=False))
        raise

    except pa.errors.SchemaError as error:
        print("\nDATA VALIDATION FAILED")
        print(error)
        raise


# --------------------------------------------------
# 3. Validate a CSV file
# --------------------------------------------------

def validate_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and validate its contents.
    """

    print(f"\nLoading dataset: {file_path}")

    data = pd.read_csv(file_path)

    print(f"Original dataset shape: {data.shape}")

    return validate_data(data)
