
"""
SCRIPT SUMMARY:
Updates the demonstration training dataset with incoming customer data.
Validates schemas, prevents duplicate customer IDs, and saves the
accumulated training dataset for DVC versioning.
"""

import sys
from pathlib import Path

import pandas as pd


# Define the project directory and training dataset.
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train_demo.csv"


def update_training_data(incoming_file):
    """Validate and append new customer records to the training dataset."""

    incoming_path = PROJECT_ROOT / "data" / incoming_file

    if not TRAIN_PATH.exists() or not incoming_path.exists():
        raise FileNotFoundError("Training or incoming dataset not found.")

    # Load the existing training data and incoming data.
    train_df = pd.read_csv(TRAIN_PATH)
    incoming_df = pd.read_csv(incoming_path)

    print(f"Existing training rows: {len(train_df)}")
    print(f"Incoming rows: {len(incoming_df)}")

    # Ensure both datasets have the same columns.
    if list(train_df.columns) != list(incoming_df.columns):
        raise ValueError("Dataset schemas do not match.")

    # Remove duplicate customer IDs within the incoming batch.
    incoming_df = incoming_df.drop_duplicates(
        subset=["customer_id"]
    )

    # Exclude customers already present in the training dataset.
    new_records = incoming_df[
        ~incoming_df["customer_id"].isin(train_df["customer_id"])
    ]

    # Append only genuinely new customer records.
    updated_df = pd.concat(
        [train_df, new_records],
        ignore_index=True
    )

    # Save the updated training dataset.
    updated_df.to_csv(TRAIN_PATH, index=False)

    print("\nTraining dataset updated successfully.")
    print(f"New records added: {len(new_records)}")
    print(f"Updated training rows: {len(updated_df)}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python update_training_data.py "
            "current.csv or current_stress.csv"
        )
        sys.exit(1)

    update_training_data(sys.argv[1])
