"""
SCRIPT SUMMARY:
Appends incoming customer batches to the main training dataset.

The script:
1. Loads the existing data/train.csv.
2. Loads an incoming batch.
3. Validates that both datasets have the same schema.
4. Removes duplicate customer IDs from the incoming batch.
5. Prevents customer IDs already present in train.csv from being added again.
6. Appends only new records to train.csv.
7. Returns the number of new records and updated dataset size.

This dataset is tracked by DVC so every update can create a new
dataset version.
"""

import sys
from pathlib import Path

import pandas as pd


# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent

# Main training dataset used by the MLOps pipeline.
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"


def update_training_data(incoming_file):
    """
    Validate and append an incoming batch to train.csv.

    Parameters
    ----------
    incoming_file : str
        Path relative to the project root.

    Returns
    -------
    dict
        Information about the dataset update.
    """

    incoming_path = PROJECT_ROOT / incoming_file

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not incoming_path.exists():
        raise FileNotFoundError(
            f"Incoming dataset not found: {incoming_path}"
        )

    # Load existing training data and incoming batch.
    train_df = pd.read_csv(TRAIN_PATH)
    incoming_df = pd.read_csv(incoming_path)

    print(f"Existing training rows : {len(train_df)}")
    print(f"Incoming batch rows    : {len(incoming_df)}")

    # Ensure both datasets have exactly the same schema.
    if list(train_df.columns) != list(incoming_df.columns):
        raise ValueError(
            "Dataset schemas do not match.\n"
            f"Training columns: {list(train_df.columns)}\n"
            f"Incoming columns: {list(incoming_df.columns)}"
        )

    # Remove duplicate customer IDs within the incoming batch.
    incoming_df = incoming_df.drop_duplicates(
        subset=["customer_id"]
    )

    # Keep only customers that do not already exist in train.csv.
    new_records = incoming_df[
        ~incoming_df["customer_id"].isin(
            train_df["customer_id"]
        )
    ]

    # Append only genuinely new customer records.
    updated_df = pd.concat(
        [train_df, new_records],
        ignore_index=True
    )

    # Save the updated training dataset.
    updated_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    new_records_count = len(new_records)
    updated_row_count = len(updated_df)

    print("\nTraining dataset updated successfully.")
    print(f"New records added     : {new_records_count}")
    print(f"Updated training rows : {updated_row_count}")

    return {
        "training_path": str(TRAIN_PATH),
        "incoming_path": str(incoming_path),
        "previous_rows": len(train_df),
        "incoming_rows": len(incoming_df),
        "new_records": new_records_count,
        "updated_rows": updated_row_count,
    }


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python update_training_data.py "
            "data/incoming_batches/batch_1.csv"
        )
        sys.exit(1)

    update_training_data(sys.argv[1])


# ==========================================
# SCRIPT SUMMARY
# ==========================================
# This script updates the main data/train.csv dataset with an incoming batch.
# Duplicate customer IDs inside the incoming batch are removed.
# Customer IDs already present in train.csv are not added again.
# Only genuinely new records are appended.
# The updated train.csv is then ready for DVC versioning.