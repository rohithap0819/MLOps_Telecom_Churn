"""
SCRIPT SUMMARY:
Appends incoming customer batches to the specified training dataset.

The script:
1. Loads the existing training dataset.
2. Loads an incoming batch.
3. Validates that both datasets have the same schema.
4. Removes duplicate customer IDs from the incoming batch.
5. Prevents customer IDs already present in the training dataset from
   being added again.
6. Appends only new records to the training dataset.
7. Returns the number of new records and updated dataset size.

The training dataset path is provided by the workflow that calls this
function. This allows the Event Watcher and Scheduler to use separate
training datasets.

The dataset is tracked by DVC so every update can create a new
dataset version.
"""

import sys
from pathlib import Path

import pandas as pd


# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent


def update_training_data(incoming_file, training_data_path):
    """
    Validate and append an incoming batch to the specified training dataset.

    Parameters
    ----------
    incoming_file : str
        Path to the incoming CSV relative to the project root.

    training_data_path : str
        Path to the training dataset relative to the project root.

    Returns
    -------
    dict
        Information about the dataset update.
    """

    # Convert both relative paths into full paths.
    incoming_path = PROJECT_ROOT / incoming_file
    training_path = PROJECT_ROOT / training_data_path

    # Make sure the training dataset already exists.
    if not training_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {training_path}"
        )

    # Make sure the incoming CSV exists.
    if not incoming_path.exists():
        raise FileNotFoundError(
            f"Incoming dataset not found: {incoming_path}"
        )

    # Load existing training data and incoming batch.
    train_df = pd.read_csv(training_path)
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

    # Keep only customers that do not already exist
    # in the current training dataset.
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
        training_path,
        index=False
    )

    new_records_count = len(new_records)
    updated_row_count = len(updated_df)

    print("\nTraining dataset updated successfully.")
    print(f"Training dataset     : {training_path}")
    print(f"New records added    : {new_records_count}")
    print(f"Updated training rows: {updated_row_count}")

    return {
        "training_path": str(training_path),
        "incoming_path": str(incoming_path),
        "previous_rows": len(train_df),
        "incoming_rows": len(incoming_df),
        "new_records": new_records_count,
        "updated_rows": updated_row_count,
    }


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python update_training_data.py "
            "<incoming_csv> <training_dataset_csv>"
        )
        sys.exit(1)

    update_training_data(
        sys.argv[1],
        sys.argv[2]
    )


# ==========================================
# SCRIPT SUMMARY
# ==========================================
# This script updates a specified training dataset with an incoming batch.
#
# The training dataset path is provided by the calling workflow, allowing
# Event Watcher and Scheduler to maintain separate datasets.
#
# Duplicate customer IDs inside the incoming batch are removed.
# Customer IDs already present in the training dataset are not added again.
# Only genuinely new records are appended.
#
# The updated training dataset is then ready for DVC versioning.