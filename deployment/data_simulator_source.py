"""Data simulator source for the telecom churn MLOps workflow.

This script prepares the datasets required for the Event Watcher
and Scheduler demonstrations.

The script:
1. Loads current.csv and current_stress.csv.
2. Combines the datasets.
3. Removes duplicate customer IDs.
4. Splits the combined data into incoming batches.
5. Saves the batches to data/data_sources/incoming_batches/.
6. Creates the Event Watcher and Scheduler training datasets
   from train.csv if they do not already exist.

Workflow datasets:
    data/data_sources/train_trigger.csv
    data/data_sources/train_scheduler.csv

Incoming batches:
    data/data_sources/incoming_batches/
"""

from pathlib import Path
import shutil

import pandas as pd


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

# This file is inside deployment/.
#
# deployment/data_simulator_source.py
#         |
#         +-- parents[0] -> deployment
#         +-- parents[1] -> project root

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "data_sources"

INCOMING_BATCH_DIR = DATA_DIR / "incoming_batches"


# ============================================================
# SOURCE DATASETS
# ============================================================

TRAINING_DATA = DATA_DIR / "train.csv"

NORMAL_DATA = DATA_DIR / "current.csv"

STRESS_DATA = DATA_DIR / "current_stress.csv"


# ============================================================
# EVENT WATCHER DATASET
# ============================================================

EVENT_WATCHER_DATA = (
    DATA_DIR / "event_watcher_data" / "train_trigger.csv"
)


# ============================================================
# SCHEDULER DATASET
# ============================================================

SCHEDULER_DATA = (
    DATA_DIR / "scheduler_data" / "train_scheduler.csv"
)


def prepare_workflow_datasets():
    """Create workflow training datasets from the initial train.csv."""

    print("\n" + "=" * 60)
    print("PREPARING MLOPS WORKFLOW TRAINING DATASETS")
    print("=" * 60)

    if not TRAINING_DATA.exists():
        raise FileNotFoundError(
            f"\nInitial training dataset not found:\n{TRAINING_DATA}"
        )

    # --------------------------------------------------------
    # Create Event Watcher training dataset
    # --------------------------------------------------------

    EVENT_WATCHER_DATA.parent.mkdir(parents=True, exist_ok=True)

    if not EVENT_WATCHER_DATA.exists():
        shutil.copy2(TRAINING_DATA, EVENT_WATCHER_DATA)

        print("\nEvent Watcher training dataset created:")
        print(f"  {EVENT_WATCHER_DATA}")
        print("  Source:")
        print(f"  {TRAINING_DATA}")
    else:
        print("\nEvent Watcher training dataset already exists.")
        print(f"  {EVENT_WATCHER_DATA}")
        print("  Existing dataset was preserved.")

    # --------------------------------------------------------
    # Create Scheduler training dataset
    # --------------------------------------------------------

    SCHEDULER_DATA.parent.mkdir(parents=True, exist_ok=True)

    if not SCHEDULER_DATA.exists():
        shutil.copy2(TRAINING_DATA, SCHEDULER_DATA)

        print("\nScheduler training dataset created:")
        print(f"  {SCHEDULER_DATA}")
        print("  Source:")
        print(f"  {TRAINING_DATA}")
    else:
        print("\nScheduler training dataset already exists.")
        print(f"  {SCHEDULER_DATA}")
        print("  Existing dataset was preserved.")


def prepare_incoming_batches():
    """Combine normal/stress data and create three incoming batches."""

    print("\n" + "=" * 60)
    print("DATA PREPARATION FOR EVENT WATCHER DEMONSTRATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load source datasets
    # --------------------------------------------------------

    normal_df = pd.read_csv(NORMAL_DATA)
    stress_df = pd.read_csv(STRESS_DATA)

    print(f"\nNormal dataset rows : {len(normal_df)}")
    print(f"Stress dataset rows : {len(stress_df)}")

    # --------------------------------------------------------
    # Remove duplicate customer IDs
    # --------------------------------------------------------

    normal_df = (
        normal_df
        .drop_duplicates(subset=["customer_id"])
        .reset_index(drop=True)
    )

    stress_df = (
        stress_df
        .drop_duplicates(subset=["customer_id"])
        .reset_index(drop=True)
    )

    print(f"\nNormal rows after duplicate removal : {len(normal_df)}")
    print(f"Stress rows after duplicate removal : {len(stress_df)}")

    # --------------------------------------------------------
    # Combine normal and stress data
    #
    # Normal records are placed first and stress records
    # are placed afterward.
    # --------------------------------------------------------

    combined_df = pd.concat(
        [normal_df, stress_df],
        ignore_index=True
    )

    print(f"\nCombined dataset rows: {len(combined_df)}")

    # --------------------------------------------------------
    # Divide combined dataset into three batches
    # --------------------------------------------------------

    total_rows = len(combined_df)
    batch_size = total_rows // 3

    batch_1 = combined_df.iloc[0:batch_size].copy()

    batch_2 = combined_df.iloc[
        batch_size:batch_size * 2
    ].copy()

    batch_3 = combined_df.iloc[
        batch_size * 2:
    ].copy()

    # --------------------------------------------------------
    # Create incoming batch directory
    # --------------------------------------------------------

    INCOMING_BATCH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save incoming batches
    # --------------------------------------------------------

    batch_1.to_csv(
        INCOMING_BATCH_DIR / "batch_1.csv",
        index=False
    )

    batch_2.to_csv(
        INCOMING_BATCH_DIR / "batch_2.csv",
        index=False
    )

    batch_3.to_csv(
        INCOMING_BATCH_DIR / "batch_3.csv",
        index=False
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nDataset preparation completed!")

    print("\nIncoming batches:")
    print(f"Batch 1: {batch_1.shape}")
    print(f"Batch 2: {batch_2.shape}")
    print(f"Batch 3: {batch_3.shape}")

    print("\nFiles saved:")
    print(f"  {INCOMING_BATCH_DIR / 'batch_1.csv'}")
    print(f"  {INCOMING_BATCH_DIR / 'batch_2.csv'}")
    print(f"  {INCOMING_BATCH_DIR / 'batch_3.csv'}")


def main():
    # --------------------------------------------------------
    # Step 1:
    # Prepare Event Watcher and Scheduler datasets.
    # --------------------------------------------------------

    prepare_workflow_datasets()

    # --------------------------------------------------------
    # Step 2:
    # Prepare incoming batches.
    # --------------------------------------------------------

    prepare_incoming_batches()


if __name__ == "__main__":
    main()


# ============================================================
# SCRIPT SUMMARY
# ============================================================

# This script prepares the data required for the MLOps
# Event Watcher and Scheduler demonstrations.
#
# Initial dataset:
#
#     data/data_sources/train.csv
#
# Workflow datasets:
#
#     data/data_sources/event_watcher_data/train_trigger.csv
#     data/data_sources/scheduler_data/train_scheduler.csv
#
# Incoming source data:
#
#     data/data_sources/current.csv
#     data/data_sources/current_stress.csv
#
# Incoming batches:
#
#     data/data_sources/incoming_batches/
#
# The workflow datasets are created only if they do not
# already exist, so existing updated datasets are preserved.
#
# Normal data is placed first and stress data afterward
# before the combined dataset is split into three batches.
#
# The generated batches can then be used by the Event Watcher
# and Scheduler workflows.