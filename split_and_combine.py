"""
SCRIPT SUMMARY:
Prepares the datasets required for the Event Watcher and Scheduler
demonstrations.

The script:

1. Loads current.csv.
2. Loads current_stress.csv.
3. Combines both datasets into one dataset.
4. Removes duplicate customer IDs.
5. Places normal data first and stress data after it.
6. Divides the combined dataset into three incoming batches.
7. Saves the three batches inside data/incoming_batches/.
8. Creates the Event Watcher training dataset from data/train.csv
   if it does not already exist.
9. Creates the Scheduler training dataset from data/train.csv
   if it does not already exist.

The two workflow training datasets are created only once.
If they already exist, they are preserved so that the Event Watcher
and Scheduler can continue using their updated datasets.

Training datasets:

    data/event_watcher_data/train_trigger.csv
    data/scheduler_data/train_scheduler.csv

These files are later used by their respective MLOps workflows
and can be versioned using DVC.

"""

from pathlib import Path
import shutil

import pandas as pd


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

# Incoming batches used by the Event Watcher demonstration.
INCOMING_BATCH_DIR = (
    DATA_DIR / "incoming_batches"
)


# ============================================================
# SOURCE DATASETS
# ============================================================

# Initial training dataset used by the main ML workflow.
TRAINING_DATA = (
    DATA_DIR / "train.csv"
)

# Normal/current customer data used to create incoming batches.
NORMAL_DATA = (
    DATA_DIR / "current.csv"
)

# Stress/drift customer data used to create incoming batches.
STRESS_DATA = (
    DATA_DIR / "current_stress.csv"
)


# ============================================================
# EVENT WATCHER DATASET
# ============================================================

# Existing mutable training dataset used by Event Watcher.
EVENT_WATCHER_DATA = (
    DATA_DIR
    / "event_watcher_data"
    / "train_trigger.csv"
)


# ============================================================
# SCHEDULER DATASET
# ============================================================

# Existing mutable training dataset used by Scheduler.
SCHEDULER_DATA = (
    DATA_DIR
    / "scheduler_data"
    / "train_scheduler.csv"
)


def prepare_workflow_datasets():
    """
    Create the Event Watcher and Scheduler training datasets
    from train.csv if they do not already exist.

    The files are intentionally created only once.

    This prevents a later run of split_and_combine.py from
    overwriting datasets that have already been updated by
    the MLOps workflows.
    """

    print("\n" + "=" * 60)
    print("PREPARING MLOPS WORKFLOW TRAINING DATASETS")
    print("=" * 60)

    # --------------------------------------------------------
    # Verify that the initial training dataset exists.
    # --------------------------------------------------------

    if not TRAINING_DATA.exists():

        raise FileNotFoundError(
            f"\nInitial training dataset not found:\n"
            f"{TRAINING_DATA}"
        )

    # --------------------------------------------------------
    # Create Event Watcher training dataset.
    #
    # Only create it if it does not already exist.
    # --------------------------------------------------------

    EVENT_WATCHER_DATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not EVENT_WATCHER_DATA.exists():

        shutil.copy2(
            TRAINING_DATA,
            EVENT_WATCHER_DATA,
        )

        print(
            "\nEvent Watcher training dataset created:"
        )

        print(
            f"  {EVENT_WATCHER_DATA}"
        )

        print(
            "  Source:"
        )

        print(
            f"  {TRAINING_DATA}"
        )

    else:

        print(
            "\nEvent Watcher training dataset already exists."
        )

        print(
            f"  {EVENT_WATCHER_DATA}"
        )

        print(
            "  Existing dataset was preserved."
        )

    # --------------------------------------------------------
    # Create Scheduler training dataset.
    #
    # Only create it if it does not already exist.
    # --------------------------------------------------------

    SCHEDULER_DATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SCHEDULER_DATA.exists():

        shutil.copy2(
            TRAINING_DATA,
            SCHEDULER_DATA,
        )

        print(
            "\nScheduler training dataset created:"
        )

        print(
            f"  {SCHEDULER_DATA}"
        )

        print(
            "  Source:"
        )

        print(
            f"  {TRAINING_DATA}"
        )

    else:

        print(
            "\nScheduler training dataset already exists."
        )

        print(
            f"  {SCHEDULER_DATA}"
        )

        print(
            "  Existing dataset was preserved."
        )


def prepare_incoming_batches():
    """
    Combine normal and stress data and divide the result
    into three incoming batches.
    """

    print("\n" + "=" * 60)
    print("DATA PREPARATION FOR EVENT WATCHER DEMONSTRATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load normal and stress datasets
    # --------------------------------------------------------

    normal_df = pd.read_csv(
        NORMAL_DATA
    )

    stress_df = pd.read_csv(
        STRESS_DATA
    )

    print(
        f"\nNormal dataset rows : "
        f"{len(normal_df)}"
    )

    print(
        f"Stress dataset rows : "
        f"{len(stress_df)}"
    )

    # --------------------------------------------------------
    # Step 2: Remove duplicate customer IDs
    #
    # This ensures that the same customer is not repeated
    # inside either source dataset.
    # --------------------------------------------------------

    normal_df = (
        normal_df
        .drop_duplicates(
            subset=["customer_id"]
        )
        .reset_index(drop=True)
    )

    stress_df = (
        stress_df
        .drop_duplicates(
            subset=["customer_id"]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nNormal rows after duplicate removal : "
        f"{len(normal_df)}"
    )

    print(
        f"Stress rows after duplicate removal : "
        f"{len(stress_df)}"
    )

    # --------------------------------------------------------
    # Step 3: Combine normal + stress data
    #
    # Normal records are intentionally placed first.
    # Stress records are placed after the normal records.
    #
    # This makes it possible for the later incoming batch
    # to contain the stress data and demonstrate drift.
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            normal_df,
            stress_df,
        ],
        ignore_index=True,
    )

    print(
        f"\nCombined dataset rows: "
        f"{len(combined_df)}"
    )

    # --------------------------------------------------------
    # Step 4: Divide the combined dataset into 3 batches
    #
    # The first two batches contain the earlier portion of
    # the combined dataset.
    #
    # The final batch contains the later portion, which
    # includes the stress records.
    # --------------------------------------------------------

    total_rows = len(combined_df)

    batch_size = total_rows // 3

    batch_1 = combined_df.iloc[
        0:batch_size
    ].copy()

    batch_2 = combined_df.iloc[
        batch_size:batch_size * 2
    ].copy()

    # The final batch takes all remaining rows.
    # This also handles cases where the total number
    # of rows is not perfectly divisible by 3.

    batch_3 = combined_df.iloc[
        batch_size * 2:
    ].copy()

    # --------------------------------------------------------
    # Step 5: Create incoming batch directory
    # --------------------------------------------------------

    INCOMING_BATCH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Step 6: Save the three incoming batches
    # --------------------------------------------------------

    batch_1.to_csv(
        INCOMING_BATCH_DIR / "batch_1.csv",
        index=False,
    )

    batch_2.to_csv(
        INCOMING_BATCH_DIR / "batch_2.csv",
        index=False,
    )

    batch_3.to_csv(
        INCOMING_BATCH_DIR / "batch_3.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Step 7: Display preparation results
    # --------------------------------------------------------

    print(
        "\nDataset preparation completed!"
    )

    print(
        "\nIncoming batches:"
    )

    print(
        f"Batch 1: {batch_1.shape}"
    )

    print(
        f"Batch 2: {batch_2.shape}"
    )

    print(
        f"Batch 3: {batch_3.shape}"
    )

    print(
        "\nFiles saved:"
    )

    print(
        f"  {INCOMING_BATCH_DIR / 'batch_1.csv'}"
    )

    print(
        f"  {INCOMING_BATCH_DIR / 'batch_2.csv'}"
    )

    print(
        f"  {INCOMING_BATCH_DIR / 'batch_3.csv'}"
    )


def main():

    # --------------------------------------------------------
    # Step 1:
    # Create the Event Watcher and Scheduler datasets
    # from train.csv if they do not already exist.
    # --------------------------------------------------------

    prepare_workflow_datasets()

    # --------------------------------------------------------
    # Step 2:
    # Prepare the incoming batches.
    # --------------------------------------------------------

    prepare_incoming_batches()


if __name__ == "__main__":
    main()


# ============================================================
# SCRIPT SUMMARY
# ============================================================
#
# This script prepares the data required for the MLOps
# demonstrations.
#
# PART 1 - WORKFLOW TRAINING DATASETS
#
# The initial Git dataset:
#
#     data/train.csv
#
# is copied to:
#
#     data/event_watcher_data/train_trigger.csv
#     data/scheduler_data/train_scheduler.csv
#
# These two files are created only if they do not already exist.
#
# This is important because Event Watcher and Scheduler modify
# these datasets over time. Running this script again must not
# overwrite their updated datasets.
#
#
# PART 2 - INCOMING BATCHES
#
# The script combines:
#
#     data/current.csv
#     data/current_stress.csv
#
# Normal records are placed first and stress records are
# placed afterward.
#
# The combined data is divided into three batches:
#
#     Batch 1 -> earlier data
#     Batch 2 -> earlier data
#     Batch 3 -> later data containing the stress portion
#
# The batches are saved to:
#
#     data/incoming_batches/
#
#
# AFTER THIS SCRIPT
#
# Event Watcher:
#
#     train_trigger.csv
#          ↓
#     update dataset
#          ↓
#     DVC versioning
#          ↓
#     drift detection
#          ↓
#     retraining decision
#
# Scheduler:
#
#     train_scheduler.csv
#          ↓
#     update dataset
#          ↓
#     DVC versioning
#          ↓
#     drift detection
#          ↓
#     retraining decision
#
#
# IMPORTANT:
#
# - data/train.csv remains the initial Git dataset.
# - train_trigger.csv is created from train.csv when needed.
# - train_scheduler.csv is created from train.csv when needed.
# - Existing workflow datasets are never overwritten.
# ============================================================