"""
SCRIPT SUMMARY:
Prepares three incoming batches for the Event Watcher demonstration.

The script:
1. Loads current.csv.
2. Loads current_stress.csv.
3. Combines both datasets into one dataset.
4. Removes duplicate customer IDs.
5. Places normal data first and stress data after it.
6. Divides the combined dataset into three equal batches.
7. Saves the three batches for the Event Watcher.

Because the stress data is kept together at the end of the combined
dataset, Batch 3 contains the stress records and can demonstrate
data drift and human retraining approval.
"""

from pathlib import Path

import pandas as pd


# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INCOMING_BATCH_DIR = DATA_DIR / "incoming_batches"

# Source datasets
NORMAL_DATA = DATA_DIR / "current.csv"
STRESS_DATA = DATA_DIR / "current_stress.csv"

# Scheduler dataset
SCHEDULED_DATA = DATA_DIR / "stress_scheduled.csv"


def main():
    print("=" * 60)
    print("DATA PREPARATION FOR MLOPS DEMONSTRATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load both datasets
    # ------------------------------------------------------------------
    normal_df = pd.read_csv(NORMAL_DATA)
    stress_df = pd.read_csv(STRESS_DATA)

    print(f"\nNormal dataset rows : {len(normal_df)}")
    print(f"Stress dataset rows : {len(stress_df)}")

    # ------------------------------------------------------------------
    # Step 2: Remove duplicate customer IDs within each dataset
    # ------------------------------------------------------------------
    normal_df = normal_df.drop_duplicates(
        subset=["customer_id"]
    ).reset_index(drop=True)

    stress_df = stress_df.drop_duplicates(
        subset=["customer_id"]
    ).reset_index(drop=True)

    print(f"Normal rows after duplicate removal : {len(normal_df)}")
    print(f"Stress rows after duplicate removal : {len(stress_df)}")

    # ------------------------------------------------------------------
    # Step 3: Combine normal + stress data
    #
    # Normal data is placed first and stress data is placed after it.
    # This prevents the stress records from being randomly distributed
    # across all three batches.
    # ------------------------------------------------------------------
    combined_df = pd.concat(
        [normal_df, stress_df],
        ignore_index=True,
    )

    print(f"\nCombined dataset rows: {len(combined_df)}")

    # ------------------------------------------------------------------
    # Step 4: Divide the combined dataset into 3 equal batches
    #
    # 943 + 800 = 1743 rows
    # 1743 / 3 = 581 rows per batch
    # ------------------------------------------------------------------
    batch_size = len(combined_df) // 3

    batch_1 = combined_df.iloc[
        0:batch_size
    ].copy()

    batch_2 = combined_df.iloc[
        batch_size:batch_size * 2
    ].copy()

    batch_3 = combined_df.iloc[
        batch_size * 2:batch_size * 3
    ].copy()

    # ------------------------------------------------------------------
    # Step 5: Create output directory
    # ------------------------------------------------------------------
    INCOMING_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 6: Save the three batches
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 7: Save stress data for Scheduler
    # ------------------------------------------------------------------
    stress_df.to_csv(
        SCHEDULED_DATA,
        index=False,
    )

    # ------------------------------------------------------------------
    # Step 8: Display results
    # ------------------------------------------------------------------
    print("\nDataset preparation completed!")

    print("\nIncoming batches:")
    print(f"Batch 1: {batch_1.shape}")
    print(f"Batch 2: {batch_2.shape}")
    print(f"Batch 3: {batch_3.shape}")

    print("\nFiles saved:")
    print(f"  {INCOMING_BATCH_DIR / 'batch_1.csv'}")
    print(f"  {INCOMING_BATCH_DIR / 'batch_2.csv'}")
    print(f"  {INCOMING_BATCH_DIR / 'batch_3.csv'}")
    print(f"  {SCHEDULED_DATA}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# SCRIPT SUMMARY
# ----------------------------------------------------------------------
# current.csv and current_stress.csv are combined first.
# The combined dataset is divided into three equal batches.
#
# With the current data:
#   Normal data  = 943 rows
#   Stress data  = 800 rows
#   Combined     = 1743 rows
#   Each batch   = 581 rows
#
# Normal data appears first and stress data appears afterward so that
# the later batch contains the stress portion instead of randomly
# spreading stress records across all three batches.
#
# The three batches are then used by the Event Watcher demonstration.
# ----------------------------------------------------------------------