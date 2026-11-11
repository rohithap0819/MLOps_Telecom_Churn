"""
SCRIPT SUMMARY:
Runs the time-based Scheduler for the MLOps Telecom Churn project.

The Scheduler:

1. Runs at a fixed interval.
2. Checks data/incoming/ for manually uploaded CSV files.
3. Processes one unprocessed CSV per scheduled run.
4. Checks the incoming data for new customer records.
5. Appends only new customer records to data/train.csv.
6. Creates a new DVC dataset version.
7. Checks data drift using the training data BEFORE the new batch.
8. Applies the Scheduler drift policy:
       < 1% drift   -> No action
       >= 1% drift  -> Human approval for retraining
9. If retraining is approved, candidate training is started.
10. Candidate training creates a separate production approval.
11. Records processed files so the same CSV is not processed again.

Important:
The Scheduler does NOT generate incoming data.
A user manually places a new CSV inside:

    data/incoming/

The Scheduler only detects and processes that file.
"""


# ============================================================
# IMPORTS
# ============================================================

import json
import time
import subprocess
import schedule

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from update_training_data import update_training_data
from monitor import process_incoming_batch
from candidate_training import run_candidate_training


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Folder where the user manually places new CSV files.
INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"

# Main training dataset.
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

# DVC metadata file.
DVC_FILE_PATH = PROJECT_ROOT / "data" / "train.csv.dvc"

# Shared DVC dataset version history.
HISTORY_FILE = (
    PROJECT_ROOT / "logs" / "dataset_version_history.json"
)

# Scheduler-specific processed-file history.
PROCESSED_FILES = (
    PROJECT_ROOT / "logs" / "scheduler_processed_files.json"
)


# ============================================================
# SCHEDULER SETTINGS
# ============================================================

# The Scheduler checks for manually uploaded files every minute.
SCHEDULE_INTERVAL_MINUTES = 1

# Scheduler drift policy:
#
# Drift < 1%
#     No retraining action.
#
# Drift >= 1%
#     Human approval is required before retraining.
#
# IMPORTANT:
# Unlike Event Watcher, the Scheduler does NOT automatically
# retrain when drift is greater than 50%.
NO_ACTION_DRIFT_THRESHOLD = 1.0


# Make sure the incoming directory exists.
INCOMING_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET VERSION HISTORY
# ============================================================

def load_dataset_version_history():
    """
    Load the shared DVC dataset version history.

    Event Watcher and Scheduler use the same history file so
    that dataset versions remain part of one continuous lineage.
    """

    if not HISTORY_FILE.exists():
        return []

    try:
        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        print("Warning: Could not read dataset version history.")
        return []


# ============================================================
# CHECK WHETHER A FILE WAS ALREADY PROCESSED
# ============================================================

def is_file_already_processed(incoming_file):
    """
    Check whether the Scheduler has already processed this CSV.

    This is separate from DVC history because a CSV containing
    zero new customer records does not create a DVC version.

    Therefore, even a zero-new-record file must still be remembered
    as processed.
    """

    if not PROCESSED_FILES.exists():
        return False

    try:
        with open(
            PROCESSED_FILES,
            "r",
            encoding="utf-8",
        ) as file:
            processed_files = json.load(file)

    except (json.JSONDecodeError, OSError):
        return False

    incoming_name = Path(incoming_file).name

    return incoming_name in processed_files


# ============================================================
# MARK FILE AS PROCESSED
# ============================================================

def mark_file_as_processed(incoming_file):
    """
    Add the incoming CSV filename to the Scheduler's
    processed-file history.

    This prevents the same file from being processed again
    during future Scheduler checks.
    """

    PROCESSED_FILES.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if PROCESSED_FILES.exists():

        try:
            with open(
                PROCESSED_FILES,
                "r",
                encoding="utf-8",
            ) as file:
                processed_files = json.load(file)

        except (json.JSONDecodeError, OSError):
            processed_files = []

    else:
        processed_files = []

    incoming_name = Path(incoming_file).name

    if incoming_name not in processed_files:
        processed_files.append(incoming_name)

    with open(
        PROCESSED_FILES,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            processed_files,
            file,
            indent=4,
        )


# ============================================================
# CREATE DVC DATASET VERSION
# ============================================================

def update_dvc_version(incoming_file, update_result):
    """
    Create a new DVC version after train.csv has been updated.

    Steps:
    1. Run dvc add on data/train.csv.
    2. Read the new MD5 hash.
    3. Create the next version number.
    4. Store the dataset lineage information.
    """

    print("\nCreating DVC dataset version...")

    # Use the project's Python environment to run DVC.
    #
    # This is intentionally:
    #
    #     python -m dvc
    #
    # instead of calling dvc.exe directly because the project's
    # DVC launcher may not always point to the current environment.
    result = subprocess.run(
        [
            "python",
            "-m",
            "dvc",
            "add",
            "data/train.csv",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print("DVC command failed.")
        print(result.stderr)

        raise RuntimeError(
            "Could not create DVC dataset version."
        )

    # --------------------------------------------------------
    # Read the DVC MD5 hash.
    # --------------------------------------------------------

    dvc_md5 = None

    if DVC_FILE_PATH.exists():

        with open(
            DVC_FILE_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                if "md5:" in line:
                    dvc_md5 = line.split(
                        ":",
                        1,
                    )[1].strip()

                    break

    if not dvc_md5:
        raise RuntimeError(
            "Could not find the MD5 hash in train.csv.dvc."
        )

    # --------------------------------------------------------
    # Load existing dataset history.
    # --------------------------------------------------------

    history = load_dataset_version_history()

    # Create the next dataset version.
    version_number = len(history) + 1
    version_label = f"V{version_number}"

    # --------------------------------------------------------
    # Create dataset lineage record.
    # --------------------------------------------------------

    version_record = {

        "version": version_label,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "dataset": "data/train.csv",

        "incoming_batch": str(
            Path(incoming_file).relative_to(
                PROJECT_ROOT
            )
        ),

        "previous_rows": update_result[
            "previous_rows"
        ],

        "incoming_rows": update_result[
            "incoming_rows"
        ],

        "new_records": update_result[
            "new_records"
        ],

        "updated_rows": update_result[
            "updated_rows"
        ],

        "dvc_md5": dvc_md5,
    }

    history.append(version_record)

    # --------------------------------------------------------
    # Save updated dataset history.
    # --------------------------------------------------------

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    print("\nDVC version created successfully.")

    print(
        f"Dataset version : {version_label}"
    )

    print(
        f"DVC MD5         : {dvc_md5}"
    )

    return version_record


# ============================================================
# PROCESS ONE MANUALLY UPLOADED CSV
# ============================================================

def process_incoming_file(incoming_file):
    """
    Process one manually uploaded incoming CSV.

    Workflow:

        Manual CSV upload
                ↓
        Read current training data
                ↓
        Append only new customer records
                ↓
        DVC version
                ↓
        Drift check
                ↓
        Scheduler drift policy
                ↓
        Human retraining approval
                ↓
        Candidate training
                ↓
        Human production approval
    """

    print("\n" + "=" * 60)
    print("SCHEDULED DATA PROCESSING STARTED")
    print("=" * 60)

    print(
        f"Incoming file: {incoming_file}"
    )

    # --------------------------------------------------------
    # STEP 1: Check whether this file was already processed.
    # --------------------------------------------------------

    if is_file_already_processed(
        incoming_file
    ):

        print(
            "\nThis CSV has already been processed."
        )

        print(
            "Skipping the file."
        )

        return

    # --------------------------------------------------------
    # STEP 2: Load the CURRENT training data.
    #
    # This must happen BEFORE appending the incoming batch.
    #
    # We need this dataset as the reference for drift detection.
    # --------------------------------------------------------

    print(
        "\nLoading current training dataset..."
    )

    reference_df = pd.read_csv(
        TRAINING_DATA_PATH
    )

    print(
        f"Reference training rows: "
        f"{len(reference_df)}"
    )

    # --------------------------------------------------------
    # STEP 3: Append only genuinely new records.
    # --------------------------------------------------------

    print(
        "\nUpdating training dataset..."
    )

    update_result = update_training_data(
        str(
            incoming_file.relative_to(
                PROJECT_ROOT
            )
        )
    )

    # --------------------------------------------------------
    # STEP 4: Handle files that contain no new records.
    # --------------------------------------------------------

    if update_result[
        "new_records"
    ] == 0:

        print(
            "\nNo new customer records were found."
        )

        print(
            "DVC version will not be created."
        )

        print(
            "This CSV has been marked as processed."
        )

        mark_file_as_processed(
            incoming_file
        )

        return

    # --------------------------------------------------------
    # STEP 5: Create a DVC dataset version.
    # --------------------------------------------------------

    update_dvc_version(
        incoming_file,
        update_result,
    )

    # --------------------------------------------------------
    # STEP 6: Check data drift.
    #
    # IMPORTANT:
    # reference_df is the training data BEFORE the incoming
    # batch was appended.
    #
    # This allows us to measure how different the incoming
    # data is from the existing training data.
    # --------------------------------------------------------

    print(
        "\nRunning data drift check..."
    )

    drift_result = process_incoming_batch(
        str(
            incoming_file.relative_to(
                PROJECT_ROOT
            )
        ),
        reference_df,
    )

    # process_incoming_batch() returns the drift percentage directly.
    # Convert it to float so the Scheduler can apply its drift policy.
    drift_percentage = float(drift_result)

    print(
        f"\nScheduler drift percentage: "
        f"{drift_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # STEP 7: Apply Scheduler drift policy.
    # --------------------------------------------------------

    if (
        drift_percentage
        < NO_ACTION_DRIFT_THRESHOLD
    ):

        print(
            "\nDrift is below 1%."
        )

        print(
            "No retraining action is required."
        )

        # Mark the file as processed so it will not
        # be picked up again on the next Scheduler run.
        mark_file_as_processed(
            incoming_file
        )

        print(
            "\nScheduled processing completed."
        )

        return

    # --------------------------------------------------------
    # STEP 8: Drift is >= 1%.
    #
    # Scheduler ALWAYS requires human approval.
    #
    # Even if drift is greater than 50%, Scheduler does
    # NOT automatically retrain.
    # --------------------------------------------------------

    print(
        "\nDrift is >= 1%."
    )

    print(
        "Human approval is required for retraining."
    )

    # This creates the retraining approval request.
    #
    # candidate_training.py will wait for the human
    # approval before actually training candidates.
    run_candidate_training(
        TRAINING_DATA_PATH,
        trigger_source="Scheduler",
    )

    # Mark the CSV as processed after the Scheduler
    # has successfully created the approval workflow.
    mark_file_as_processed(
        incoming_file
    )

    print(
        "\nRetraining approval request created."
    )

    print(
        "After retraining approval, candidate models "
        "will be evaluated."
    )

    print(
        "A separate production approval will then "
        "be required before promotion."
    )

    print(
        "\nScheduled processing completed."
    )


# ============================================================
# FIND NEXT UNPROCESSED CSV
# ============================================================

def find_next_incoming_file():
    """
    Find the oldest unprocessed CSV inside data/incoming/.

    The Scheduler does not create or simulate incoming data.
    It only looks for files manually placed in this directory.

    Files are sorted by filename so timestamped files are
    normally processed in upload/creation order.
    """

    incoming_files = sorted(
        INCOMING_DIR.glob("*.csv")
    )

    for file_path in incoming_files:

        if not is_file_already_processed(
            file_path
        ):

            return file_path

    return None


# ============================================================
# RUN ONE SCHEDULED CHECK
# ============================================================

def run_scheduler_job():
    """
    Perform one Scheduler check.

    Only one CSV is processed during each scheduled run.
    """

    print("\n" + "=" * 60)
    print("SCHEDULER CHECK")
    print("=" * 60)

    incoming_file = (
        find_next_incoming_file()
    )

    if incoming_file is None:

        print(
            "No new manually uploaded CSV files found."
        )

        return

    print(
        f"New incoming CSV found: "
        f"{incoming_file.name}"
    )

    process_incoming_file(
        incoming_file
    )


# ============================================================
# START SCHEDULER
# ============================================================

schedule.every(
    SCHEDULE_INTERVAL_MINUTES
).minutes.do(
    run_scheduler_job
)


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SCHEDULED DATA PROCESSING SERVICE STARTED")
    print("=" * 60)

    print(
        f"Frequency: Every "
        f"{SCHEDULE_INTERVAL_MINUTES} minute(s)"
    )

    print(
        f"Incoming directory: "
        f"{INCOMING_DIR}"
    )

    print(
        "\nManual workflow:"
    )

    print(
        "1. Place a new CSV inside data/incoming/"
    )

    print(
        "2. Scheduler detects the CSV."
    )

    print(
        "3. New customer records are appended."
    )

    print(
        "4. DVC version is created."
    )

    print(
        "5. Drift is checked."
    )

    print(
        "6. Drift >= 1% creates human retraining approval."
    )

    print(
        "7. Approved retraining creates production approval."
    )

    print(
        "\nWaiting for manually uploaded incoming data..."
    )

    print(
        "Press Ctrl+C to stop the Scheduler."
    )

    try:

        while True:

            schedule.run_pending()

            time.sleep(1)

    except KeyboardInterrupt:

        print(
            "\n\nScheduled data processing service stopped."
        )


# ============================================================
# SCRIPT SUMMARY
# ============================================================

# This script implements the time-based Scheduler for the
# Telecom Churn MLOps project.
#
# IMPORTANT:
# The Scheduler does NOT generate incoming data.
# The user manually places new CSV files inside:
#
#     data/incoming/
#
# The Scheduler periodically checks this directory and
# processes one unprocessed CSV at a time.
#
# Each incoming CSV:
#
#   1. Is compared against the existing customer IDs.
#   2. Only genuinely new customers are appended to train.csv.
#   3. A new DVC dataset version is created.
#   4. Data drift is calculated against the training data
#      BEFORE the incoming batch was added.
#
# Scheduler drift policy:
#
#   < 1%  -> No action.
#   >= 1% -> Human approval required for retraining.
#
# The Scheduler never automatically retrains based on high drift.
#
# If retraining is approved:
#
#   Candidate models
#          ↓
#   Human production approval
#          ↓
#   Model promotion
#
# Therefore, the Scheduler provides a time-based alternative
# to the Event Watcher while keeping the same controlled
# human-approval workflow.
# ============================================================
