"""
Scheduler - Time-Based MLOps Workflow
=====================================

This script demonstrates a time-based MLOps retraining workflow.

The Scheduler:

1. Runs at a fixed interval of 1 minute.
2. Checks data/incoming/ for new CSV files.
3. Processes one unprocessed CSV file per scheduled run.
4. Waits until the incoming CSV is completely copied.
5. Checks the incoming file against the current training data.
6. Appends only new customer records to the Scheduler training dataset.
7. Versions the updated dataset using DVC.
8. Checks for data drift.
9. If drift is < 1%, no action is taken.
10. If drift is >= 1%, human approval is required for retraining.
11. If retraining is approved, candidate models are trained.
12. Production promotion remains a separate human approval step.

Training dataset used by the Scheduler:

    data/scheduler_data/train_scheduler.csv

IMPORTANT:
    split_and_combine.py creates the initial train_scheduler.csv.

The Scheduler does NOT create or copy the training dataset.

Incoming CSV files are simulated/placed into:

    data/incoming/

For example, simulate_incoming.py can place the prepared
incoming batches into this folder.

The Scheduler then detects those files during its next
scheduled run.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import schedule

from approval import create_pending_approval, get_pending_approval
from candidate_training import run_candidate_training
from config import INCOMING_DATA_PATH
from monitor import process_incoming_batch
from update_training_data import update_training_data


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Folder where incoming CSV batches arrive.
INCOMING_DIR = PROJECT_ROOT / INCOMING_DATA_PATH

# Scheduler's mutable training dataset.
#
# This file is created initially by split_and_combine.py.
# The Scheduler only updates it when new incoming data arrives.
TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "scheduler_data"
    / "train_scheduler.csv"
)

# DVC metadata file for the Scheduler training dataset.
DVC_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "scheduler_data"
    / "train_scheduler.csv.dvc"
)

# Stores incoming filenames that have already been processed.
PROCESSED_FILES_LOG = (
    PROJECT_ROOT
    / "logs"
    / "scheduler_processed_files.json"
)

# Stores dataset version history.
DATASET_VERSION_HISTORY_LOG = (
    PROJECT_ROOT
    / "logs"
    / "dataset_version_history.json"
)


# ============================================================
# SCHEDULER SETTINGS
# ============================================================

# Scheduler checks for new data every 1 minute.
SCHEDULE_INTERVAL_MINUTES = 1

# Drift below 1% requires no action.
NO_ACTION_DRIFT_THRESHOLD = 1.0

# Number of consecutive checks for file stability.
# This prevents the Scheduler from reading a partially
# copied CSV.
STABLE_FILE_CHECKS = 3

# Maximum time to wait for an incoming file to become stable.
STABLE_FILE_TIMEOUT = 60


# ============================================================
# DIRECTORY SETUP
# ============================================================

def ensure_directories():
    """
    Create directories required by the Scheduler.

    The training dataset itself is NOT created here.
    It must already exist because split_and_combine.py
    creates it.
    """

    # Create the incoming directory if required.
    INCOMING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Make sure the Scheduler dataset parent directory exists.
    TRAINING_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create the logs directory.
    PROCESSED_FILES_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PROCESSED FILE HISTORY
# ============================================================

def load_processed_files():
    """
    Load the list of incoming files that have already
    been processed.

    Returns:
        set: Processed file names.
    """

    if not PROCESSED_FILES_LOG.exists():
        return set()

    try:

        with open(
            PROCESSED_FILES_LOG,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return set(data)

    except (json.JSONDecodeError, OSError):

        print(
            "Warning: Could not read processed file history."
        )

        print(
            "Starting with an empty processing history."
        )

        return set()


def save_processed_files(processed_files):
    """
    Save processed file names to the history file.
    """

    PROCESSED_FILES_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PROCESSED_FILES_LOG,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            sorted(processed_files),
            file,
            indent=4,
        )


def mark_file_as_processed(file_name):
    """
    Add a file to the processed-file history.
    """

    processed_files = load_processed_files()

    processed_files.add(file_name)

    save_processed_files(processed_files)


# ============================================================
# DATASET VERSION HISTORY
# ============================================================

def load_dataset_version_history():
    """
    Load the Scheduler's dataset version history.

    Returns:
        list: Dataset version records.
    """

    if not DATASET_VERSION_HISTORY_LOG.exists():
        return []

    try:

        with open(
            DATASET_VERSION_HISTORY_LOG,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):

        print(
            "Warning: Could not read dataset version history."
        )

        return []


def save_dataset_version_history(history):
    """
    Save dataset version history.
    """

    DATASET_VERSION_HISTORY_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DATASET_VERSION_HISTORY_LOG,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )


# ============================================================
# DVC VERSIONING
# ============================================================

def version_dataset_with_dvc():
    """
    Track the updated Scheduler training dataset using DVC.

    Returns:
        bool:
            True  -> DVC completed successfully.
            False -> DVC failed.
    """

    print("\n" + "=" * 60)
    print("DVC DATASET VERSIONING")
    print("=" * 60)

    try:

        # Use the same Python environment that is
        # running the Scheduler.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dvc",
                "add",
                "data/scheduler_data/train_scheduler.csv",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:

            print("\nDVC versioning failed.")

            if result.stdout.strip():
                print(result.stdout)

            if result.stderr.strip():
                print(result.stderr)

            return False

        print(
            "\nDVC versioning completed successfully."
        )

        if result.stdout.strip():
            print(result.stdout)

        # Make sure the DVC metadata file was created.
        if not DVC_FILE_PATH.exists():

            print(
                "\nERROR: DVC command completed but "
                "the .dvc metadata file was not found."
            )

            return False

        # Load existing dataset version history.
        history = load_dataset_version_history()

        # Create a simple version number.
        version_number = len(history) + 1

        version_entry = {
            "version": f"V{version_number}",
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "dataset": str(
                TRAINING_DATA_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        }

        history.append(version_entry)

        save_dataset_version_history(history)

        print(
            f"\nDataset version created: "
            f"V{version_number}"
        )

        print(
            f"Dataset history: "
            f"{DATASET_VERSION_HISTORY_LOG}"
        )

        return True

    except Exception as exc:

        print(
            f"\nError while running DVC: {exc}"
        )

        return False


# ============================================================
# FIND NEXT INCOMING FILE
# ============================================================

def find_next_incoming_file():
    """
    Find the next unprocessed CSV file.

    The Scheduler processes only ONE file during
    each scheduled run.

    Returns:
        Path or None.
    """

    processed_files = load_processed_files()

    csv_files = sorted(
        INCOMING_DIR.glob("*.csv")
    )

    for file_path in csv_files:

        if file_path.name not in processed_files:

            return file_path

    return None


# ============================================================
# WAIT FOR FILE TO BECOME STABLE
# ============================================================

def wait_for_stable_file(
    file_path,
    stable_checks=STABLE_FILE_CHECKS,
    timeout=STABLE_FILE_TIMEOUT,
):
    """
    Wait until the incoming file has finished copying.

    The file is considered stable when its size and
    modification time remain unchanged for several
    consecutive checks.

    This is useful when simulate_incoming.py is copying
    a batch into data/incoming/.
    """

    print(
        "\nChecking whether the incoming file "
        "has finished copying..."
    )

    previous_signature = None
    consecutive_stable_checks = 0

    for _ in range(timeout):

        time.sleep(1)

        if not file_path.exists():

            print(
                "\nIncoming file disappeared "
                "while waiting."
            )

            return False

        stat = file_path.stat()

        current_signature = (
            stat.st_size,
            stat.st_mtime_ns,
        )

        if (
            current_signature
            == previous_signature
        ):

            consecutive_stable_checks += 1

        else:

            consecutive_stable_checks = 0

        previous_signature = current_signature

        if (
            consecutive_stable_checks
            >= stable_checks
        ):

            print(
                "Incoming file is stable and ready."
            )

            return True

    print(
        "\nIncoming file did not become stable "
        "within the timeout."
    )

    return False


# ============================================================
# PROCESS ONE INCOMING FILE
# ============================================================

def process_incoming_file(incoming_file):
    """
    Process one incoming CSV file.

    Workflow:

    1. Make sure the incoming file is completely copied.
    2. Load the current Scheduler training data.
    3. Append only new customer records.
    4. Version the updated dataset using DVC.
    5. Check data drift.
    6. If drift < 1%, take no action.
    7. If drift >= 1%, request human approval.
    8. If approved, train candidate models.
    9. Mark the incoming file as processed.
    """

    print("\n" + "=" * 60)
    print("PROCESSING INCOMING DATA")
    print("=" * 60)

    print(
        f"\nIncoming file: "
        f"{incoming_file.name}"
    )

    # --------------------------------------------------------
    # Wait for the incoming CSV to finish copying.
    # --------------------------------------------------------

    if not wait_for_stable_file(
        incoming_file
    ):

        print(
            "\nFile is not ready."
        )

        print(
            "This file will be checked again "
            "during a future Scheduler run."
        )

        return

    # --------------------------------------------------------
    # Make sure the Scheduler training dataset exists.
    # --------------------------------------------------------

    if not TRAINING_DATA_PATH.exists():

        print(
            "\nERROR: Scheduler training dataset not found."
        )

        print(
            f"\nExpected file:"
        )

        print(
            f"  {TRAINING_DATA_PATH}"
        )

        print(
            "\nRun split_and_combine.py first."
        )

        print(
            "The Scheduler does not create the "
            "initial training dataset."
        )

        return

    # --------------------------------------------------------
    # Load the current training data BEFORE adding the
    # incoming records.
    #
    # This dataset is used as the reference for drift.
    # --------------------------------------------------------

    reference_df = pd.read_csv(
        TRAINING_DATA_PATH
    )

    print(
        f"\nCurrent training rows: "
        f"{len(reference_df)}"
    )

    # ========================================================
    # STEP 1 — UPDATE TRAINING DATA
    # ========================================================

    print("\n" + "=" * 60)
    print("UPDATING TRAINING DATASET")
    print("=" * 60)

    # update_training_data() receives:
    #
    #   1. Incoming CSV path
    #   2. Scheduler training dataset path
    #
    # The Scheduler training dataset was already
    # created by split_and_combine.py.

    update_result = update_training_data(
        str(
            incoming_file.relative_to(
                PROJECT_ROOT
            )
        ),
        str(
            TRAINING_DATA_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
    )

    new_records = update_result[
        "new_records"
    ]

    print(
        f"\nNew customer records added: "
        f"{new_records}"
    )

    print(
        f"Updated training dataset rows: "
        f"{update_result['updated_rows']}"
    )

    # --------------------------------------------------------
    # If there are no new customer records:
    #
    # - Do not create a DVC version.
    # - Do not perform drift detection.
    # - Mark the file as processed.
    # --------------------------------------------------------

    if new_records == 0:

        print(
            "\nNo new customer records were added."
        )

        print(
            "Skipping DVC versioning "
            "and drift detection."
        )

        mark_file_as_processed(
            incoming_file.name
        )

        return

    # ========================================================
    # STEP 2 — DVC VERSIONING
    # ========================================================

    dvc_success = version_dataset_with_dvc()

    if not dvc_success:

        print(
            "\nDVC versioning failed."
        )

        print(
            "The incoming file will NOT be "
            "marked as processed."
        )

        return

    # ========================================================
    # STEP 3 — DRIFT DETECTION
    # ========================================================

    print("\n" + "=" * 60)
    print("DRIFT DETECTION")
    print("=" * 60)

    try:

        # process_incoming_batch() returns the
        # calculated drift percentage.
        #
        # Example:
        #   0.00
        #   15.79
        #   73.68

        drift_percentage = process_incoming_batch(
            incoming_file.relative_to(
                PROJECT_ROOT
            ),
            reference_df,
        )

        # Convert to float in case the monitor
        # returns a numeric type such as numpy.float64.
        drift_percentage = float(
            drift_percentage
        )

    except Exception as exc:

        print(
            f"\nDrift detection failed: {exc}"
        )

        print(
            "The incoming file will NOT be "
            "marked as processed."
        )

        return

    print(
        f"\nDetected drift: "
        f"{drift_percentage:.2f}%"
    )

    # ========================================================
    # STEP 4 — APPLY DRIFT POLICY
    # ========================================================

    if (
        drift_percentage
        < NO_ACTION_DRIFT_THRESHOLD
    ):

        # ----------------------------------------------------
        # CASE 1:
        # Drift is below 1%.
        # ----------------------------------------------------

        print(
            "\nDrift is below 1%."
        )

        print(
            "No retraining action is required."
        )

        print(
            "No human approval is required."
        )

        mark_file_as_processed(
            incoming_file.name
        )

        return

    # --------------------------------------------------------
    # CASE 2:
    # Drift is 1% or higher.
    #
    # Scheduler always requires human approval
    # before retraining.
    #
    # Unlike the Event Watcher, the Scheduler
    # does not automatically retrain when drift
    # becomes severe.
    # --------------------------------------------------------

    print(
        "\nDrift is >= 1%."
    )

    print(
        "Human approval is required "
        "before retraining."
    )

    # ========================================================
    # STEP 5 — RETRAINING APPROVAL
    # ========================================================

    print(
        "\nChecking retraining approval status..."
    )

    training_data_relative = str(
        TRAINING_DATA_PATH.relative_to(PROJECT_ROOT)
    )

    pending_request = get_pending_approval()

    # --------------------------------------------------------
    # Create a retraining approval request if one does not
    # already exist for this Scheduler workflow.
    # --------------------------------------------------------
    if pending_request is None:

        create_pending_approval(
            trigger_source="Scheduler",
            approval_type="retraining",
            training_data_path=training_data_relative,
        )

        print(
            "\nRetraining approval is now pending."
        )

        print(
            "Use the existing human approval interface to "
            "approve or reject retraining."
        )

        print(
            "The incoming file will remain unprocessed "
            "until a decision is made."
        )

        return

    # Make sure the existing pending request belongs to
    # this Scheduler retraining workflow.
    if (
        pending_request.get("approval_type") != "retraining"
        or pending_request.get("trigger_source") != "Scheduler"
        or pending_request.get("training_data_path")
        != training_data_relative
    ):

        print(
            "\nA different approval request is already pending."
        )

        print(
            "The Scheduler will wait until that request "
            "is completed."
        )

        return

    # --------------------------------------------------------
    # Human approval is still pending.
    # --------------------------------------------------------
    if pending_request.get("status") == "pending":

        print(
            "\nRetraining approval is still pending."
        )

        print(
            "The Scheduler will check again during the "
            "next scheduled run."
        )

        return

    # --------------------------------------------------------
    # Human rejected retraining.
    # --------------------------------------------------------
    if pending_request.get("decision") == "reject_retraining":

        print(
            "\nRetraining was rejected by the human reviewer."
        )

        print(
            "No candidate models will be trained."
        )

        mark_file_as_processed(
            incoming_file.name
        )

        return

    # --------------------------------------------------------
    # Human approved retraining.
    # --------------------------------------------------------
    if pending_request.get("decision") != "approve_retraining":

        print(
            "\nApproval request has an unexpected decision."
        )

        print(
            "The Scheduler will not start retraining."
        )

        return

    print(
        "\nRetraining approved by the human reviewer."
    )

    # ========================================================
    # STEP 6 — CANDIDATE TRAINING
    # ========================================================

    print("\n" + "=" * 60)
    print("STARTING CANDIDATE MODEL TRAINING")
    print("=" * 60)

    try:

        training_result = run_candidate_training(
            TRAINING_DATA_PATH,
            trigger_source="Scheduler",
        )

        print(
            "\nCandidate training completed."
        )

        if training_result is not None:

            print(
                f"Training result: "
                f"{training_result}"
            )

    except Exception as exc:

        print(
            f"\nCandidate training failed: "
            f"{exc}"
        )

        print(
            "The incoming file will NOT be "
            "marked as processed."
        )

        return

    # ========================================================
    # STEP 7 — COMPLETE
    # ========================================================

    mark_file_as_processed(
        incoming_file.name
    )

    print("\n" + "=" * 60)
    print("SCHEDULER PROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"\nProcessed file : "
        f"{incoming_file.name}"
    )

    print(
        f"Training data  : "
        f"{TRAINING_DATA_PATH}"
    )

    print(
        f"New records    : "
        f"{new_records}"
    )

    print(
        f"Drift detected : "
        f"{drift_percentage:.2f}%"
    )

    print(
        "Retraining     : Approved and completed"
    )


# ============================================================
# SCHEDULED JOB
# ============================================================

def run_scheduler_job():
    """
    Execute one Scheduler cycle.

    Only one unprocessed CSV file is handled
    during each scheduled run.
    """

    print("\n\n" + "#" * 60)

    print(
        "SCHEDULER RUN"
    )

    print(
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print("#" * 60)

    # Find the next unprocessed incoming CSV.
    incoming_file = find_next_incoming_file()

    if incoming_file is None:

        print(
            "\nNo new incoming CSV files found."
        )

        return

    print(
        f"\nNext file selected: "
        f"{incoming_file.name}"
    )

    # Process only one file during this run.
    process_incoming_file(
        incoming_file
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Start the Scheduler service.
    """

    print("\n" + "=" * 60)
    print("TELECOM CHURN MLOPS SCHEDULER")
    print("=" * 60)

    ensure_directories()

    # --------------------------------------------------------
    # Verify that the Scheduler training dataset exists.
    #
    # This dataset must have been created previously by:
    #
    #     python split_and_combine.py
    #
    # The Scheduler intentionally does not create it.
    # --------------------------------------------------------

    if not TRAINING_DATA_PATH.exists():

        print(
            "\nERROR: Scheduler training dataset "
            "does not exist."
        )

        print(
            f"\nExpected file:"
        )

        print(
            f"  {TRAINING_DATA_PATH}"
        )

        print(
            "\nRun:"
        )

        print(
            "  python split_and_combine.py"
        )

        print(
            "\nThen start scheduler.py again."
        )

        sys.exit(1)

    print(
        "\nTraining dataset:"
    )

    print(
        f"  {TRAINING_DATA_PATH}"
    )

    print(
        "\nIncoming directory:"
    )

    print(
        f"  {INCOMING_DIR}"
    )

    print(
        "\nScheduler interval:"
    )

    print(
        f"  Every "
        f"{SCHEDULE_INTERVAL_MINUTES} minute(s)"
    )

    print(
        "\nDrift policy:"
    )

    print(
        "  < 1%       -> No action"
    )

    print(
        "  >= 1%      -> Human retraining approval"
    )

    print(
        "  Production  -> Separate human approval"
    )

    print(
        "\nScheduler started."
    )

    print(
        "Waiting for incoming CSV files..."
    )

    # --------------------------------------------------------
    # Run one check immediately when Scheduler starts.
    # --------------------------------------------------------

    run_scheduler_job()

    # --------------------------------------------------------
    # Schedule future checks every 1 minute.
    # --------------------------------------------------------

    schedule.every(
        SCHEDULE_INTERVAL_MINUTES
    ).minutes.do(
        run_scheduler_job
    )

    # --------------------------------------------------------
    # Keep the Scheduler service running.
    # --------------------------------------------------------

    while True:

        schedule.run_pending()

        time.sleep(1)


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# SCRIPT SUMMARY
# ============================================================

# Purpose:
#   Demonstrates a time-based MLOps workflow for Telecom Churn.
#
# Input:
#   CSV files placed into:
#
#       data/incoming/
#
#   These files can be placed manually or simulated using:
#
#       simulate_incoming.py
#
# Main dataset:
#
#       data/scheduler_data/train_scheduler.csv
#
#   This dataset is initially created by:
#
#       split_and_combine.py
#
# Steps:
#
#   1. Run a scheduled check every 1 minute.
#
#   2. Find one unprocessed CSV from data/incoming/.
#
#   3. Wait until the CSV is completely copied.
#
#   4. Load the current Scheduler training dataset.
#
#   5. Append only new customer records.
#
#   6. Version the updated dataset with DVC.
#
#   7. Detect data drift.
#
#   8. If drift < 1%:
#        No action.
#
#   9. If drift >= 1%:
#        Create a pending human approval request.
#
#  10. The Scheduler waits for the human decision.
#
#  11. If retraining is approved:
#        Train Logistic Regression and Decision Tree
#        candidate models.
#
#  12. Production promotion remains a separate
#      human approval stage.
#
# Important:
#
#   - One incoming file is processed per scheduled run.
#   - The Scheduler does not generate incoming data.
#   - simulate_incoming.py can simulate incoming batches.
#   - The initial train_scheduler.csv is created by
#     split_and_combine.py.
#   - Processed files are recorded in:
#
#       logs/scheduler_processed_files.json
#
#   - Dataset version history is recorded in:
#
#       logs/dataset_version_history.json