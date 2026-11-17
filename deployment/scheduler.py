"""
Scheduler - Time-Based MLOps Workflow
=====================================

This script demonstrates a time-based MLOps retraining workflow.

The Scheduler:

1. Runs at a fixed interval of 1 minute.
2. Checks data/data_sources/incoming/ for new CSV files.
3. Processes one unprocessed CSV file per scheduled run.
4. Waits until the incoming CSV is completely copied.
5. Checks the incoming file against the current training data.
6. Checks for data drift against the current training data.
7. Appends only new customer records to the Scheduler training dataset.
8. Versions the updated dataset using DVC.
9. If drift is < 1%, no retraining action is required.
10. If drift is >= 1%, human approval is required for retraining.
11. The data version is preserved even when retraining is rejected.
12. If retraining is approved, candidate models are trained.
13. Production promotion remains a separate human approval step.
14. The Scheduler does not move to the next batch until the current batch workflow is complete.

Training dataset used by the Scheduler:

    data/data_sources/train_scheduler.csv

IMPORTANT:
   deployment/data_simulator_source.py creates the initial train_scheduler.csv.

The Scheduler does NOT create or copy the training dataset.

Incoming CSV files are simulated/placed into:

    data/data_sources/incoming/

For example, deployment/data_simulator_stream.py can place the prepared
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

from deployment.model_human_approval import (
    create_pending_approval,
    get_pending_approval,
)

from deployment.retraining import (
    run_candidate_training,
)

from deployment.config import (
    INCOMING_DATA_PATH,
)

from deployment.monitoring import (
    process_incoming_batch,
)

from deployment.data_simulator_version import (
    update_training_data,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# scheduler.py is located inside the deployment/ folder.
#
# parents[0] -> deployment
# parents[1] -> project root
#
# This allows the Scheduler to correctly locate
# project-level folders such as data/ and logs/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder where incoming CSV batches arrive.
INCOMING_DIR = PROJECT_ROOT / INCOMING_DATA_PATH

# Scheduler's mutable training dataset.
#
# This file is created initially by deployment/data_simulator_source.py.
# The Scheduler only updates it when new incoming data arrives.
TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "data_sources"
    / "scheduler_data"
    / "train_scheduler.csv"
)

DVC_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "data_sources"
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

# Stores which incoming file is currently waiting for a
# retraining approval. This lets the Scheduler recognize
# the same approval after the dashboard approves or rejects it.
SCHEDULER_APPROVAL_STATE_LOG = (
    PROJECT_ROOT
    / "logs"
    / "scheduler_approval_state.json"
)

# The approval module stores the latest approval request here.
# We read this file so the Scheduler can recognize completed
# decisions (approved/rejected), not only pending requests.
PENDING_APPROVAL_PATH = (
    PROJECT_ROOT
    / "logs"
    / "pending_approval.json"
)

# Complete approval history. This is the source of truth for
# completed approval decisions because pending_approval.json may
# be replaced or cleared after a decision is submitted.
APPROVAL_HISTORY_PATH = (
    PROJECT_ROOT
    / "logs"
    / "approval_requests.json"
)

# Stores the complete workflow state for the ONE active incoming batch.
# This prevents the Scheduler from moving to the next batch while
# retraining or model approval is still pending.
SCHEDULER_WORKFLOW_STATE_LOG = (
    PROJECT_ROOT
    / "logs"
    / "scheduler_workflow_state.json"
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
    It must already exist because deployment/data_simulator_source.py
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

    SCHEDULER_WORKFLOW_STATE_LOG.parent.mkdir(
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
# SCHEDULER APPROVAL STATE
# ============================================================

def load_scheduler_approval_state():
    """
    Load the Scheduler-specific record that links a retraining
    approval request to the incoming CSV file that caused it.

    This is needed because deployment/model_human_approval.py's get_pending_approval()
    returns only requests whose status is still "pending".
    After the dashboard rejects or approves a request, the
    Scheduler needs to know that the completed decision belongs
    to the same incoming file.
    """

    if not SCHEDULER_APPROVAL_STATE_LOG.exists():
        return None

    try:
        with open(
            SCHEDULER_APPROVAL_STATE_LOG,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        print(
            "\nWarning: Could not read Scheduler approval state."
        )
        return None


def save_scheduler_approval_state(
    incoming_file_name,
    request_id,
):
    """
    Save the link between an incoming CSV and its approval request.
    """

    SCHEDULER_APPROVAL_STATE_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "incoming_file": incoming_file_name,
        "request_id": request_id,
    }

    with open(
        SCHEDULER_APPROVAL_STATE_LOG,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=4,
        )


def clear_scheduler_approval_state():
    """
    Remove the Scheduler-specific approval state after the
    associated incoming file has completed its workflow.
    """

    if SCHEDULER_APPROVAL_STATE_LOG.exists():
        try:
            SCHEDULER_APPROVAL_STATE_LOG.unlink()
        except OSError:
            pass


def load_latest_approval_request():
    """
    Read the latest approval request regardless of whether it is
    pending, approved, or rejected.

    deployment/model_human_approval.py's get_pending_approval() intentionally returns
    only pending requests, so this helper is used when the
    Scheduler needs to recognize a completed dashboard decision.
    """

    if not PENDING_APPROVAL_PATH.exists():
        return None

    try:
        with open(
            PENDING_APPROVAL_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):
        print(
            "\nWarning: Could not read the latest approval request."
        )
        return None


def get_scheduler_approval_for_file(
    incoming_file_name,
):
    """
    Return the approval request associated with the supplied
    incoming file, if one exists.

    Returns:
        dict or None
    """

    scheduler_state = load_scheduler_approval_state()

    if scheduler_state is None:
        return None

    if (
        scheduler_state.get("incoming_file")
        != incoming_file_name
    ):
        return None

    latest_request = load_latest_approval_request()

    if latest_request is None:
        return None

    if (
        latest_request.get("request_id")
        != scheduler_state.get("request_id")
    ):
        return None

    return latest_request


# ============================================================
# SCHEDULER WORKFLOW STATE
# ============================================================

def load_scheduler_workflow_state():
    """
    Load the persistent workflow state for the active incoming batch.

    The Scheduler processes batches sequentially. Therefore, when a
    batch is waiting for human retraining approval or model approval,
    the Scheduler must keep working on that same batch instead of
    selecting the next CSV.
    """

    if not SCHEDULER_WORKFLOW_STATE_LOG.exists():
        return None

    try:
        with open(
            SCHEDULER_WORKFLOW_STATE_LOG,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        print(
            "\nWarning: Could not read Scheduler workflow state."
        )
        return None


def save_scheduler_workflow_state(state):
    """
    Save the current batch workflow state.
    """

    SCHEDULER_WORKFLOW_STATE_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SCHEDULER_WORKFLOW_STATE_LOG,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(state, file, indent=4)


def clear_scheduler_workflow_state():
    """
    Remove the active batch workflow state after the batch is complete.
    """

    if SCHEDULER_WORKFLOW_STATE_LOG.exists():
        try:
            SCHEDULER_WORKFLOW_STATE_LOG.unlink()
        except OSError:
            pass


def create_scheduler_workflow_state(
    incoming_file_name,
    drift_percentage,
    dataset_version=None,
    retraining_request_id=None,
    stage="retraining_pending",
):
    """
    Create the persistent state for one incoming batch.

    Stages used by the Scheduler:

        retraining_pending
        candidate_training
        model_approval_pending
        completed
    """

    state = {
        "incoming_file": incoming_file_name,
        "stage": stage,
        "drift_percentage": float(drift_percentage),
        "dataset_version": dataset_version,
        "retraining_request_id": retraining_request_id,
        "model_approval_request_id": None,
    }

    save_scheduler_workflow_state(state)
    return state


def update_scheduler_workflow_state(**updates):
    """
    Update selected fields in the active workflow state.
    """

    state = load_scheduler_workflow_state()

    if state is None:
        return None

    state.update(updates)
    save_scheduler_workflow_state(state)
    return state


def load_approval_history():
    """
    Load the complete approval history.

    This file contains approval requests after they are approved
    or rejected, so it is used to recover a completed decision
    for the exact request ID stored in the Scheduler workflow.
    """

    if not APPROVAL_HISTORY_PATH.exists():
        return []

    try:
        with open(
            APPROVAL_HISTORY_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        # Normal expected format is a list of approval records.
        if isinstance(data, list):
            return data

        # Be tolerant if the history is wrapped in a dictionary.
        if isinstance(data, dict):
            for key in ("requests", "approval_requests", "history"):
                if isinstance(data.get(key), list):
                    return data[key]

        return []

    except (json.JSONDecodeError, OSError):
        print(
            "\nWarning: Could not read approval history."
        )
        return []


def get_latest_approval_for_request(request_id):
    """
    Find the approval decision for one exact request ID.

    Lookup order:

    1. Check pending_approval.json.
    2. If it does not contain this request ID, search the complete
       approval_requests.json history.

    This prevents the Scheduler from becoming stuck when the
    dashboard/API has already completed the approval and the
    latest pending request no longer contains the Scheduler's ID.
    """

    # First check the latest approval file. This handles requests
    # that are still pending and also works when the completed
    # request remains there.
    request = load_latest_approval_request()

    if (
        request is not None
        and request.get("request_id") == request_id
    ):
        return request

    # If the latest pending request does not match, search the
    # permanent approval history using the exact Scheduler ID.
    matching_requests = [
        item
        for item in load_approval_history()
        if isinstance(item, dict)
        and item.get("request_id") == request_id
    ]

    if matching_requests:
        return matching_requests[-1]

    return None


def finish_current_batch(incoming_file_name):
    """
    Mark the active batch as complete and clear its workflow state.
    """

    mark_file_as_processed(incoming_file_name)
    clear_scheduler_approval_state()
    clear_scheduler_workflow_state()


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
                "data/data_sources/scheduler_data/train_scheduler.csv",
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

    This is useful when deployment/data_simulator_stream.py is copying
    a batch into data/data_sources/incoming/.
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
    Process one incoming CSV through the complete Scheduler workflow.

    Important design rule:

    The incoming data is ALWAYS appended to the Scheduler training
    dataset and versioned with DVC after drift is calculated.

    Human approval controls whether candidate retraining happens.
    It does NOT control whether the incoming data is preserved.

    Workflow:

    1. Wait for the incoming CSV to become stable.
    2. Detect whether this is a new or already-active workflow.
    3. Append new records to train_scheduler.csv.
    4. Create a DVC dataset version.
    5. Calculate drift against the current training data.
    6. If drift < 1%, complete the batch.
    7. If drift >= 1%, create ONE retraining approval request.
    8. If retraining is rejected, keep the data version and complete the batch.
    9. If retraining is approved, train candidate models.
   10. Wait for the separate model/production approval.
   11. Whether the model is approved or rejected, complete this batch.
   12. Only then allow the Scheduler to move to the next CSV.
    """

    print("\n" + "=" * 60)
    print("PROCESSING INCOMING DATA")
    print("=" * 60)

    print(
        f"\nIncoming file: {incoming_file.name}"
    )

    # --------------------------------------------------------
    # Wait for the incoming CSV to finish copying.
    # --------------------------------------------------------

    if not wait_for_stable_file(incoming_file):
        print("\nFile is not ready.")
        print(
            "This file will be checked again during the next Scheduler run."
        )
        return

    # --------------------------------------------------------
    # Check whether this batch already has a persistent workflow.
    # --------------------------------------------------------

    workflow_state = load_scheduler_workflow_state()

    if workflow_state is not None:

        active_file = workflow_state.get("incoming_file")

        if active_file != incoming_file.name:
            print(
                "\nAnother Scheduler batch is currently active."
            )
            print(
                f"Active batch: {active_file}"
            )
            print(
                "The Scheduler will wait before processing another batch."
            )
            return

        return resume_scheduler_workflow(
            incoming_file,
            workflow_state,
        )

    # --------------------------------------------------------
    # New batch: verify the Scheduler training dataset exists.
    # --------------------------------------------------------

    if not TRAINING_DATA_PATH.exists():
        print("\nERROR: Scheduler training dataset not found.")
        print(f"\nExpected file:\n  {TRAINING_DATA_PATH}")
        print("\nRun deployment/data_simulator_source.py first.")
        return

    reference_df = pd.read_csv(TRAINING_DATA_PATH)

    print(
        f"\nCurrent training rows: {len(reference_df)}"
    )

    # ========================================================
    # STEP 1 — CHECK FOR NEW RECORDS
    # ========================================================

    print("\n" + "=" * 60)
    print("CHECKING INCOMING DATA")
    print("=" * 60)

    try:
        incoming_df = pd.read_csv(incoming_file)

        if list(incoming_df.columns) != list(reference_df.columns):
            print(
                "\nERROR: Incoming CSV schema does not match "
                "the Scheduler training dataset."
            )
            print(f"\nIncoming columns:\n  {list(incoming_df.columns)}")
            print(f"\nExpected columns:\n  {list(reference_df.columns)}")
            print("\nThe incoming file will NOT be marked as processed.")
            return

        if "customer_id" not in incoming_df.columns:
            print("\nERROR: customer_id column not found.")
            print(
                "The Scheduler cannot determine whether records are new."
            )
            return

        incoming_unique = incoming_df.drop_duplicates(
            subset=["customer_id"]
        )

        existing_ids = set(reference_df["customer_id"])

        new_records = incoming_unique[
            ~incoming_unique["customer_id"].isin(existing_ids)
        ]

        new_record_count = len(new_records)

        print(f"\nIncoming rows       : {len(incoming_df)}")
        print(f"Unique incoming IDs : {len(incoming_unique)}")
        print(f"New customer records: {new_record_count}")

    except Exception as exc:
        print(f"\nFailed to inspect incoming CSV: {exc}")
        print("The incoming file will NOT be marked as processed.")
        return

    # ========================================================
    # STEP 2 — NO NEW RECORDS
    # ========================================================

    if new_record_count == 0:
        print("\nNo new customer records were found.")
        print("No new dataset version is required.")
        finish_current_batch(incoming_file.name)
        return

    # ========================================================
    # STEP 3 — ALWAYS PRESERVE / VERSION THE INCOMING DATA
    # ========================================================

    # IMPORTANT:
    # Data preservation is independent of human retraining approval.
    # The incoming records are added to train_scheduler.csv for
    # lineage/history and immediately versioned with DVC.

    print("\n" + "=" * 60)
    print("UPDATING SCHEDULER TRAINING DATASET")
    print("=" * 60)

    try:
        update_result = update_training_data(
            str(incoming_file.relative_to(PROJECT_ROOT)),
            str(TRAINING_DATA_PATH.relative_to(PROJECT_ROOT)),
        )

    except Exception as exc:
        print(f"\nTraining dataset update failed: {exc}")
        print("The incoming file will NOT be marked as processed.")
        return

    print(
        f"\nNew customer records added: "
        f"{update_result['new_records']}"
    )
    print(
        f"Updated training dataset rows: "
        f"{update_result['updated_rows']}"
    )

    dvc_success = version_dataset_with_dvc()

    if not dvc_success:
        print("\nDVC versioning failed.")
        print(
            "The batch will remain unprocessed so the dataset version "
            "can be completed safely."
        )
        return

    # Read the version that was just recorded.
    history = load_dataset_version_history()
    dataset_version = None
    if history:
        dataset_version = history[-1].get("version")

    # ========================================================
    # STEP 4 — DRIFT DETECTION
    # ========================================================

# ========================================================

    print("\n" + "=" * 60)
    print("DRIFT DETECTION")
    print("=" * 60)

    try:
        drift_percentage = process_incoming_batch(
            incoming_file.relative_to(PROJECT_ROOT),
            reference_df,
        )
        drift_percentage = float(drift_percentage)

    except Exception as exc:
        print(f"\nDrift detection failed: {exc}")
        print("The training dataset was NOT updated.")
        print("The incoming file will NOT be marked as processed.")
        return

    print(f"\nDetected drift: {drift_percentage:.2f}%")


    # ========================================================
    # STEP 5 — DRIFT POLICY
    # ========================================================

    if drift_percentage < NO_ACTION_DRIFT_THRESHOLD:
        print("\nDrift is below 1%.")
        print("No retraining action is required.")
        print("The incoming data has still been preserved and versioned.")

        finish_current_batch(incoming_file.name)

        print("\n" + "=" * 60)
        print("SCHEDULER PROCESSING COMPLETED")
        print("=" * 60)
        print(f"\nProcessed file : {incoming_file.name}")
        print(f"Training data  : {TRAINING_DATA_PATH}")
        print(f"New records    : {update_result['new_records']}")
        print(f"Drift detected : {drift_percentage:.2f}%")
        print(f"Dataset version: {dataset_version}")
        print("Retraining     : Not required")
        return

    # ========================================================
    # STEP 6 — CREATE RETRAINING APPROVAL
    # ========================================================

    print("\nDrift is >= 1%.")
    print("Human approval is required before retraining.")

    training_data_relative = str(
        TRAINING_DATA_PATH.relative_to(PROJECT_ROOT)
    )

    new_request = create_pending_approval(
        trigger_source="Scheduler",
        approval_type="retraining",
        training_data_path=training_data_relative,
    )

    if new_request is None:
        print("\nUnable to create retraining approval request.")
        print("The batch will remain unprocessed.")
        return

    request_id = new_request.get("request_id")

    create_scheduler_workflow_state(
        incoming_file_name=incoming_file.name,
        drift_percentage=drift_percentage,
        dataset_version=dataset_version,
        retraining_request_id=request_id,
        stage="retraining_pending",
    )

    save_scheduler_approval_state(
        incoming_file.name,
        request_id,
    )

    print("\nRetraining approval is now pending.")
    print(
        "The incoming data has already been preserved in the "
        "versioned Scheduler dataset."
    )
    print(
        "The Scheduler will stay on this batch until the human decision "
        "is completed."
    )


# ============================================================
# RESUME AN ACTIVE BATCH WORKFLOW
# ============================================================

def resume_scheduler_workflow(incoming_file, workflow_state):
    """
    Continue an already-started batch workflow.

    This function is what prevents Batch N+1 from starting while
    Batch N is waiting for a human decision.
    """

    stage = workflow_state.get("stage")
    retraining_request_id = workflow_state.get(
        "retraining_request_id"
    )
    model_request_id = workflow_state.get(
        "model_approval_request_id"
    )

    print(
        f"\nActive workflow stage: {stage}"
    )
    print(
        "The Scheduler will continue this batch before selecting "
        "another incoming file."
    )

    # ========================================================
    # RETRAINING APPROVAL
    # ========================================================

    if stage == "retraining_pending":

        request = get_latest_approval_for_request(
            retraining_request_id
        )

        if request is None:
            print(
                "\nRetraining approval request could not be found."
            )
            print(
                "The Scheduler will keep this batch active."
            )
            return

        decision = request.get("decision")
        status = request.get("status")

        if status == "pending" or decision is None:
            print("\nRetraining approval is still pending.")
            print("No new approval request will be created.")
            print(
                "The Scheduler will wait for the human decision."
            )
            return

        # ----------------------------------------------------
        # Retraining rejected.
        # ----------------------------------------------------

        if decision == "reject_retraining":
            print("\nRetraining was rejected by the human reviewer.")
            print("No candidate models will be trained.")
            print(
                "The incoming data version is preserved and remains "
                "in train_scheduler.csv."
            )

            finish_current_batch(incoming_file.name)

            print("\nBatch completed.")
            print("The next incoming batch can now be processed.")
            return

        # ----------------------------------------------------
        # Retraining approved.
        # ----------------------------------------------------

        if decision != "approve_retraining":
            print(
                f"\nUnexpected retraining decision: {decision}"
            )
            print("The Scheduler will keep the batch active.")
            return

        print("\nRetraining approved by the human reviewer.")
        print(
            "The preserved/versioned Scheduler dataset will now be "
            "used for candidate training."
        )

        update_scheduler_workflow_state(
            stage="candidate_training"
        )
        stage = "candidate_training"

    # ========================================================
    # CANDIDATE TRAINING
    # ========================================================

    if stage == "candidate_training":

        # If a production approval already exists for this batch,
        # candidate training has already completed. Do not train again.
        existing_request = load_latest_approval_request()

        if (
            existing_request is not None
            and existing_request.get("approval_type") == "production"
            and existing_request.get("trigger_source") == "Scheduler"
            and existing_request.get("status") == "pending"
        ):
            model_request_id = existing_request.get("request_id")

            update_scheduler_workflow_state(
                stage="model_approval_pending",
                model_approval_request_id=model_request_id,
            )

            print(
                "\nCandidate training has already created a "
                "production/model approval request."
            )
            print(
                "The Scheduler will now wait for that human decision."
            )
            return

        print("\n" + "=" * 60)
        print("STARTING CANDIDATE MODEL TRAINING")
        print("=" * 60)

        try:
            training_result = run_candidate_training(
                TRAINING_DATA_PATH,
                trigger_source="Scheduler",
                retraining_approved=True,
            )

        except Exception as exc:
            print(
                f"\nCandidate training failed: {exc}"
            )
            print(
                "The batch remains active and will be retried "
                "during the next Scheduler run."
            )
            return

        # deployment/retraining.py creates a separate production
        # approval request after successful candidate training.
        latest_request = load_latest_approval_request()

        if (
            latest_request is not None
            and latest_request.get("approval_type") == "production"
            and latest_request.get("trigger_source") == "Scheduler"
        ):
            model_request_id = latest_request.get("request_id")

            update_scheduler_workflow_state(
                stage="model_approval_pending",
                model_approval_request_id=model_request_id,
            )

            print(
                "\nCandidate training completed."
            )

            if training_result is not None:
                print(
                    f"Training result: {training_result}"
                )

            print("\nModel/production approval is now pending.")
            print(
                "The Scheduler will remain on this batch until "
                "the human makes the final model decision."
            )
            return

        print(
            "\nCandidate training did not create the expected "
            "production approval request."
        )
        print(
            "The batch remains active and will be checked again."
        )
        return

    # ========================================================
    # MODEL / PRODUCTION APPROVAL
    # ========================================================

    if stage == "model_approval_pending":

        request = get_latest_approval_for_request(
            model_request_id
        )

        if request is None:
            print(
                "\nModel approval request could not be found."
            )
            print(
                "The Scheduler will keep this batch active."
            )
            return

        decision = request.get("decision")
        status = request.get("status")

        if status == "pending" or decision is None:
            print("\nModel/production approval is still pending.")
            print("No new model approval request will be created.")
            print(
                "The Scheduler will wait and keep this batch active."
            )
            return

        print(
            f"\nFinal model approval decision received: {decision}"
        )

        if decision == "reject_all":
            print("The candidate models were rejected.")
            print(
                "The dataset version remains preserved for lineage "
                "and future analysis."
            )
        else:
            print(
                "A candidate model was approved for production."
            )
            print(
                "The dataset version remains preserved as part of "
                "the batch lineage."
            )

        # IMPORTANT:
        # Whether the model is approved OR rejected, the batch is
        # now complete. Only after this point can the next incoming
        # CSV be selected.
        finish_current_batch(incoming_file.name)

        print("\n" + "=" * 60)
        print("SCHEDULER BATCH COMPLETED")
        print("=" * 60)
        print(f"\nProcessed file : {incoming_file.name}")
        print(
            f"Dataset version: "
            f"{workflow_state.get('dataset_version')}"
        )
        print(
            f"Drift detected : "
            f"{workflow_state.get('drift_percentage', 0):.2f}%"
        )
        print("The Scheduler can now move to the next batch.")
        return

    print(
        f"\nUnknown Scheduler workflow stage: {stage}"
    )
    print(
        "The Scheduler will not select another batch until the "
        "workflow state is resolved."
    )


# ============================================================
# SCHEDULED JOB
# ============================================================

def run_scheduler_job():
    """
    Execute one Scheduler cycle.

    The Scheduler processes batches sequentially. If a batch is
    already active and waiting for a human decision, that same
    batch is resumed instead of selecting a new CSV.
    """

    print("\n\n" + "#" * 60)
    print("SCHEDULER RUN")
    print(
        time.strftime("%Y-%m-%d %H:%M:%S")
    )
    print("#" * 60)

    # --------------------------------------------------------
    # IMPORTANT:
    # Always resume an active workflow first.
    #
    # This is the sequencing gate that prevents:
    # Batch 1 -> pending model approval -> Batch 2
    #
    # Instead:
    # Batch 1 -> pending model approval -> wait -> final decision
    # -> complete Batch 1 -> Batch 2
    # --------------------------------------------------------

    active_workflow = load_scheduler_workflow_state()

    if active_workflow is not None:
        active_file_name = active_workflow.get(
            "incoming_file"
        )

        active_file = (
            INCOMING_DIR / active_file_name
        )

        if not active_file.exists():
            print(
                "\nERROR: Active Scheduler batch file is missing."
            )
            print(
                f"Expected file: {active_file}"
            )
            print(
                "The Scheduler will not move to another batch "
                "while this workflow state exists."
            )
            return

        print(
            f"\nActive batch resumed: {active_file_name}"
        )

        process_incoming_file(
            active_file
        )

        return

    # --------------------------------------------------------
    # No active workflow exists.
    # Find the next unprocessed CSV.
    # --------------------------------------------------------

    incoming_file = find_next_incoming_file()

    if incoming_file is None:
        print("\nNo new incoming CSV files found.")
        return

    print(
        f"\nNext file selected: {incoming_file.name}"
    )

    # Process only one file during this Scheduler run.
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
    #     python deployment/data_simulator_source.py
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
            "  python deployment/data_simulator_source.py"
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
#       data/data_sources/incoming/
#
#   These files can be placed manually or simulated using:
#
#       deployment/data_simulator_stream.py
#
# Main dataset:
#
#       data/data_sources/train_scheduler.csv
#
#   This dataset is initially created by:
#
#       deployment/data_simulator_source.py
#
# Steps:
#
#   1. Run a scheduled check every 1 minute.
#
#   2. Find one unprocessed CSV from data/data_sources/incoming/.
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
#   - deployment/data_simulator_stream.py can simulate incoming batches.
#   - The initial train_scheduler.csv is created by
#     deployment/data_simulator_source.py
#   - Processed files are recorded in:
#
#       logs/scheduler_processed_files.json
#
#   - Scheduler approval-to-file state is recorded in:
#
#       logs/scheduler_approval_state.json
#
#     This allows the Scheduler to recognize the same approval
#     after the dashboard approves or rejects the request.
#
#   - Dataset version history is recorded in:
#
#       logs/dataset_version_history.json
