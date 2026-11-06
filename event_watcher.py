"""
SCRIPT SUMMARY:
Monitors the incoming data folder and manages the event-driven MLOps flow.

Workflow:
1. Detect a new incoming CSV batch.
2. Wait until the file is completely copied.
3. Append the batch to data/train.csv.
4. Update the DVC metadata for train.csv.
5. Record the dataset version and lineage.
6. Check the incoming batch for data drift.
7. Apply the drift policy:
   - Drift < 1%       -> No action.
   - Drift 1% to 50% -> Human approval required for retraining.
   - Drift > 50%     -> Automatic retraining.
8. After retraining, production promotion remains a separate
   human approval stage.
9. This script never directly promotes a model to production.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from config import INCOMING_DATA_PATH
from monitor import load_reference_data, process_incoming_batch
from update_training_data import update_training_data
from candidate_training import run_candidate_training


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TRAINING_DATA_PATH = (
    PROJECT_ROOT / "data" / "train.csv"
)

DVC_FILE_PATH = (
    PROJECT_ROOT / "data" / "train.csv.dvc"
)

DATASET_HISTORY_PATH = (
    PROJECT_ROOT / "logs" / "dataset_version_history.json"
)


# ============================================================
# DRIFT POLICY
# ============================================================

# Drift below 1% does not require any action.
NO_ACTION_DRIFT_THRESHOLD = 1.0

# Drift greater than 50% is considered severe drift.
# Severe drift automatically starts candidate retraining.
AUTO_RETRAIN_DRIFT_THRESHOLD = 50.0


# ============================================================
# DATASET VERSIONING
# ============================================================

def update_dvc_version(incoming_file, update_result):
    """
    Update DVC metadata after train.csv has been changed.

    DVC stores the content hash of train.csv inside
    data/train.csv.dvc. This function also records the
    dataset lineage in a JSON history file.
    """

    print("\nUpdating DVC dataset version...")

    try:

        # Run DVC using the installed command.
        result = subprocess.run(
            [
                "dvc",
                "add",
                "data/train.csv"
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout.strip():
            print(result.stdout.strip())

        if not DVC_FILE_PATH.exists():
            raise FileNotFoundError(
                "DVC metadata file was not created."
            )

        # Read the DVC metadata file.
        dvc_content = DVC_FILE_PATH.read_text(
            encoding="utf-8"
        )

        # Extract the MD5 content hash.
        dataset_hash = None

        for line in dvc_content.splitlines():

            stripped_line = line.strip()

            if stripped_line.startswith("- md5:"):
                dataset_hash = (
                    stripped_line.split(":", 1)[1].strip()
                )
                break

        # Load existing lineage history.
        DATASET_HISTORY_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if DATASET_HISTORY_PATH.exists():

            with open(
                DATASET_HISTORY_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                history = json.load(file)

        else:

            history = []

        # Dataset version number is based on the number
        # of recorded updates.
        version_number = len(history) + 1

        version_label = f"V{version_number}"

        version_record = {
            "version": version_label,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "dataset": "data/train.csv",
            "incoming_batch": str(
                Path(incoming_file).name
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
            "dvc_md5": dataset_hash
        }

        history.append(version_record)

        with open(
            DATASET_HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=4
            )

        print("\nDVC dataset version created.")
        print(f"Dataset version : {version_label}")
        print(
            f"Training rows    : "
            f"{update_result['updated_rows']}"
        )
        print(f"DVC MD5          : {dataset_hash}")
        print(
            f"Lineage recorded : "
            f"{DATASET_HISTORY_PATH}"
        )

        return version_record

    except subprocess.CalledProcessError as error:

        print("\nDVC update failed.")
        print(error.stderr)

        raise


# ============================================================
# EVENT HANDLER
# ============================================================

class IncomingFileHandler(FileSystemEventHandler):

    """
    Detects newly created and modified CSV files.

    Each distinct file version is processed once.
    Reusing or overwriting an existing filename is supported.
    """

    def __init__(self):

        # Load the initial reference dataset used for
        # drift comparison.
        self.reference_df = load_reference_data()

        # Store the last processed version of each file.
        self.processed_versions = {}

        # Prevent simultaneous processing of the same file.
        self.active_files = set()

        self.lock = Lock()

    def on_created(self, event):

        if not event.is_directory:
            self.handle_file(event.src_path)

    def on_modified(self, event):

        if not event.is_directory:
            self.handle_file(event.src_path)

    def on_moved(self, event):

        if not event.is_directory:
            self.handle_file(event.dest_path)

    def handle_file(self, file_path):

        file_path = Path(file_path).resolve()

        # Only process CSV files.
        if file_path.suffix.lower() != ".csv":
            return

        # Ignore files that are currently being processed.
        with self.lock:

            if file_path in self.active_files:
                return

            self.active_files.add(file_path)

        try:

            print(
                f"\nIncoming CSV detected: "
                f"{file_path.name}"
            )

            # ------------------------------------------------
            # Wait until the file has finished copying.
            # ------------------------------------------------

            stable_signature = (
                self.wait_for_stable_file(file_path)
            )

            if stable_signature is None:

                print(
                    "File did not become stable. "
                    "Skipping."
                )

                return

            # ------------------------------------------------
            # Avoid processing the same file version twice.
            # ------------------------------------------------

            with self.lock:

                if (
                    self.processed_versions.get(file_path)
                    == stable_signature
                ):

                    print(
                        "This file version was "
                        "already processed."
                    )

                    return

            print("File is ready for processing.")

            # =================================================
            # STEP 1 — APPEND INCOMING DATA TO TRAIN.CSV
            # =================================================

            print("\n" + "=" * 55)
            print("UPDATING TRAINING DATASET")
            print("=" * 55)

            update_result = update_training_data(
                file_path
            )

            print(
                f"\nNew records added: "
                f"{update_result['new_records']}"
            )

            print(
                f"Current training dataset size: "
                f"{update_result['updated_rows']} rows"
            )

            # If the incoming batch contains only duplicate
            # records, do not create a new DVC version.
            if update_result["new_records"] == 0:

                print(
                    "\nNo new records were added."
                )

                print(
                    "Skipping DVC version creation "
                    "for this duplicate batch."
                )

                return

            # =================================================
            # STEP 2 — UPDATE DVC
            # =================================================

            dataset_version = update_dvc_version(
                file_path,
                update_result
            )

            # =================================================
            # STEP 3 — CHECK DATA DRIFT
            # =================================================

            print("\n" + "=" * 55)
            print("RUNNING DATA DRIFT CHECK")
            print("=" * 55)

            # process_incoming_batch() must return the
            # calculated drift percentage.
            #
            # Example:
            #     0.00
            #     15.79
            #     73.68
            #
            # This value is then used by the drift policy below.

            drift_percentage = process_incoming_batch(
                file_path,
                self.reference_df
            )

            print(
                f"\nOverall drift: "
                f"{drift_percentage:.2f}%"
            )

            # =================================================
            # STEP 4 — APPLY DRIFT POLICY
            # =================================================

            if drift_percentage < NO_ACTION_DRIFT_THRESHOLD:

                # ------------------------------------------------
                # CASE 1:
                # Drift is less than 1%.
                # ------------------------------------------------

                print(
                    "\nDrift is below 1%."
                )

                print(
                    "No retraining required."
                )

                print(
                    "No human approval required."
                )

            elif drift_percentage <= AUTO_RETRAIN_DRIFT_THRESHOLD:

                # ------------------------------------------------
                # CASE 2:
                # Drift is between 1% and 50%.
                #
                # Human approval is required before retraining.
                # ------------------------------------------------

                print(
                    "\nDrift is between 1% and 50%."
                )

                print(
                    "Human approval is required "
                    "before retraining."
                )

                print(
                    "\nCreating retraining "
                    "approval request..."
                )

                # run_candidate_training() creates the
                # retraining approval request.
                #
                # Training will NOT start until the
                # human reviewer approves retraining.

                run_candidate_training(
                    TRAINING_DATA_PATH,
                    trigger_source="Event Watcher"
                )

            else:

                # ------------------------------------------------
                # CASE 3:
                # Drift is greater than 50%.
                #
                # Severe drift automatically starts retraining.
                # No retraining approval is required.
                # ------------------------------------------------

                print(
                    "\nSevere drift detected!"
                )

                print(
                    f"Drift = "
                    f"{drift_percentage:.2f}%"
                )

                print(
                    "Drift is greater than 50%."
                )

                print(
                    "Automatic retraining "
                    "will start."
                )

                # retraining_approved=True bypasses the
                # retraining approval gate.
                #
                # Candidate training starts immediately.
                #
                # IMPORTANT:
                # This does NOT promote a model to production.
                # Candidate training will still create a separate
                # production approval request.

                run_candidate_training(
                    TRAINING_DATA_PATH,
                    trigger_source="Event Watcher",
                    retraining_approved=True
                )

            # ------------------------------------------------
            # Mark this file version as successfully processed.
            # ------------------------------------------------

            with self.lock:

                self.processed_versions[
                    file_path
                ] = stable_signature

            print(
                f"\nSuccessfully processed: "
                f"{file_path.name}"
            )

            print(
                f"Dataset version created: "
                f"{dataset_version['version']}"
            )

        except Exception as error:

            print(
                f"\nError processing "
                f"{file_path.name}: {error}"
            )

        finally:

            with self.lock:

                self.active_files.discard(
                    file_path
                )

    def wait_for_stable_file(
        self,
        file_path,
        stable_checks=3,
        timeout=60
    ):
        """
        Wait until file size and modification time remain
        unchanged for several consecutive checks.
        """

        previous_signature = None
        consecutive_stable_checks = 0

        for _ in range(timeout):

            time.sleep(1)

            if not file_path.exists():
                return None

            stat = file_path.stat()

            current_signature = (
                stat.st_size,
                stat.st_mtime_ns
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

                return current_signature

        return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    incoming_folder = Path(
        INCOMING_DATA_PATH
    )

    incoming_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    event_handler = IncomingFileHandler()

    observer = PollingObserver(
        timeout=2
    )

    observer.schedule(
        event_handler,
        str(incoming_folder),
        recursive=False
    )

    observer.start()

    print(
        "\n=========================================="
    )

    print(
        "EVENT-TRIGGERED MONITORING STARTED"
    )

    print(
        "=========================================="
    )

    print(
        f"Watching folder: "
        f"{incoming_folder}"
    )

    print(
        "Detecting new and modified CSV files."
    )

    print(
        "Incoming batches will update "
        "train.csv and DVC."
    )

    print(
        "Drift policy:"
    )

    print(
        "  < 1%       -> No action"
    )

    print(
        "  1% - 50%   -> Human retraining approval"
    )

    print(
        "  > 50%      -> Automatic retraining"
    )

    print(
        "Production promotion always requires "
        "separate human approval."
    )

    print(
        "Waiting for incoming data..."
    )

    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        observer.stop()

        print(
            "\nEvent monitoring stopped."
        )

    observer.join()


# ==========================================
# SCRIPT SUMMARY
# ==========================================
# This script monitors the incoming data folder for CSV files.
#
# It waits for each file to finish copying before processing it.
#
# Each incoming batch is appended to data/train.csv.
#
# DVC metadata is updated after every successful dataset update.
#
# Dataset version and lineage information are recorded in logs.
#
# Data drift is checked after the batch arrives.
#
# Drift policy:
#   - Less than 1%:
#       No action.
#
#   - 1% to 50%:
#       Human approval is required before retraining.
#
#   - Greater than 50%:
#       Candidate retraining starts automatically.
#
# After candidate training, production promotion remains
# a separate human approval stage.
#
# This script never directly promotes a model to production.