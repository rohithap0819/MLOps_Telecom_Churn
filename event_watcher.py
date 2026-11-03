
import time
from pathlib import Path
from threading import Lock

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from config import INCOMING_DATA_PATH
from monitor import load_reference_data, process_incoming_batch
from candidate_training import run_candidate_training


class IncomingFileHandler(FileSystemEventHandler):
    """
    Detects newly created and modified CSV files.

    Each distinct file version is processed once.
    Reusing or overwriting an existing filename is supported.
    """

    def __init__(self):

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

        if file_path.suffix.lower() != ".csv":
            return

        # Ignore files that are currently being processed.
        with self.lock:

            if file_path in self.active_files:
                return

            self.active_files.add(file_path)

        try:

            print(f"\nIncoming CSV detected: {file_path.name}")

            # Wait until the file has finished copying.
            stable_signature = self.wait_for_stable_file(file_path)

            if stable_signature is None:
                print("File did not become stable. Skipping.")
                return

            # Avoid processing the same file version twice.
            with self.lock:

                if self.processed_versions.get(file_path) == stable_signature:
                    print("This file version was already processed.")
                    return

            print("File is ready for processing.")

            # Run the existing monitoring pipeline.
            retraining_required = process_incoming_batch(
                file_path,
                self.reference_df
            )

            if retraining_required:

                print("\nDrift threshold exceeded!")
                print("Starting candidate training...")

                run_candidate_training(
                    file_path,
                    trigger_source="Event Watcher"
                )

            else:

                print("\nDrift below threshold. Training skipped.")

            # Mark this version as processed only after monitoring succeeds.
            with self.lock:
                self.processed_versions[file_path] = stable_signature

            print(f"\nSuccessfully processed: {file_path.name}")

        except Exception as error:

            print(f"\nError processing {file_path.name}: {error}")

        finally:

            with self.lock:
                self.active_files.discard(file_path)

    def wait_for_stable_file(self, file_path, stable_checks=3, timeout=60):
        """
        Wait until file size and modification time remain unchanged
        for several consecutive checks.
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

            if current_signature == previous_signature:

                consecutive_stable_checks += 1

            else:

                consecutive_stable_checks = 0

            previous_signature = current_signature

            if consecutive_stable_checks >= stable_checks:

                return current_signature

        return None


if __name__ == "__main__":

    incoming_folder = Path(INCOMING_DATA_PATH)
    incoming_folder.mkdir(parents=True, exist_ok=True)

    event_handler = IncomingFileHandler()

    observer = PollingObserver(timeout=2)

    observer.schedule(
        event_handler,
        str(incoming_folder),
        recursive=False
    )

    observer.start()

    print("\n==========================================")
    print("EVENT-TRIGGERED MONITORING STARTED")
    print("==========================================")
    print(f"Watching folder: {incoming_folder}")
    print("Detecting new and modified CSV files.")
    print("Waiting for incoming data...")

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        observer.stop()
        print("\nEvent monitoring stopped.")

    observer.join()


# ==========================================
# SCRIPT SUMMARY
# ==========================================
# This script monitors the incoming data folder for CSV files.
# It detects new files, modified files and renamed files.
# It waits for file copying to finish before processing.
# Each distinct file version is monitored only once.
# Drift triggers candidate training and human approval.
# Normal batches update monitoring history without retraining.
