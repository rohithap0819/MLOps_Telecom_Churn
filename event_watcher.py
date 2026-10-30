import time
from pathlib import Path

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from config import INCOMING_DATA_PATH
from monitor import load_reference_data, process_incoming_batch
from candidate_training import run_candidate_training


class IncomingFileHandler(FileSystemEventHandler):

    def __init__(self):
        self.reference_df = load_reference_data()
        self.processed_files = set()

    def on_created(self, event):

        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if file_path.suffix.lower() != ".csv":
            return

        if file_path in self.processed_files:
            return

        print(f"\nNew incoming file detected: {file_path.name}")

        # Wait until the file has finished copying
        previous_size = -1

        for _ in range(10):
            time.sleep(1)

            if not file_path.exists():
                return

            current_size = file_path.stat().st_size

            if current_size == previous_size:
                break

            previous_size = current_size

        self.processed_files.add(file_path)

        try:
            retraining_required = process_incoming_batch(
                file_path,
                self.reference_df
            )

            if retraining_required:
                print("\nDrift threshold exceeded!")
                print("Starting candidate training...")

                run_candidate_training(file_path)

            else:
                print("\nDrift below threshold. Training skipped.")

            self.processed_files.add(file_path)

        except Exception as error:
            print(f"\nError processing incoming file: {error}")


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

    print("\nEvent-triggered monitoring started.")
    print(f"Watching folder: {incoming_folder}")
    print("Waiting for new CSV files... Press CTRL+C to stop.")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()
        print("\nEvent monitoring stopped.")

    observer.join()