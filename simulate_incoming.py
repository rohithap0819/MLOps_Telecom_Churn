
import time
import shutil
from pathlib import Path
from datetime import datetime

# Project directories
BASE_DIR = Path(__file__).resolve().parent
INCOMING_DIR = BASE_DIR / "data" / "incoming"

# Datasets to simulate
DATASETS = [
    ("current.csv", "normal"),
    ("current_stress.csv", "drift")
]

# Send a new dataset every 60 seconds
INTERVAL = 60

INCOMING_DIR.mkdir(parents=True, exist_ok=True)


def send_dataset(source_name, dataset_type):
    source_path = BASE_DIR / "data" / source_name

    if not source_path.exists():
        print(f"Dataset not found: {source_path}")
        return

    # Generate a unique filename so the watcher detects a new file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_name = f"{dataset_type}_{timestamp}.csv"
    destination_path = INCOMING_DIR / destination_name

    shutil.copy2(source_path, destination_path)

    print("\n" + "=" * 50)
    print(f"Dataset sent: {destination_name}")
    print(f"Dataset type: {dataset_type}")
    print(f"Time: {datetime.now()}")
    print("=" * 50)


def main():
    print("Incoming Data Simulator Started")
    print(f"Monitoring interval: {INTERVAL} seconds")
    print("Press CTRL+C to stop.")

    index = 0

    try:
        while True:
            source_name, dataset_type = DATASETS[index]

            send_dataset(source_name, dataset_type)

            # Alternate between normal and drift datasets
            index = (index + 1) % len(DATASETS)

            print(f"\nNext dataset will be sent in {INTERVAL} seconds.")
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nIncoming Data Simulator stopped.")


if __name__ == "__main__":
    main()
