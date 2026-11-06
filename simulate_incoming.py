"""
SCRIPT SUMMARY:
Simulates incoming customer data for the Event Watcher demonstration.

The script sends three prepared batches from data/incoming_batches
to data/incoming at 60-second intervals. Each batch is copied with
a unique timestamped filename so the Event Watcher detects it as
a new incoming file.
"""

import time
import shutil
from pathlib import Path
from datetime import datetime


# ----------------------------------------------------------------------
# PROJECT DIRECTORIES
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

BATCH_DIR = BASE_DIR / "data" / "incoming_batches"
INCOMING_DIR = BASE_DIR / "data" / "incoming"


# ----------------------------------------------------------------------
# SIMULATION SETTINGS
# ----------------------------------------------------------------------

BATCHES = [
    "batch_1.csv",
    "batch_2.csv",
    "batch_3.csv",
]

# Send one batch every 60 seconds
INTERVAL = 60


# Create incoming directory if it does not already exist
INCOMING_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# SEND ONE BATCH
# ----------------------------------------------------------------------

def send_batch(batch_name):
    """Copy one prepared batch into the incoming data directory."""

    source_path = BATCH_DIR / batch_name

    if not source_path.exists():
        print(f"Batch not found: {source_path}")
        return False

    # Create a unique filename so the Event Watcher detects a new file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_name = f"incoming_{timestamp}_{batch_name}"
    destination_path = INCOMING_DIR / destination_name

    shutil.copy2(source_path, destination_path)

    print("\n" + "=" * 55)
    print("INCOMING DATA SENT")
    print("=" * 55)
    print(f"Batch          : {batch_name}")
    print(f"Rows           : {len(open(source_path, encoding='utf-8').readlines()) - 1}")
    print(f"Destination    : {destination_name}")
    print(f"Time           : {datetime.now()}")
    print("=" * 55)

    return True


# ----------------------------------------------------------------------
# MAIN SIMULATION
# ----------------------------------------------------------------------

def main():

    print("=" * 55)
    print("INCOMING DATA SIMULATOR STARTED")
    print("=" * 55)
    print(f"Number of batches : {len(BATCHES)}")
    print(f"Interval          : {INTERVAL} seconds")
    print("Sequence          : Batch 1 -> Batch 2 -> Batch 3")
    print("Press CTRL+C to stop.")
    print("=" * 55)

    try:

        for index, batch_name in enumerate(BATCHES):

            send_batch(batch_name)

            # Wait only if another batch remains
            if index < len(BATCHES) - 1:

                print(
                    f"\nNext batch ({BATCHES[index + 1]}) "
                    f"will be sent in {INTERVAL} seconds..."
                )

                time.sleep(INTERVAL)

        print("\n" + "=" * 55)
        print("ALL INCOMING BATCHES SENT")
        print("=" * 55)

    except KeyboardInterrupt:

        print("\nIncoming Data Simulator stopped.")


# ----------------------------------------------------------------------
# PROGRAM ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# SCRIPT SUMMARY
# ----------------------------------------------------------------------
# This script simulates real-time customer data arrival.
#
# The data preparation step is handled separately by split_and_combine.py.
# That script creates three prepared batches of 581 rows each.
#
# This simulator only performs the stimulation step:
#
#     batch_1.csv
#          ↓
#     wait 60 seconds
#          ↓
#     batch_2.csv
#          ↓
#     wait 60 seconds
#          ↓
#     batch_3.csv
#
# Each batch is copied into data/incoming with a unique timestamped
# filename. This allows the Event Watcher to detect each batch as a
# new incoming dataset and perform its drift-check/retraining workflow.
# ----------------------------------------------------------------------