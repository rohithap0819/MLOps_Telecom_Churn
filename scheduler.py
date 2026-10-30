import time
import schedule
from pathlib import Path

from candidate_training import run_candidate_training


PROJECT_ROOT = Path(__file__).resolve().parent

SCHEDULED_DATA_PATH = (
    PROJECT_ROOT / "data" / "stress_scheduled.csv"
)


def run_scheduled_training():
    """
    Execute candidate training at the scheduled time.
    """

    print("\n" + "=" * 55)
    print("SCHEDULED RETRAINING STARTED")
    print("=" * 55)

    if not SCHEDULED_DATA_PATH.exists():
        print("Scheduled dataset not found.")
        return

    print(f"Training dataset: {SCHEDULED_DATA_PATH}")

    run_candidate_training(SCHEDULED_DATA_PATH)

    print("\nScheduled retraining completed.")
    print("No production model has been promoted.")


# Schedule retraining every 1 minute (demonstration mode)2 AM
schedule.every(1).minutes.do(
    run_scheduled_training
)


if __name__ == "__main__":

    print("\nScheduled retraining service started.")
    print("Frequency: Every 1 minute (demonstration mode)")
    print(f"Dataset: {SCHEDULED_DATA_PATH}")
    print("Waiting for the scheduled execution...")

    while True:
        schedule.run_pending()
        time.sleep(30)