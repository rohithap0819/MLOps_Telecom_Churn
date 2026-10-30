from candidate_training import run_candidate_training

import pandas as pd
from pathlib import Path
from datetime import datetime

from evidently import Report
from evidently.presets import DataDriftPreset

from src.data.validate import validate_csv
from config import (
    REFERENCE_DATA_PATH,
    DRIFT_THRESHOLD,
    DRIFT_REPORT_PATH,
    INCOMING_DATA_PATH
)

def load_reference_data():
    """
    Load and validate the reference training dataset.
    """

    print("\nLoading reference dataset...")

    reference_path = Path(REFERENCE_DATA_PATH)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {reference_path}"
        )

    reference_df = validate_csv(str(reference_path))

    print(f"Reference dataset loaded: {reference_df.shape}")

    return reference_df

def prepare_features(df):
    """
    Remove columns that should not be used for drift detection.
    """

    excluded_columns = ["customer_id", "churn"]

    features = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    return features

def detect_data_drift(reference_df, current_df, batch_name):
    """
    Compare incoming data against reference data using Evidently.
    """

    print("\nRunning Evidently drift detection...")

    reference_features = prepare_features(reference_df)
    current_features = prepare_features(current_df)

    report = Report(
        metrics=[DataDriftPreset()]
    )

    result = report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    # Extract drift statistics
    metric = next(iter(result.metric_results.values()))

    drifted_columns = metric.count.value
    drift_share = metric.share.value

    # Save HTML report
    report_directory = Path(DRIFT_REPORT_PATH)
    report_directory.mkdir(parents=True, exist_ok=True)

    report_path = report_directory / f"drift_report_{batch_name}.html"

    result.save_html(str(report_path))

    print("\nDrift Detection Results")
    print("-----------------------")
    print(f"Drifted features : {drifted_columns}")
    print(f"Drift percentage: {drift_share * 100:.2f}%")
    print(f"Threshold       : {DRIFT_THRESHOLD * 100:.2f}%")
    print(f"Report saved    : {report_path}")

    # Decide whether retraining should be triggered
    retraining_required = drift_share >= DRIFT_THRESHOLD

        # Save monitoring metrics for the Streamlit dashboard
    history_path = report_directory / "monitoring_history.csv"

    monitoring_record = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_name": batch_name,
        "total_features": len(reference_features.columns),
        "drifted_features": drifted_columns,
        "drift_percentage": round(drift_share * 100, 2),
        "drift_threshold": round(DRIFT_THRESHOLD * 100, 2),
        "retraining_required": retraining_required,
        "report_path": str(report_path)
    }])

    monitoring_record.to_csv(
        history_path,
        mode="a",
        header=not history_path.exists(),
        index=False
    )

    print(f"Monitoring history saved: {history_path}")

    print(f"Retraining required: {retraining_required}")

    return retraining_required


def process_incoming_batch(file_path, reference_df):
    """
    Validate and monitor a newly received data batch.
    """

    batch_path = Path(file_path)

    print("\n" + "=" * 50)
    print(f"Processing incoming batch: {batch_path.name}")
    print("=" * 50)

    if not batch_path.exists():
        raise FileNotFoundError(
            f"Incoming dataset not found: {batch_path}"
        )

    # Validate incoming dataset
    current_df = validate_csv(str(batch_path))

    # Generate a unique batch name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = f"{batch_path.stem}_{timestamp}"

    # Detect data drift
    retraining_required = detect_data_drift(
        reference_df,
        current_df,
        batch_name
    )

    return retraining_required


def run_monitoring():
    """
    Run monitoring for all CSV files in the incoming folder.
    """

    print("\nStarting MLOps monitoring workflow...")

    # Load reference dataset
    reference_df = load_reference_data()

    # Find incoming CSV files
    incoming_directory = Path(INCOMING_DATA_PATH)
    incoming_directory.mkdir(parents=True, exist_ok=True)

    incoming_files = list(incoming_directory.glob("*.csv"))

    if not incoming_files:
        print("\nNo incoming CSV files found.")
        return

    print(f"\nFound {len(incoming_files)} incoming batch(es).")

    # Process each incoming batch
    for file_path in incoming_files:

        retraining_required = process_incoming_batch(
            file_path,
            reference_df
        )

        if retraining_required:
            print("\nACTION: Drift threshold exceeded!")
            print("Starting automatic candidate training...")

            run_candidate_training(file_path)

        else:
            print("\nACTION: No retraining trigger detected.")
            print("Candidate training skipped.")

    print("\nMonitoring workflow completed.")


if __name__ == "__main__":
    run_monitoring()