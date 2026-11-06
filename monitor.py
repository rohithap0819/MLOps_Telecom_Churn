"""
SCRIPT SUMMARY:
Monitors incoming data for feature drift using Evidently.

Drift policy:
    - Drift < 1%:
        No action is taken.

    - Drift >= 1% and <= 50%:
        Human approval is required before retraining.

    - Drift > 50%:
        Candidate retraining starts automatically.

After candidate training, production promotion is still handled
through a separate human approval workflow.

The script also:
    - Validates incoming datasets.
    - Generates Evidently HTML reports.
    - Saves monitoring history for the Streamlit dashboard.
"""


import pandas as pd
from pathlib import Path
from datetime import datetime

from evidently import Report
from evidently.presets import DataDriftPreset
from candidate_training import run_candidate_training

from src.data.validate import validate_csv
from config import (
    REFERENCE_DATA_PATH,
    DRIFT_THRESHOLD,
    DRIFT_REPORT_PATH,
    INCOMING_DATA_PATH
)


# ============================================================
# DRIFT POLICY
# ============================================================

# Drift below 1% requires no action.
NO_ACTION_DRIFT_THRESHOLD = 1.0

# Drift greater than 50% automatically starts retraining.
AUTO_RETRAIN_DRIFT_THRESHOLD = 50.0


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

    reference_df = validate_csv(
        str(reference_path)
    )

    print(
        f"Reference dataset loaded: "
        f"{reference_df.shape}"
    )

    return reference_df


def prepare_features(df):
    """
    Remove columns that should not be used for drift detection.

    customer_id:
        Unique identifier and therefore not useful for
        detecting feature drift.

    churn:
        Target variable and should not be included in
        feature drift detection.
    """

    excluded_columns = [
        "customer_id",
        "churn"
    ]

    features = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    return features


def detect_data_drift(
    reference_df,
    current_df,
    batch_name
):
    """
    Compare incoming data against reference data
    using Evidently.

    Returns:
        float:
            Drift percentage.

    Example:
        0.00
        15.79
        73.68
    """

    print(
        "\nRunning Evidently drift detection..."
    )

    # Prepare reference and incoming datasets
    # for feature drift analysis.
    reference_features = prepare_features(
        reference_df
    )

    current_features = prepare_features(
        current_df
    )

    # Create the Evidently data drift report.
    report = Report(
        metrics=[
            DataDriftPreset(
                num_method="kl_div",
                cat_method="kl_div",
            )
        ]
    )

    # Compare reference data against
    # the current incoming batch.
    result = report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    # --------------------------------------------------------
    # Extract drift statistics
    # --------------------------------------------------------

    metric = next(
        iter(result.metric_results.values())
    )

    drifted_columns = metric.count.value

    # Evidently returns drift share as a decimal.
    #
    # Example:
    #     0.1579 -> 15.79%
    #
    drift_percentage = (
        metric.share.value * 100
    )

    # --------------------------------------------------------
    # Save Evidently HTML report
    # --------------------------------------------------------

    report_directory = Path(
        DRIFT_REPORT_PATH
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        report_directory
        / f"drift_report_{batch_name}.html"
    )

    result.save_html(
        str(report_path)
    )

    # --------------------------------------------------------
    # Print drift results
    # --------------------------------------------------------

    print(
        "\nDrift Detection Results"
    )

    print(
        "-----------------------"
    )

    print(
        f"Drifted features : "
        f"{drifted_columns}"
    )

    print(
        f"Drift percentage : "
        f"{drift_percentage:.2f}%"
    )

    print(
        f"Configured threshold : "
        f"{DRIFT_THRESHOLD * 100:.2f}%"
    )

    print(
        f"Report saved     : "
        f"{report_path}"
    )

    # --------------------------------------------------------
    # Determine whether any drift action is required.
    #
    # The Boolean is retained for monitoring history because
    # the Streamlit dashboard may already use this column.
    #
    # Any drift >= 1% is considered an action-triggering
    # drift condition.
    # --------------------------------------------------------

    retraining_required = (
        drift_percentage
        >= NO_ACTION_DRIFT_THRESHOLD
    )

    # --------------------------------------------------------
    # Save monitoring metrics for the Streamlit dashboard
    # --------------------------------------------------------

    history_path = (
        report_directory
        / "monitoring_history.csv"
    )

    monitoring_record = pd.DataFrame([
        {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "batch_name": batch_name,

            "total_features":
                len(
                    reference_features.columns
                ),

            "drifted_features":
                drifted_columns,

            "drift_percentage":
                round(
                    drift_percentage,
                    2
                ),

            "drift_threshold":
                round(
                    NO_ACTION_DRIFT_THRESHOLD,
                    2
                ),

            "retraining_required":
                retraining_required,

            "report_path":
                str(report_path)
        }
    ])

    monitoring_record.to_csv(
        history_path,
        mode="a",
        header=not history_path.exists(),
        index=False
    )

    print(
        f"Monitoring history saved: "
        f"{history_path}"
    )

    print(
        f"Retraining required: "
        f"{retraining_required}"
    )

    # IMPORTANT:
    # Return the actual percentage instead of True/False.
    #
    # event_watcher.py needs this value to distinguish:
    #
    #   < 1%       -> no action
    #   1% - 50%   -> human approval
    #   > 50%      -> automatic retraining

    return drift_percentage


def process_incoming_batch(
    file_path,
    reference_df
):
    """
    Validate and monitor a newly received
    data batch.

    Returns:
        float:
            Drift percentage.
    """

    batch_path = Path(
        file_path
    )

    print(
        "\n" + "=" * 50
    )

    print(
        f"Processing incoming batch: "
        f"{batch_path.name}"
    )

    print(
        "=" * 50
    )

    if not batch_path.exists():
        raise FileNotFoundError(
            f"Incoming dataset not found: "
            f"{batch_path}"
        )

    # --------------------------------------------------------
    # Validate incoming dataset
    # --------------------------------------------------------

    current_df = validate_csv(
        str(batch_path)
    )

    # --------------------------------------------------------
    # Generate a unique batch name
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    batch_name = (
        f"{batch_path.stem}_{timestamp}"
    )

    # --------------------------------------------------------
    # Detect data drift
    # --------------------------------------------------------

    drift_percentage = detect_data_drift(
        reference_df,
        current_df,
        batch_name
    )

    return drift_percentage


def run_monitoring():
    """
    Run monitoring for all CSV files
    in the incoming folder.

    This function is used for manual monitoring runs.
    """

    print(
        "\nStarting MLOps monitoring workflow..."
    )

    # --------------------------------------------------------
    # Load reference dataset
    # --------------------------------------------------------

    reference_df = load_reference_data()

    # --------------------------------------------------------
    # Find incoming CSV files
    # --------------------------------------------------------

    incoming_directory = Path(
        INCOMING_DATA_PATH
    )

    incoming_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    incoming_files = list(
        incoming_directory.glob("*.csv")
    )

    if not incoming_files:

        print(
            "\nNo incoming CSV files found."
        )

        return

    print(
        f"\nFound "
        f"{len(incoming_files)} "
        f"incoming batch(es)."
    )

    # --------------------------------------------------------
    # Process each incoming batch
    # --------------------------------------------------------

    for file_path in incoming_files:

        drift_percentage = (
            process_incoming_batch(
                file_path,
                reference_df
            )
        )

        # ====================================================
        # DRIFT POLICY
        # ====================================================

        if (
            drift_percentage
            < NO_ACTION_DRIFT_THRESHOLD
        ):

            # ------------------------------------------------
            # Drift < 1%
            # ------------------------------------------------

            print(
                "\nACTION: Drift is below 1%."
            )

            print(
                "No retraining required."
            )

            print(
                "Candidate training skipped."
            )

        elif (
            drift_percentage
            <= AUTO_RETRAIN_DRIFT_THRESHOLD
        ):

            # ------------------------------------------------
            # Drift >= 1% and <= 50%
            # ------------------------------------------------

            print(
                "\nACTION: Drift is between "
                "1% and 50%."
            )

            print(
                "Human approval is required "
                "before retraining."
            )

            # run_candidate_training() creates
            # the pending retraining approval.
            #
            # Training will begin only after
            # the human reviewer approves it.

            run_candidate_training(
                Path(
                    REFERENCE_DATA_PATH
                ),
                trigger_source="Manual Monitoring"
            )

        else:

            # ------------------------------------------------
            # Drift > 50%
            # ------------------------------------------------

            print(
                "\nACTION: Severe drift detected!"
            )

            print(
                f"Drift percentage: "
                f"{drift_percentage:.2f}%"
            )

            print(
                "Drift is greater than 50%."
            )

            print(
                "Starting automatic "
                "candidate training..."
            )

            # Severe drift bypasses the
            # retraining approval gate.
            #
            # Production promotion is still
            # controlled by a separate human approval.

            run_candidate_training(
                Path(
                    REFERENCE_DATA_PATH
                ),
                trigger_source="Manual Monitoring",
                retraining_approved=True
            )

    print(
        "\nMonitoring workflow completed."
    )


if __name__ == "__main__":
    run_monitoring()


# ============================================================
# SCRIPT SUMMARY
# ============================================================
# This script monitors incoming datasets for feature drift.
#
# Data validation:
#   Incoming CSV files are validated before drift analysis.
#
# Drift detection:
#   Evidently compares incoming features against the
#   reference training dataset.
#
# Drift policy:
#
#   Drift < 1%
#       -> No action.
#
#   Drift >= 1% and <= 50%
#       -> Human approval is required before retraining.
#
#   Drift > 50%
#       -> Candidate retraining starts automatically.
#
# Retraining:
#   Candidate training uses the reference training dataset
#   (data/train.csv).
#
# Production:
#   Candidate training does NOT automatically promote a model.
#   Production promotion remains a separate human approval.
#
# Monitoring:
#   Evidently HTML reports and monitoring_history.csv
#   are generated for dashboard visibility.
#
# Important:
#   process_incoming_batch() now returns the actual
#   drift percentage instead of a Boolean.