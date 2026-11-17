import os
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from model_human_approval import (
    create_pending_approval,
    get_pending_approval
)


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "data_sources"
    / "train.csv"
)
CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates"
METRICS_PATH = PROJECT_ROOT / "logs" / "candidate_metrics.csv"

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "production_model.joblib"
)

PRODUCTION_METRICS_PATH = (
    PROJECT_ROOT
    / "logs"
    / "production_metrics.csv"
)

TARGET_COLUMN = "churn"
RANDOM_STATE = 42

CANDIDATE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

mlflow.set_tracking_uri(
    (PROJECT_ROOT / "mlruns").as_uri()
)

mlflow.set_experiment(
    "Telecom_Churn_Candidate_Training"
)


# --------------------------------------------------
# 2. Load and prepare the dataset
# --------------------------------------------------

def _run_candidate_training(
    training_data_path=DATA_PATH,
    trigger_source="Manual"
):

    print(
        "\nNo production model has been selected or promoted."
    )

    from data.validation import validate_csv

    df = validate_csv(
        training_data_path
    )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    data = df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    # Normalize churn labels
    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
        })
    )

    if data[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Unexpected values found in the churn target column."
        )

    X = data.drop(
        columns=[
            TARGET_COLUMN,
            "customer_id"
        ],
        errors="ignore"
    )

    y = data[TARGET_COLUMN].astype(int)

    print(
        f"Dataset shape: {data.shape}"
    )

    print(
        f"Feature count: {X.shape[1]}"
    )

    print(
        f"Churn rate: {y.mean():.2%}"
    )


    # --------------------------------------------------
    # 3. Identify numeric and categorical features
    # --------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    print(
        f"Numeric features: {len(numeric_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )


    # --------------------------------------------------
    # 4. Create preprocessing pipeline
    # --------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ]
    )


    # --------------------------------------------------
    # 5. Split data into training and testing sets
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\nTraining rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )


    # --------------------------------------------------
    # 5A. Evaluate existing production model
    # --------------------------------------------------

    if PRODUCTION_MODEL_PATH.exists():

        print(
            "\n" + "=" * 55
        )

        print(
            "EVALUATING EXISTING PRODUCTION MODEL"
        )

        print(
            "=" * 55
        )

        production_model = joblib.load(
            PRODUCTION_MODEL_PATH
        )

        production_predictions = (
            production_model.predict(X_test)
        )

        production_probabilities = (
            production_model.predict_proba(X_test)[:, 1]
        )

        production_metrics = {
            "accuracy": accuracy_score(
                y_test,
                production_predictions
            ),

            "precision": precision_score(
                y_test,
                production_predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                production_predictions,
                zero_division=0
            ),

            "f1_score": f1_score(
                y_test,
                production_predictions,
                zero_division=0
            ),

            "roc_auc": roc_auc_score(
                y_test,
                production_probabilities
            ),
        }

        production_results = pd.DataFrame([
            {
                "model_name":
                    "existing_production",
                **production_metrics,
            }
        ])

        production_results.to_csv(
            PRODUCTION_METRICS_PATH,
            index=False
        )

        print(
            "\nExisting production model results:"
        )

        for metric_name, value in (
            production_metrics.items()
        ):

            print(
                f"{metric_name}: {value:.4f}"
            )

        print(
            f"\nProduction metrics saved: "
            f"{PRODUCTION_METRICS_PATH}"
        )

    else:

        print(
            "\nNo existing production model found."
        )

        print(
            "Skipping production model comparison."
        )


    # --------------------------------------------------
    # 6. Define candidate models
    # --------------------------------------------------

    candidates = {

        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "decision_tree":
            DecisionTreeClassifier(
                max_depth=6,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
    }


    # --------------------------------------------------
    # 7. Train and evaluate both candidates
    # --------------------------------------------------

    results = []

    for model_name, estimator in (
        candidates.items()
    ):

        print(
            f"\n{'=' * 55}"
        )

        print(
            f"Training candidate: {model_name}"
        )

        print(
            "=" * 55
        )

        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    estimator
                ),
            ]
        )

        with mlflow.start_run(
            run_name=model_name
        ) as run:

            # Train
            model_pipeline.fit(
                X_train,
                y_train
            )

            # Predict
            predictions = (
                model_pipeline.predict(
                    X_test
                )
            )

            probabilities = (
                model_pipeline.predict_proba(
                    X_test
                )[:, 1]
            )

            # Evaluate
            metrics = {

                "accuracy":
                    accuracy_score(
                        y_test,
                        predictions
                    ),

                "precision":
                    precision_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                "f1_score":
                    f1_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                "roc_auc":
                    roc_auc_score(
                        y_test,
                        probabilities
                    ),
            }

            # Save candidate model
            model_path = (
                CANDIDATE_DIR
                / f"{model_name}.joblib"
            )

            joblib.dump(
                model_pipeline,
                model_path
            )

            # Log to MLflow
            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE
            )

            mlflow.log_param(
                "test_size",
                0.20
            )

            mlflow.log_metrics(
                metrics
            )

            mlflow.log_artifact(
                str(model_path),
                artifact_path="candidate_model"
            )

            result = {
                "model_name":
                    model_name,

                "mlflow_run_id":
                    run.info.run_id,

                "model_path":
                    str(
                        model_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                **metrics,
            }

            results.append(
                result
            )

            print(
                "\nEvaluation results:"
            )

            for metric_name, value in (
                metrics.items()
            ):

                print(
                    f"{metric_name}: "
                    f"{value:.4f}"
                )

            print(
                f"\nSaved candidate model: "
                f"{model_path}"
            )

            print(
                f"MLflow run ID: "
                f"{run.info.run_id}"
            )


    # --------------------------------------------------
    # 8. Save candidate comparison results
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        METRICS_PATH,
        index=False
    )

    print(
        "\n" + "=" * 55
    )

    print(
        "CANDIDATE TRAINING COMPLETED"
    )

    print(
        "=" * 55
    )

    print(
        "\nCandidate comparison:"
    )

    print(
        results_df[
            [
                "model_name",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nMetrics saved to: "
        f"{METRICS_PATH}"
    )

    print(
        f"Candidate models saved in: "
        f"{CANDIDATE_DIR}"
    )


    # --------------------------------------------------
    # 9. Create production approval request
    # --------------------------------------------------
    # Candidate models have been trained successfully.
    # A human must now choose which candidate, if any,
    # should be promoted to production.

    print(
        "\n" + "=" * 55
    )

    print(
        "PRODUCTION APPROVAL REQUIRED"
    )

    print(
        "=" * 55
    )

    print(
        "Candidate training completed successfully."
    )

    print(
        "Human approval is required before "
        "promoting a model."
    )

    create_pending_approval(
        trigger_source,
        approval_type="production",
        training_data_path=str(
            training_data_path
        )
    )

    print(
        "\nProduction approval request created."
    )

    print(
        "No model has been promoted to production yet."
    )

    return results_df


# --------------------------------------------------
# 10. Prevent simultaneous candidate training
# --------------------------------------------------

import time


TRAINING_LOCK_PATH = (
    PROJECT_ROOT
    / "logs"
    / "candidate_training.lock"
)


def run_candidate_training(
    training_data_path=DATA_PATH,
    trigger_source="Manual",
    retraining_approved=False
):

    lock_fd = None

    try:

        # --------------------------------------------------
        # Try to acquire the training lock
        # --------------------------------------------------

        while True:

            try:

                lock_fd = os.open(
                    str(
                        TRAINING_LOCK_PATH
                    ),
                    os.O_CREAT
                    | os.O_EXCL
                    | os.O_WRONLY
                )

                os.write(
                    lock_fd,
                    str(
                        os.getpid()
                    ).encode()
                )

                break

            except FileExistsError:

                try:

                    existing_pid = int(
                        TRAINING_LOCK_PATH
                        .read_text()
                        .strip()
                    )

                    if os.name == "nt":

                        import subprocess

                        result = subprocess.run(
                            [
                                "tasklist",
                                "/FI",
                                f"PID eq {existing_pid}",
                                "/NH"
                            ],
                            capture_output=True,
                            text=True
                        )

                        process_running = (
                            str(existing_pid)
                            in result.stdout
                        )

                    else:

                        try:

                            os.kill(
                                existing_pid,
                                0
                            )

                            process_running = True

                        except ProcessLookupError:

                            process_running = False

                    if process_running:

                        print(
                            "\nAnother candidate training "
                            "process is already running."
                        )

                        print(
                            "This training request "
                            "will be skipped."
                        )

                        return None

                    print(
                        "\nRemoving stale training lock..."
                    )

                    TRAINING_LOCK_PATH.unlink(
                        missing_ok=True
                    )

                except (
                    ValueError,
                    FileNotFoundError
                ):

                    print(
                        "\nRemoving invalid training lock..."
                    )

                    TRAINING_LOCK_PATH.unlink(
                        missing_ok=True
                    )

    except Exception:

        if lock_fd is not None:

            os.close(
                lock_fd
            )

            TRAINING_LOCK_PATH.unlink(
                missing_ok=True
            )

        raise


    try:

        print(
            "\nTraining lock acquired."
        )


        # ==================================================
        # Check for an existing pending approval request
        # ==================================================
        #
        # IMPORTANT:
        #
        # There are TWO different approval types:
        #
        # 1. retraining
        # 2. production
        #
        # For the automatic Event Watcher path
        # (>50% drift), retraining approval is NOT required.
        #
        # Therefore, an existing pending RETRAINING approval
        # must not block automatic retraining.
        #
        # However, a pending PRODUCTION approval means that
        # another candidate-training cycle has already completed
        # and is waiting for a human to select a production model.
        #
        # In that situation, we should not start another
        # candidate-training cycle.
        # ==================================================

        pending_request = (
            get_pending_approval()
        )

        if pending_request:

            pending_approval_type = (
                pending_request.get(
                    "approval_type",
                    "unknown"
                )
            )

            print(
                "\nA human approval request "
                "is already pending."
            )

            print(
                f"Approval type: "
                f"{pending_approval_type}"
            )

            print(
                f"Pending request ID: "
                f"{pending_request['request_id']}"
            )


            # --------------------------------------------------
            # AUTOMATIC RETRAINING PATH
            # --------------------------------------------------
            #
            # If Event Watcher detected >50% drift,
            # retraining_approved=True is passed.
            #
            # A previous/stale retraining request should NOT
            # prevent automatic retraining.
            # --------------------------------------------------

            if (
                retraining_approved
                and
                pending_approval_type
                == "retraining"
            ):

                print(
                    "\nAutomatic retraining trigger detected."
                )

                print(
                    "Existing retraining approval "
                    "will be bypassed."
                )

                print(
                    "Candidate training will start automatically."
                )

                # Continue to the training section below.


            # --------------------------------------------------
            # PRODUCTION APPROVAL IS DIFFERENT
            # --------------------------------------------------
            #
            # A production approval means candidate models
            # have already been trained and are waiting for
            # human selection.
            #
            # Do not start another training cycle.
            # --------------------------------------------------

            elif (
                pending_approval_type
                == "production"
            ):

                print(
                    "\nProduction approval is still pending."
                )

                print(
                    "Candidate training will be skipped."
                )

                print(
                    "Complete the production approval "
                    "before starting another training cycle."
                )

                return None


            # --------------------------------------------------
            # OTHER PENDING APPROVAL TYPES
            # --------------------------------------------------

            else:

                print(
                    "\nCandidate training will be skipped."
                )

                return None


        # ==================================================
        # Retraining approval gate
        # ==================================================
        #
        # This gate applies only when:
        #
        # retraining_approved=False
        #
        # This is the normal path for:
        #
        # 1% - 50% drift
        #
        # Event Watcher:
        #     run_candidate_training(...)
        #
        # Scheduler:
        #     run_candidate_training(...)
        #
        # Both require human retraining approval.
        #
        # The >50% Event Watcher path passes:
        #
        # retraining_approved=True
        #
        # and therefore skips this section.
        # ==================================================

        if not retraining_approved:

            print(
                "\nRETRAINING APPROVAL REQUIRED"
            )

            print(
                "Drift-triggered retraining "
                "requires human approval."
            )

            create_pending_approval(
                trigger_source,
                approval_type="retraining",
                training_data_path=str(
                    training_data_path
                )
            )

            print(
                "\nCandidate training has NOT started."
            )

            print(
                "Waiting for human approval to retrain."
            )

            return None


        # ==================================================
        # Candidate training approved
        # ==================================================
        #
        # This section is reached in TWO situations:
        #
        # 1. Human approved retraining
        #
        # 2. Event Watcher detected >50% drift
        #    and automatically bypassed retraining approval
        #
        # Production approval is still required after
        # candidate training.
        # ==================================================

        if retraining_approved:

            print(
                "\nRetraining approval condition satisfied."
            )

        else:

            print(
                "\nRetraining approved by human reviewer."
            )

        print(
            "Candidate training started."
        )

        print(
            f"Trigger source: "
            f"{trigger_source}"
        )


        # --------------------------------------------------
        # Train and evaluate candidate models
        # --------------------------------------------------

        results_df = _run_candidate_training(
            training_data_path,
            trigger_source
        )

        return results_df


    finally:

        if lock_fd is not None:

            os.close(
                lock_fd
            )

        TRAINING_LOCK_PATH.unlink(
            missing_ok=True
        )

        print(
            "\nTraining lock released."
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    run_candidate_training(
        trigger_source="Manual"
    )


# ============================================================
# SCRIPT SUMMARY
# ============================================================
#
# Purpose:
#     Train and evaluate Logistic Regression and Decision Tree
#     candidate models using the updated training dataset.
#
# Main steps:
#     1. Load and validate the training dataset.
#     2. Prepare numeric and categorical features.
#     3. Split data into training and testing sets.
#     4. Evaluate the existing production model, if available.
#     5. Train Logistic Regression and Decision Tree candidates.
#     6. Calculate accuracy, precision, recall, F1 and ROC-AUC.
#     7. Save candidate models and metrics.
#     8. Create a production approval request.
#
# Retraining approval:
#     1% - 50% drift:
#         Human retraining approval is required.
#
#     >50% Event Watcher drift:
#         Automatic retraining is allowed by passing
#         retraining_approved=True.
#
# Production approval:
#     Production promotion always requires a separate
#     human approval after candidate training.
#
# Inputs:
#     - data_sources/train.csv by default
#     - or another training CSV supplied by the workflow.
#
# Outputs:
#     - models/candidates/logistic_regression.joblib
#     - models/candidates/decision_tree.joblib
#     - logs/candidate_metrics.csv
#     - logs/production_metrics.csv
#     - production approval request
#
# Role in the MLOps workflow:
#
#     Drift
#        |
#        +---- 1%-50% ----> Retraining Approval
#        |                         |
#        |                         v
#        |                  Candidate Training
#        |
#        +---- >50% ------> Automatic Candidate Training
#                                  |
#                                  v
#                         Production Approval
#                                  |
#                                  v
#                         Production Promotion
#
# ============================================================