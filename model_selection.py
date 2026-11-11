"""
Model selection for Telecom Customer Churn.

This script trains and compares multiple candidate ML models using
Stratified 5-Fold Cross-Validation and tracks the experiments in MLflow.

Run:
    python model_selection.py

Purpose:
    1. Load and validate the historical training dataset.
    2. Define candidate models.
    3. Evaluate candidates using Stratified 5-Fold Cross-Validation.
    4. Compare model performance using F1-score and other metrics.
    5. Select the best candidate based on mean F1-score.
    6. Train the selected candidate on the complete dataset.
    7. Save candidate models locally.
    8. Track parameters, metrics, and model artifacts in MLflow.

This script is part of the learner-facing ML/model-selection stage.
Human approval and automated retraining are handled later by the
MLOps workflow.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# Get the root directory of the project.
ROOT_DIR = Path(__file__).resolve().parent

# Location of the historical training dataset.
DATA_PATH = ROOT_DIR / "data" / "train.csv"

# Directory where candidate models will be saved.
CANDIDATE_DIR = ROOT_DIR / "models" / "candidates"

# File containing the model comparison results.
METRICS_PATH = ROOT_DIR / "logs" / "model_selection_metrics.csv"

# Target column.
TARGET = "churn"

# Random state ensures reproducible results.
RANDOM_STATE = 42

# Number of folds used for cross-validation.
CV_FOLDS = 5

# MLflow local tracking directory.
MLFLOW_DIR = ROOT_DIR / "mlruns"

# MLflow experiment name.
MLFLOW_EXPERIMENT = "Telecom_Churn_Model_Selection"


# Create required directories if they do not already exist.
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


# Configure MLflow to use the project's local mlruns directory.
mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())

# Create the experiment if it does not already exist.
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

def load_data():
    """
    Load and prepare the historical training dataset.

    Returns:
        X: Feature dataframe.
        y: Binary target series.
    """

    print("\nLoading training dataset...")

    # Load the CSV file.
    df = pd.read_csv(DATA_PATH)

    # Make sure the target column exists.
    if TARGET not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET}' is missing."
        )

    # Remove rows where the target is missing.
    df = df.dropna(subset=[TARGET]).copy()

    # Convert different possible churn labels into 0 and 1.
    y = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
            }
        )
    )

    # Stop if unexpected target values are found.
    if y.isna().any():
        raise ValueError(
            "Unexpected values found in the churn target column."
        )

    # Remove the target from the feature dataset.
    #
    # customer_id is also removed because it is an identifier,
    # not a useful predictive feature.
    X = df.drop(
        columns=[TARGET, "customer_id"],
        errors="ignore",
    )

    return X, y


# ============================================================
# 3. BUILD PREPROCESSING + MODEL PIPELINE
# ============================================================

def build_pipeline(estimator, numeric_cols, categorical_cols):
    """
    Build a complete preprocessing and ML model pipeline.

    Numeric columns:
        Missing values -> median imputation -> standardization

    Categorical columns:
        Missing values -> most frequent value -> one-hot encoding

    Keeping preprocessing inside the pipeline is important because
    cross-validation must fit preprocessing separately for each fold.
    """

    # Pipeline for numerical features.
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Pipeline for categorical features.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Apply the correct preprocessing to each feature type.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_cols,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            ),
        ]
    )

    # Combine preprocessing and the ML model.
    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


# ============================================================
# 4. MAIN MODEL-SELECTION WORKFLOW
# ============================================================

def main():

    print("=" * 70)
    print("MODEL SELECTION - TELECOM CUSTOMER CHURN")
    print("=" * 70)

    # --------------------------------------------------------
    # Stage 1: Load data
    # --------------------------------------------------------

    print("\nSTAGE 1 - DATA LOADING")

    X, y = load_data()

    print(f"Rows: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Churn rate: {y.mean():.1%}")

    # Identify numerical features.
    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identify categorical features.
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    # --------------------------------------------------------
    # Stage 2: Define candidate models
    # --------------------------------------------------------

    print("\nSTAGE 2 - CANDIDATE MODELS")

    # Define the models that we want to compare.
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    for name in candidates:
        print(f"- {name}")

    # --------------------------------------------------------
    # Stage 3: Cross-validation
    # --------------------------------------------------------

    print(
        f"\nSTAGE 3 - STRATIFIED {CV_FOLDS}-FOLD CROSS-VALIDATION"
    )

    print(
        "Preprocessing is fitted separately inside each fold."
    )

    # StratifiedKFold keeps the churn/non-churn distribution
    # approximately consistent across folds.
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Define the metrics that we want to calculate.
    scoring = {
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),

        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),

        "roc_auc": "roc_auc",
    }

    results = []

    # Evaluate every candidate model.
    for name, estimator in candidates.items():

        print(f"\nEvaluating: {name}")

        # Build the complete preprocessing + model pipeline.
        pipeline = build_pipeline(
            estimator=estimator,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        # Create a separate MLflow run for this candidate.
        with mlflow.start_run(
            run_name=name
        ) as run:

            print("Running cross-validation...")

            # Perform 5-fold cross-validation.
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )

            # Store the candidate name.
            row = {
                "model": name,
                "mlflow_run_id": run.info.run_id,
            }

            # Calculate mean and standard deviation
            # for every evaluation metric.
            for metric in scoring:

                values = scores[f"test_{metric}"]

                mean_value = values.mean()
                std_value = values.std()

                row[f"{metric}_mean"] = mean_value
                row[f"{metric}_std"] = std_value

            results.append(row)

            # ------------------------------------------------
            # Log parameters to MLflow
            # ------------------------------------------------

            mlflow.log_param(
                "model_name",
                name,
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE,
            )

            mlflow.log_param(
                "cv_folds",
                CV_FOLDS,
            )

            mlflow.log_param(
                "test_strategy",
                "StratifiedKFold",
            )

            mlflow.log_param(
                "numeric_features",
                len(numeric_cols),
            )

            mlflow.log_param(
                "categorical_features",
                len(categorical_cols),
            )

            # Log model-specific parameters.
            if name == "Logistic Regression":

                mlflow.log_param(
                    "max_iter",
                    1000,
                )

                mlflow.log_param(
                    "class_weight",
                    "balanced",
                )

            elif name == "Decision Tree":

                mlflow.log_param(
                    "max_depth",
                    6,
                )

                mlflow.log_param(
                    "class_weight",
                    "balanced",
                )

            # ------------------------------------------------
            # Log cross-validation metrics
            # ------------------------------------------------

            mlflow.log_metric(
                "f1_mean",
                row["f1_mean"],
            )

            mlflow.log_metric(
                "f1_std",
                row["f1_std"],
            )

            mlflow.log_metric(
                "precision_mean",
                row["precision_mean"],
            )

            mlflow.log_metric(
                "precision_std",
                row["precision_std"],
            )

            mlflow.log_metric(
                "recall_mean",
                row["recall_mean"],
            )

            mlflow.log_metric(
                "recall_std",
                row["recall_std"],
            )

            mlflow.log_metric(
                "roc_auc_mean",
                row["roc_auc_mean"],
            )

            mlflow.log_metric(
                "roc_auc_std",
                row["roc_auc_std"],
            )

            # Print results to the terminal.
            print(
                f"Mean F1-score : "
                f"{row['f1_mean']:.3f} "
                f"(+/- {row['f1_std']:.3f})"
            )

            print(
                f"Mean Precision: "
                f"{row['precision_mean']:.3f}"
            )

            print(
                f"Mean Recall   : "
                f"{row['recall_mean']:.3f}"
            )

            print(
                f"Mean ROC-AUC  : "
                f"{row['roc_auc_mean']:.3f}"
            )

            print(
                f"MLflow Run ID : "
                f"{run.info.run_id}"
            )

    # --------------------------------------------------------
    # Stage 4: Compare candidates
    # --------------------------------------------------------

    print("\nSTAGE 4 - MODEL COMPARISON")

    # Convert the results into a dataframe.
    results_df = pd.DataFrame(results)

    # Sort models by mean F1-score.
    results_df = results_df.sort_values(
        "f1_mean",
        ascending=False,
    ).reset_index(drop=True)

    # Save comparison results locally.
    results_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    # Display the comparison table.
    display_columns = [
        "model",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            formatters={
                column: "{:.3f}".format
                for column in results_df.select_dtypes(
                    include="number"
                ).columns
            },
        )
    )

    print(
        f"\nComparison metrics saved to: "
        f"{METRICS_PATH}"
    )

    # --------------------------------------------------------
    # Stage 5: Select the best candidate
    # --------------------------------------------------------

    print("\nSTAGE 5 - CANDIDATE SELECTION")

    # The model with the highest mean F1-score
    # becomes the selected candidate.
    winner = results_df.iloc[0]

    winner_name = winner["model"]

    print(
        f"Selected candidate by mean "
        f"cross-validation F1: {winner_name}"
    )

    print(
        f"Mean F1-score: "
        f"{winner['f1_mean']:.3f}"
    )

    # --------------------------------------------------------
    # Stage 6: Train selected candidate on full dataset
    # --------------------------------------------------------

    print(
        "\nSTAGE 6 - FINAL CANDIDATE TRAINING"
    )

    # Get the estimator corresponding to the winning model.
    selected_estimator = candidates[winner_name]

    # Build the final preprocessing + model pipeline.
    selected_pipeline = build_pipeline(
        estimator=selected_estimator,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    print(
        f"Training selected candidate: "
        f"{winner_name}"
    )

    # Train the selected candidate using
    # the complete available training dataset.
    selected_pipeline.fit(X, y)

    # Create a clean filename for the model.
    model_filename = (
        winner_name.lower()
        .replace(" ", "_")
    )

    selected_model_path = (
        CANDIDATE_DIR
        / f"{model_filename}.joblib"
    )

    # Save the trained candidate model.
    joblib.dump(
        selected_pipeline,
        selected_model_path,
    )

    print(
        f"Selected candidate saved to: "
        f"{selected_model_path}"
    )

    # --------------------------------------------------------
    # Stage 7: Log selected candidate artifact
    # --------------------------------------------------------

    print(
        "\nSTAGE 7 - MLflow SELECTED CANDIDATE"
    )

    # Create a separate MLflow run for the
    # final selected candidate.
    with mlflow.start_run(
        run_name="selected_candidate"
    ):

        # Log which model was selected.
        mlflow.log_param(
            "selected_model",
            winner_name,
        )

        # Log the selection criterion.
        mlflow.log_param(
            "selection_metric",
            "f1_mean",
        )

        # Log the selected model's CV metrics.
        mlflow.log_metric(
            "cv_f1_mean",
            winner["f1_mean"],
        )

        mlflow.log_metric(
            "cv_f1_std",
            winner["f1_std"],
        )

        mlflow.log_metric(
            "cv_precision_mean",
            winner["precision_mean"],
        )

        mlflow.log_metric(
            "cv_recall_mean",
            winner["recall_mean"],
        )

        mlflow.log_metric(
            "cv_roc_auc_mean",
            winner["roc_auc_mean"],
        )

        # Log the trained model as an MLflow artifact.
        mlflow.sklearn.log_model(
            selected_pipeline,
            artifact_path="selected_candidate_model",
        )

        # Also log the locally saved joblib model.
        mlflow.log_artifact(
            str(selected_model_path),
            artifact_path="candidate_model_joblib",
        )

        print(
            "Selected candidate logged to MLflow."
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL SELECTION COMPLETED")
    print("=" * 70)

    print(
        f"Selected model : {winner_name}"
    )

    print(
        f"Mean CV F1     : "
        f"{winner['f1_mean']:.3f}"
    )

    print(
        f"Mean CV Recall : "
        f"{winner['recall_mean']:.3f}"
    )

    print(
        f"Mean CV ROC-AUC: "
        f"{winner['roc_auc_mean']:.3f}"
    )

    print(
        f"\nCandidate model: "
        f"{selected_model_path}"
    )

    print(
        f"Metrics file   : "
        f"{METRICS_PATH}"
    )

    print(
        f"MLflow tracking: "
        f"{MLFLOW_DIR}"
    )

    print(
        "\nNext step:"
    )

    print(
        "Review the candidate experiments in MLflow "
        "before moving into the main MLOps workflow."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# FILE SUMMARY
# ============================================================
#
# Purpose:
#     Select the best ML candidate for telecom customer churn.
#
# Steps:
#     1. Load and prepare train.csv.
#     2. Identify numerical and categorical features.
#     3. Build preprocessing pipelines.
#     4. Define Logistic Regression and Decision Tree candidates.
#     5. Evaluate both models using Stratified 5-Fold CV.
#     6. Calculate F1, Precision, Recall and ROC-AUC.
#     7. Log candidate experiments to MLflow.
#     8. Compare candidates using mean F1-score.
#     9. Select the best candidate.
#    10. Train the selected candidate on the complete dataset.
#    11. Save the selected candidate as a joblib model.
#    12. Log the selected candidate and metrics to MLflow.
#
# Inputs:
#     data/train.csv
#
# Outputs:
#     models/candidates/<selected_model>.joblib
#     logs/model_selection_metrics.csv
#     MLflow experiment and model artifacts
#
# Important:
#     This script performs model selection for the learner-facing
#     ML stage. It does not perform automated drift-triggered
#     retraining or human approval. Those responsibilities belong
#     to the later MLOps workflow.
# ============================================================