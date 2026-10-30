"""Main MLOps pipeline for telecom customer churn.

Run:
    python main.py
    python main.py --batch current
    python main.py --batch stress
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.data.validate import validate_data


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

TARGET = "churn"
RANDOM_STATE = 42

EXPECTED_COLUMNS = {
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "churn",
}


def load_dataset(filename, require_target=True):
    """Load a dataset and validate it using Pandera."""

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"\nLoading dataset: {filename}")

    df = pd.read_csv(path)

    print(f"Original dataset shape: {df.shape}")

    # Step 1: Validate the dataset using Pandera
    df = validate_data(df)

    # Step 2: Additional business-level checks
    if df["customer_id"].isna().any():
        raise ValueError(f"{filename} contains missing customer IDs.")

    if df["customer_id"].duplicated().any():
        raise ValueError(f"{filename} contains duplicate customer IDs.")

    if require_target and df[TARGET].isna().any():
        raise ValueError(f"{filename} contains missing churn labels.")

    return df


def normalize_target(series):
    """Convert common churn labels into numeric binary labels."""
    labels = series.astype(str).str.strip().str.lower().map(
        {
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
        }
    )

    if labels.isna().any():
        raise ValueError("Unexpected values found in the churn target.")

    return labels.astype(int)


def build_pipeline(model_name, numeric_cols, categorical_cols):
    """Build preprocessing and estimator as one reusable pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )

    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def run_drift_report(reference_df, current_df, batch_name):
    """Create a feature drift report, excluding IDs and target labels."""
    print(f"\nDRIFT CHECK - {batch_name.upper()} BATCH")

    reference_features = reference_df.drop(
        columns=[TARGET, "customer_id"], errors="ignore"
    )
    current_features = current_df.drop(
        columns=[TARGET, "customer_id"], errors="ignore"
    )

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        reference_data=reference_features,
        current_data=current_features,
    )

    report_path = LOGS_DIR / f"drift_report_{batch_name}.html"
    result.save_html(str(report_path))

    print(f"Reference rows: {len(reference_features):,}")
    print(f"Current rows: {len(current_features):,}")
    print(f"Drift report saved: {report_path.relative_to(ROOT_DIR)}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MLOps Telecom Churn Training and Monitoring Pipeline"
    )
    parser.add_argument(
        "--batch",
        choices=["current", "stress"],
        default="current",
        help="Current batch to evaluate for drift.",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "decision_tree"],
        default="logistic_regression",
        help="Candidate model to train.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("MLOPS TELECOM CHURN - MAIN PIPELINE")
    print("=" * 70)

    # Stage 1: Load data
    print("\nSTAGE 1 - DATA LOADING")
    train_df = load_dataset("train.csv")
    current_df = load_dataset(
        "current_stress.csv" if args.batch == "stress" else "current.csv"
    )

    print(f"Historical training data: {len(train_df):,} rows")
    print(f"Monitoring batch ({args.batch}): {len(current_df):,} rows")

    # Stage 2: Prepare training data
    print("\nSTAGE 2 - DATA PREPARATION")
    y = normalize_target(train_df[TARGET])
    X = train_df.drop(columns=[TARGET, "customer_id"])

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(f"Features: {X.shape[1]}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Training churn rate: {y.mean():.1%}")

    # Stage 3: Split data
    print("\nSTAGE 3 - TRAIN / HOLDOUT SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Training rows: {len(X_train):,}")
    print(f"Holdout rows: {len(X_test):,}")

    # Stage 4: Train and evaluate
    print("\nSTAGE 4 - MODEL TRAINING AND EVALUATION")
    pipeline = build_pipeline(args.model, numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():12}: {value:.3f}")

    # Stage 5: Track the experiment
    print("\nSTAGE 5 - MLFLOW EXPERIMENT TRACKING")
    mlflow.set_tracking_uri((ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment("telecom-churn-pipeline")

    with mlflow.start_run(run_name=f"main-{args.model}-{args.batch}"):
        mlflow.log_param("model", args.model)
        mlflow.log_param("batch", args.batch)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("holdout_fraction", 0.20)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=X_train.head(3),
        )

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    # Stage 6: Drift monitoring
    print("\nSTAGE 6 - FEATURE DRIFT MONITORING")
    reference_features = train_df.drop(
        columns=[TARGET, "customer_id"], errors="ignore"
    )
    current_features = current_df.drop(
        columns=[TARGET, "customer_id"], errors="ignore"
    )

    run_drift_report(reference_features, current_df, args.batch)

    print("\nSTAGE 7 - PIPELINE SUMMARY")
    print(f"Model evaluated: {args.model}")
    print(f"Monitoring batch: {args.batch}")
    print(f"Holdout F1-score: {metrics['f1']:.3f}")
    print("Training and monitoring run completed.")
    print("Human approval and scheduled retraining are separate workflow stages.")


if __name__ == "__main__":
    main()