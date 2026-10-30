
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
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


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates"
METRICS_PATH = PROJECT_ROOT / "logs" / "candidate_metrics.csv"

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "production" / "production_model.joblib"
)

PRODUCTION_METRICS_PATH = (
    PROJECT_ROOT / "logs" / "production_metrics.csv"
)

TARGET_COLUMN = "churn"
RANDOM_STATE = 42

CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
mlflow.set_experiment("Telecom_Churn_Candidate_Training")


# --------------------------------------------------
# 2. Load and prepare the dataset
# --------------------------------------------------

def run_candidate_training(training_data_path=DATA_PATH):
    print("\nNo production model has been selected or promoted.")

    from src.data.validate import validate_csv

    df = validate_csv(training_data_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    data = df.dropna(subset=[TARGET_COLUMN]).copy()

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
        raise ValueError("Unexpected values found in the churn target column.")

    X = data.drop(columns=[TARGET_COLUMN, "customer_id"], errors="ignore")
    y = data[TARGET_COLUMN].astype(int)

    print(f"Dataset shape: {data.shape}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Churn rate: {y.mean():.2%}")


    # --------------------------------------------------
    # 3. Identify numeric and categorical features
    # --------------------------------------------------

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")


    # --------------------------------------------------
    # 4. Create preprocessing pipeline
    # --------------------------------------------------

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
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
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

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")


        # --------------------------------------------------
    # 5A. Evaluate existing production model
    # --------------------------------------------------

    if PRODUCTION_MODEL_PATH.exists():

        print("\n" + "=" * 55)
        print("EVALUATING EXISTING PRODUCTION MODEL")
        print("=" * 55)

        production_model = joblib.load(PRODUCTION_MODEL_PATH)

        production_predictions = production_model.predict(X_test)
        production_probabilities = production_model.predict_proba(X_test)[:, 1]

        production_metrics = {
            "accuracy": accuracy_score(y_test, production_predictions),
            "precision": precision_score(
                y_test, production_predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, production_predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, production_predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, production_probabilities),
        }

        production_results = pd.DataFrame([{
            "model_name": "existing_production",
            **production_metrics,
        }])

        production_results.to_csv(
            PRODUCTION_METRICS_PATH,
            index=False
        )

        print("\nExisting production model results:")

        for metric_name, value in production_metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print(f"\nProduction metrics saved: {PRODUCTION_METRICS_PATH}")

    else:
        print("\nNo existing production model found.")
        print("Skipping production model comparison.")



    # --------------------------------------------------
    # 6. Define candidate models
    # --------------------------------------------------

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


    # --------------------------------------------------
    # 7. Train and evaluate both candidates
    # --------------------------------------------------

    results = []

    for model_name, estimator in candidates.items():

        print(f"\n{'=' * 55}")
        print(f"Training candidate: {model_name}")
        print("=" * 55)

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:

            # Train
            model_pipeline.fit(X_train, y_train)

            # Predict
            predictions = model_pipeline.predict(X_test)
            probabilities = model_pipeline.predict_proba(X_test)[:, 1]

            # Evaluate
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }

            # Save candidate model
            model_path = CANDIDATE_DIR / f"{model_name}.joblib"
            joblib.dump(model_pipeline, model_path)

            # Log to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", 0.20)

            mlflow.log_metrics(metrics)

            mlflow.log_artifact(str(model_path), artifact_path="candidate_model")

            result = {
                "model_name": model_name,
                "mlflow_run_id": run.info.run_id,
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
                **metrics,
            }

            results.append(result)

            print("\nEvaluation results:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            print(f"\nSaved candidate model: {model_path}")
            print(f"MLflow run ID: {run.info.run_id}")


    # --------------------------------------------------
    # 8. Save candidate comparison results
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(METRICS_PATH, index=False)

    print("\n" + "=" * 55)
    print("CANDIDATE TRAINING COMPLETED")
    print("=" * 55)

    print("\nCandidate comparison:")
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

    print(f"\nMetrics saved to: {METRICS_PATH}")
    print(f"Candidate models saved in: {CANDIDATE_DIR}")

    print("\nNo production model has been selected or promoted.")


if __name__ == "__main__":
    run_candidate_training()