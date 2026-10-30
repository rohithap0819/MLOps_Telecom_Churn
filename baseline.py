"""Baseline model evaluation for telecom customer churn.

Run:
    python baseline.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "train.csv"
TARGET = "churn"
RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("BASELINE MODEL EVALUATION - TELECOM CUSTOMER CHURN")
    print("=" * 70)

    # Stage 1: Load data
    print("\nSTAGE 1 - DATA LOADING")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {DATA_PATH.name}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Churn rate: {df[TARGET].astype(str).str.lower().isin(['yes', '1', 'true']).mean():.1%}")

    # Stage 2: Prepare features and target
    print("\nSTAGE 2 - DATA PREPARATION")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    df = df.dropna(subset=[TARGET]).copy()

    y = df[TARGET].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    )

    if y.isna().any():
        raise ValueError("Unexpected values found in the churn target column.")

    X = df.drop(columns=[TARGET, "customer_id"], errors="ignore")

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"Features used: {X.shape[1]}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Non-churn customers: {(y == 0).sum():,}")
    print(f"Churn customers: {(y == 1).sum():,}")

    # Stage 3: Split data
    print("\nSTAGE 3 - TRAIN / TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    # Stage 4: Build preprocessing pipeline
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

    # Stage 5: Train and evaluate both models
    print("\nSTAGE 4 - BASELINE MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    models = {
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

    results = []

    for model_name, estimator in models.items():
        print(f"\nMODEL: {model_name}")
        print("-" * 50)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        auc = roc_auc_score(y_test, probabilities)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
            }
        )

        print(f"Accuracy  : {accuracy:.3f}")
        print(f"Precision : {precision:.3f}")
        print(f"Recall    : {recall:.3f}")
        print(f"F1-score  : {f1:.3f}")
        print(f"ROC-AUC   : {auc:.3f}")

        print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
        print(confusion_matrix(y_test, predictions))

        print("\nClassification report:")
        print(classification_report(y_test, predictions, zero_division=0))

    # Stage 6: Compare results
    print("\nSTAGE 5 - BASELINE COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(results_df.to_string(index=False, formatters={
        "accuracy": "{:.3f}".format,
        "precision": "{:.3f}".format,
        "recall": "{:.3f}".format,
        "f1": "{:.3f}".format,
        "roc_auc": "{:.3f}".format,
    }))

    print("\nBaseline evaluation completed.")
    print("Note: The best baseline model is not automatically the production model.")
    print("Model selection and production evaluation will be handled in later stages.")


if __name__ == "__main__":
    main()