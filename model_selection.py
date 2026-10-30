"""Model selection for telecom customer churn.

Run:
    python model_selection.py
"""

from pathlib import Path

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


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "train.csv"
TARGET = "churn"
RANDOM_STATE = 42


def load_data():
    """Load and prepare the historical training dataset."""
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing.")

    df = df.dropna(subset=[TARGET]).copy()

    y = df[TARGET].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    )

    if y.isna().any():
        raise ValueError("Unexpected values found in the churn target column.")

    X = df.drop(columns=[TARGET, "customer_id"], errors="ignore")
    return X, y


def build_pipeline(model):
    """Build preprocessing and model steps inside one pipeline."""
    numeric_cols = model["numeric_cols"]
    categorical_cols = model["categorical_cols"]

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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model["estimator"]),
        ]
    )


def main():
    print("=" * 70)
    print("MODEL SELECTION - TELECOM CUSTOMER CHURN")
    print("=" * 70)

    # Stage 1: Load data
    print("\nSTAGE 1 - DATA LOADING")
    X, y = load_data()
    print(f"Rows: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Churn rate: {y.mean():.1%}")

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Stage 2: Define candidate models
    print("\nSTAGE 2 - CANDIDATE MODELS")

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

    # Stage 3: Cross-validation
    print("\nSTAGE 3 - STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("Preprocessing is fitted separately inside each fold.")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    results = []

    for name, estimator in candidates.items():
        print(f"\nEvaluating: {name}")

        pipeline = build_pipeline(
            {
                "estimator": estimator,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
            }
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        results.append(row)

        print(f"Mean F1-score : {row['f1_mean']:.3f} (+/- {row['f1_std']:.3f})")
        print(f"Mean Precision: {row['precision_mean']:.3f}")
        print(f"Mean Recall   : {row['recall_mean']:.3f}")
        print(f"Mean ROC-AUC  : {row['roc_auc_mean']:.3f}")

    # Stage 4: Compare candidates
    print("\nSTAGE 4 - MODEL COMPARISON")
    results_df = pd.DataFrame(results).sort_values(
        "f1_mean", ascending=False
    )
    print(results_df.to_string(index=False, formatters={
        column: "{:.3f}".format
        for column in results_df.select_dtypes(include="number").columns
    }))

    # Stage 5: Select candidate
    winner = results_df.iloc[0]

    print("\nSTAGE 5 - CANDIDATE SELECTION")
    print(f"Selected candidate by mean cross-validation F1: {winner['model']}")
    print(f"Mean F1-score: {winner['f1_mean']:.3f}")
    print("\nThis is a candidate selection, not automatic production promotion.")
    print("The final model must still pass the main pipeline's evaluation gates.")


if __name__ == "__main__":
    main()