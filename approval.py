
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates"
PRODUCTION_DIR = PROJECT_ROOT / "models" / "production"
REGISTRY_PATH = PROJECT_ROOT / "models" / "production_registry.json"

CANDIDATE_METRICS_PATH = PROJECT_ROOT / "logs" / "candidate_metrics.csv"
PRODUCTION_METRICS_PATH = PROJECT_ROOT / "logs" / "production_metrics.csv"
APPROVAL_LOG_PATH = PROJECT_ROOT / "logs" / "approval_requests.json"

PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
APPROVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load production registry
# --------------------------------------------------

def load_registry():
    if not REGISTRY_PATH.exists():
        return {"production_model": None}

    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)


# --------------------------------------------------
# 3. Load model metrics
# --------------------------------------------------

def load_candidate_metrics():
    if not CANDIDATE_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Candidate metrics not found. Run candidate_training.py first."
        )

    metrics = pd.read_csv(CANDIDATE_METRICS_PATH)

    if metrics.empty:
        raise ValueError("No candidate model results found.")

    return metrics


def load_production_metrics():
    if not PRODUCTION_METRICS_PATH.exists():
        print("\nWarning: Production metrics not found.")
        return pd.DataFrame()

    return pd.read_csv(PRODUCTION_METRICS_PATH)


# --------------------------------------------------
# 4. Display production and candidate comparison
# --------------------------------------------------

def display_model_comparison(candidate_metrics, production_metrics):

    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 75)

    display_columns = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    if not production_metrics.empty:
        comparison = pd.concat(
            [production_metrics, candidate_metrics],
            ignore_index=True,
        )
    else:
        comparison = candidate_metrics.copy()

    print(comparison[display_columns].to_string(index=False))

    if not production_metrics.empty:
        production_f1 = production_metrics.iloc[0]["f1_score"]

        print("\nCandidate performance against production:")

        for _, row in candidate_metrics.iterrows():
            difference = row["f1_score"] - production_f1

            if difference > 0:
                status = "Better"
            elif difference < 0:
                status = "Worse"
            else:
                status = "Equal"

            print(
                f"{row['model_name']}: "
                f"{status} (F1 difference: {difference:+.4f})"
            )

    print("\nNote: Metrics are evaluated on the same test split "
          "for this training run.")


# --------------------------------------------------
# 5. Record approval decision
# --------------------------------------------------

def record_decision(decision, reviewer):

    if APPROVAL_LOG_PATH.exists():
        with open(APPROVAL_LOG_PATH, "r", encoding="utf-8") as file:
            approval_history = json.load(file)
    else:
        approval_history = []

    request = {
        "request_id": datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        ),
        "decision": decision,
        "reviewer": reviewer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    approval_history.append(request)

    with open(APPROVAL_LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(approval_history, file, indent=4)

    return request


# --------------------------------------------------
# 6. Promote approved candidate
# --------------------------------------------------

def promote_candidate(model_name, reviewer):

    candidate_path = CANDIDATE_DIR / f"{model_name}.joblib"

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {candidate_path}"
        )

    registry = load_registry()

    production_path = PRODUCTION_DIR / "production_model.joblib"

    previous_model = registry.get("production_model")
    previous_version = registry.get("model_version")

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    # Preserve the existing production model before replacement.
    if production_path.exists():

        backup_path = (
            PRODUCTION_DIR /
            f"production_model_backup_{timestamp}.joblib"
        )

        shutil.copy2(production_path, backup_path)

        print(f"\nPrevious production model backed up: {backup_path}")

    # Copy candidate to a temporary file, then replace production.
    temporary_path = PRODUCTION_DIR / "production_model.tmp"

    shutil.copy2(candidate_path, temporary_path)
    temporary_path.replace(production_path)

    # Update production registry.
    registry.update({
        "production_model": model_name,
        "model_version": timestamp,
        "approved_by": reviewer,
        "approval_status": "approved",
        "approval_timestamp": datetime.now(timezone.utc).isoformat(),
        "previous_production_model": previous_model,
        "previous_model_version": previous_version,
    })

    save_registry(registry)

    print("\n" + "=" * 55)
    print("PRODUCTION MODEL PROMOTED")
    print("=" * 55)
    print(f"Approved model : {model_name}")
    print(f"Approved by    : {reviewer}")
    print(f"Model location : {production_path}")


def make_approval_decision(
    decision,
    reviewer,
    confirm_lower_f1=False
):
    """
    Reusable approval workflow for the FastAPI backend
    and Streamlit dashboard.
    """

    if not reviewer or not reviewer.strip():
        raise ValueError("Reviewer name cannot be empty.")

    valid_decisions = [
        "logistic_regression",
        "decision_tree",
        "reject_all"
    ]

    if decision not in valid_decisions:
        raise ValueError("Invalid model approval decision.")

    candidate_metrics = load_candidate_metrics()
    production_metrics = load_production_metrics()

    if decision == "reject_all":
        record = record_decision(decision, reviewer)

        return {
            "status": "rejected",
            "message": "Both candidates rejected. Production model unchanged.",
            "request_id": record["request_id"]
        }

    selected = candidate_metrics[
        candidate_metrics["model_name"] == decision
    ]

    if selected.empty:
        raise ValueError(
            f"Metrics not found for candidate: {decision}"
        )

    # Compare candidate performance with production.
    if not production_metrics.empty:

        production_f1 = float(
            production_metrics.iloc[0]["f1_score"]
        )

        candidate_f1 = float(
            selected.iloc[0]["f1_score"]
        )

        if candidate_f1 < production_f1 and not confirm_lower_f1:
            return {
                "status": "confirmation_required",
                "message": (
                    "Candidate has a lower F1-score "
                    "than the production model."
                ),
                "production_f1": production_f1,
                "candidate_f1": candidate_f1
            }

    # Record and promote the approved candidate.
    record = record_decision(decision, reviewer)

    promote_candidate(decision, reviewer)

    return {
        "status": "approved",
        "message": f"{decision} promoted to production successfully.",
        "request_id": record["request_id"]
    }

# --------------------------------------------------
# 7. Human approval workflow
# --------------------------------------------------

def main():

    candidate_metrics = load_candidate_metrics()
    production_metrics = load_production_metrics()

    display_model_comparison(
        candidate_metrics,
        production_metrics,
    )

    print("\n" + "=" * 55)
    print("HUMAN PRODUCTION APPROVAL")
    print("=" * 55)

    print("1. Approve Logistic Regression")
    print("2. Approve Decision Tree")
    print("3. Reject both candidates")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    reviewer = input("Enter reviewer name: ").strip()

    if not reviewer:
        print("Reviewer name cannot be empty. No changes made.")
        return

    choices = {
        "1": "logistic_regression",
        "2": "decision_tree",
        "3": "reject_all",
    }

    if choice not in choices:
        print("Invalid choice. No changes made.")
        return

    decision = choices[choice]

    # Warn if the selected candidate is worse than production.
    if decision != "reject_all" and not production_metrics.empty:

        production_f1 = production_metrics.iloc[0]["f1_score"]

        selected = candidate_metrics[
            candidate_metrics["model_name"] == decision
        ]

        if selected.empty:
            print("Selected candidate metrics not found.")
            return

        candidate_f1 = selected.iloc[0]["f1_score"]

        if candidate_f1 < production_f1:
            print("\nWARNING: Selected candidate has a lower F1-score "
                  "than the existing production model.")

            print(f"Production F1: {production_f1:.4f}")
            print(f"Candidate F1 : {candidate_f1:.4f}")

            confirmation = input(
                "Do you still want to approve this candidate? (yes/no): "
            ).strip().lower()

            if confirmation != "yes":
                print("\nPromotion cancelled. Production model unchanged.")
                return

    record = record_decision(decision, reviewer)

    if decision == "reject_all":

        print("\nBoth candidates were rejected.")
        print("The existing production model remains unchanged.")

        registry = load_registry()

        if registry.get("production_model") is not None:
            print(
                f"Current production model: "
                f"{registry['production_model']}"
            )
        else:
            print("No production model has been approved yet.")

        print(f"Decision recorded. Request ID: {record['request_id']}")
        return

    promote_candidate(decision, reviewer)

    print(f"\nDecision recorded. Request ID: {record['request_id']}")


if __name__ == "__main__":
    main()
