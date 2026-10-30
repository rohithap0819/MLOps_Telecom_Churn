import streamlit as st
import requests
from pathlib import Path
import pandas as pd

# FastAPI backend URL
API_URL = "http://127.0.0.1:8001"

# Page configuration
st.set_page_config(
    page_title="Telecom Churn MLOps Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom Customer Churn - MLOps Dashboard")
st.write("Monitor the production model and access churn prediction services.")

st.divider()

# Fetch API health
try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )
    health_response.raise_for_status()
    health_data = health_response.json()

    model_response = requests.get(
        f"{API_URL}/model-info",
        timeout=5
    )
    model_response.raise_for_status()
    model_data = model_response.json()

    # Dashboard metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "API Status",
            "Online" if health_data["status"] == "healthy" else "Offline"
        )

    with col2:
        st.metric(
            "Production Model",
            model_data.get("production_model") or "Not Available"
        )

    with col3:
        st.metric(
            "Approval Status",
            model_data.get("approval_status") or "Not Available"
        )

    st.divider()

    st.subheader("Production Model Information")

    st.write("**Model Version:**", model_data.get("model_version", "N/A"))
    st.write("**Approved By:**", model_data.get("approved_by", "N/A"))

except requests.exceptions.RequestException:
    st.error(
        "Unable to connect to FastAPI. "
        "Please start the backend on port 8001."
    )

st.divider()

st.caption("Telecom Churn Prediction | MLOps Project")


st.divider()

st.header("🔍 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to churn.")

with st.form("customer_prediction_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure_months = st.number_input(
            "Tenure (Months)", min_value=0, max_value=100, value=12
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.subheader("Billing Information")

    col4, col5 = st.columns(2)

    with col4:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=85.5,
            step=1.0
        )

    with col5:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1026.0,
            step=10.0
        )

    predict_button = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )

if predict_button:

    customer_data = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }

    try:
        with st.spinner("Predicting customer churn..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=customer_data,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

        st.subheader("Prediction Result")

        if result["churn_prediction"] == 1:
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is unlikely to churn.")

        col6, col7 = st.columns(2)

        with col6:
            st.metric("Prediction", result["churn_label"])

        with col7:
            st.metric(
                "Churn Probability",
                f"{result['churn_probability'] * 100:.2f}%"
            )

        st.progress(result["churn_probability"])

    except requests.exceptions.RequestException as error:
        st.error(f"Prediction request failed: {error}")




# ==========================================
# MODEL MONITORING
# ==========================================

st.divider()

st.header("📈 Model Monitoring")
st.write("Track data drift and monitor incoming data batches.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = PROJECT_ROOT / "logs" / "monitoring_history.csv"

if HISTORY_PATH.exists():

    history_df = pd.read_csv(HISTORY_PATH)

    if not history_df.empty:

        latest = history_df.iloc[-1]

        st.subheader("Latest Monitoring Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Features",
                int(latest["total_features"])
            )

        with col2:
            st.metric(
                "Drifted Features",
                int(latest["drifted_features"])
            )

        with col3:
            st.metric(
                "Drift Percentage",
                f"{latest['drift_percentage']:.2f}%"
            )

        with col4:
            st.metric(
                "Drift Threshold",
                f"{latest['drift_threshold']:.2f}%"
            )

        # Drift status
        if latest["retraining_required"]:
            st.warning("Drift threshold exceeded. Candidate training was triggered.")
        else:
            st.success("Drift is within the configured threshold.")

        st.write("**Latest Batch:**", latest["batch_name"])
        st.write("**Monitoring Timestamp:**", latest["timestamp"])

        # Monitoring history
        st.subheader("Monitoring History")

        st.dataframe(
            history_df.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Drift percentage chart
        st.subheader("Drift Percentage Across Batches")

        chart_df = history_df[["batch_name", "drift_percentage"]].set_index("batch_name")

        st.bar_chart(chart_df)

        # Download latest Evidently report
        st.subheader("Evidently Drift Report")

        report_path = Path(str(latest["report_path"]))

        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path

        if report_path.exists():

            with open(report_path, "rb") as report_file:
                st.download_button(
                    label="Download Latest Drift Report",
                    data=report_file.read(),
                    file_name=report_path.name,
                    mime="text/html"
                )

        else:
            st.info("Latest Evidently HTML report was not found.")

    else:
        st.info("Monitoring history is currently empty.")

else:
    st.info(
        "No monitoring history available. "
        "Run the monitoring pipeline to generate results."
    )


# ==========================================
# MODEL COMPARISON
# ==========================================

st.divider()

st.header("⚖️ Model Comparison")
st.write("Compare the active production model against candidate models.")

try:
    comparison_response = requests.get(
        f"{API_URL}/model-comparison",
        timeout=10
    )
    comparison_response.raise_for_status()

    comparison_data = comparison_response.json()

    production_df = pd.DataFrame(comparison_data["production"])
    candidates_df = pd.DataFrame(comparison_data["candidates"])

    comparison_df = pd.concat(
        [production_df, candidates_df],
        ignore_index=True
    )

    display_columns = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    st.dataframe(
        comparison_df[display_columns].style.format({
            "accuracy": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1_score": "{:.4f}",
            "roc_auc": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("F1-score Comparison")

    chart_df = comparison_df[
        ["model_name", "f1_score"]
    ].set_index("model_name")

    st.bar_chart(chart_df)

    st.subheader("ROC-AUC Comparison")

    roc_df = comparison_df[
        ["model_name", "roc_auc"]
    ].set_index("model_name")

    st.bar_chart(roc_df)

except requests.exceptions.RequestException as error:
    st.error(f"Unable to retrieve model comparison: {error}")



# ==========================================
# HUMAN MODEL APPROVAL
# ==========================================

st.divider()

st.header("👤 Human Model Approval")

st.write(
    "Review candidate model performance and decide "
    "which model should become the production model."
)

try:
    approval_response = requests.get(
        f"{API_URL}/model-comparison",
        timeout=10
    )
    approval_response.raise_for_status()

    approval_data = approval_response.json()

    production_df = pd.DataFrame(approval_data["production"])
    candidates_df = pd.DataFrame(approval_data["candidates"])

    if not production_df.empty:

        current_production_f1 = float(
            production_df.iloc[0]["f1_score"]
        )

        st.info(
            f"Current Production Model: "
            f"{production_df.iloc[0]['model_name']} | "
            f"F1-score: {current_production_f1:.4f}"
        )

    else:
        current_production_f1 = None
        st.warning("No production model is currently registered.")

    st.subheader("Select Your Approval Decision")

    decision_options = {
        "Approve Logistic Regression": "logistic_regression",
        "Approve Decision Tree": "decision_tree",
        "Reject Both Candidates": "reject_all"
    }

    selected_label = st.selectbox(
        "Model Approval Decision",
        list(decision_options.keys())
    )

    selected_decision = decision_options[selected_label]

    reviewer = st.text_input(
        "Reviewer Name",
        placeholder="Enter your name"
    )

    lower_f1_confirmation = False

    if (
        selected_decision != "reject_all"
        and current_production_f1 is not None
    ):

        selected_model = candidates_df[
            candidates_df["model_name"] == selected_decision
        ]

        if not selected_model.empty:

            candidate_f1 = float(
                selected_model.iloc[0]["f1_score"]
            )

            if candidate_f1 < current_production_f1:

                st.warning(
                    f"Selected candidate F1-score ({candidate_f1:.4f}) "
                    f"is lower than production F1-score "
                    f"({current_production_f1:.4f})."
                )

                lower_f1_confirmation = st.checkbox(
                    "I understand the performance difference "
                    "and still want to approve this model."
                )

    approval_blocked = (
        selected_decision != "reject_all"
        and current_production_f1 is not None
        and candidate_f1 is not None
        and candidate_f1 < current_production_f1
        and not lower_f1_confirmation
    )

    if st.button(
        "Submit Approval Decision",
        type="primary",
        use_container_width=True,
        disabled=approval_blocked
    ):

        if not reviewer.strip():
            st.error("Please enter the reviewer name.")

        elif (
            selected_decision != "reject_all"
            and current_production_f1 is not None
            and candidate_f1 < current_production_f1
            and not lower_f1_confirmation
        ):
            st.error(
                "Please confirm the performance warning before proceeding."
            )

        else:

            approval_payload = {
                "decision": selected_decision,
                "reviewer": reviewer.strip(),
                "confirm_lower_f1": lower_f1_confirmation
            }

            try:
                result_response = requests.post(
                    f"{API_URL}/approval",
                    json=approval_payload,
                    timeout=30
                )

                result_response.raise_for_status()
                result = result_response.json()

                if result["status"] == "approved":
                    st.success(result["message"])
                    st.write(
                        "Approval Request ID:",
                        result["request_id"]
                    )

                elif result["status"] == "rejected":
                    st.info(result["message"])
                    st.write(
                        "Approval Request ID:",
                        result["request_id"]
                    )

                else:
                    st.warning(result["message"])

            except requests.exceptions.RequestException as error:
                st.error(f"Approval request failed: {error}")

except requests.exceptions.RequestException as error:
    st.error(f"Unable to load approval information: {error}")
