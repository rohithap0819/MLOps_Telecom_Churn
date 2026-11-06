import os
import streamlit as st
import requests
from pathlib import Path
import pandas as pd
import json

# FastAPI backend URL
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")

# Page configuration
st.set_page_config(
    page_title="Telecom Churn MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-subtitle {
        color: #888888;
        font-size: 15px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.07);
        padding: 12px;
        border-radius: 10px;
    }

    /* Reduce sidebar metric value font size */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        line-height: 1.3 !important;
    }

    /* Slightly reduce metric label size */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR - PRODUCTION STATUS
# ==========================================

st.sidebar.title("📊 Telecom MLOps")
st.sidebar.caption("Production Control Center")

st.sidebar.divider()

st.sidebar.subheader("Production Status")

# Fetch the latest production status from FastAPI.
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

    api_status = "Online"

except requests.exceptions.RequestException as error:
    health_data = {}
    model_data = {}
    api_status = "Offline"
    st.sidebar.error(f"Unable to connect to FastAPI: {error}")

# API Status
st.sidebar.metric(
    "API Status",
    api_status
)

# Production Model
st.sidebar.metric(
    "Production Model",
    model_data.get("production_model") or "Not Available"
)

# Approval Status
st.sidebar.metric(
    "Approval Status",
    model_data.get("approval_status") or "Not Available"
)

# Model Version
st.sidebar.metric(
    "Model Version",
    model_data.get("model_version") or "Not Available"
)

if api_status == "Offline":
    st.sidebar.error("Unable to connect to FastAPI.")

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "Telecom Customer Churn | MLOps Platform"
)


# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Telecom Customer Churn</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Machine Learning Operations | Prediction, Monitoring and Model Management'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# HORIZONTAL DASHBOARD NAVIGATION
# ==========================================

tab_prediction, tab_monitoring, tab_comparison, tab_approval = st.tabs([
    "🔍 Customer Prediction",
    "📈 Model Monitoring",
    "⚖️ Model Comparison",
    "👤 Human Approval"
])

with tab_prediction:

    st.header("Customer Churn Prediction")
    st.write(
        "Enter customer details to predict whether a customer "
        "is likely to churn."
    )

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
# LIVE MLOPS MONITORING
# ==========================================

@st.fragment(run_every="15s")
def render_live_monitoring():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    HISTORY_PATH = PROJECT_ROOT / "logs" / "monitoring_history.csv"
    CANDIDATE_PATH = PROJECT_ROOT / "logs" / "candidate_metrics.csv"
    REGISTRY_PATH = (
        PROJECT_ROOT / "models" / "production_registry.json"
    )

    st.header("Live MLOps Monitoring")

    st.caption(
        "Automatically refreshes monitoring information every 15 seconds."
    )

    if not HISTORY_PATH.exists():
        st.info("No monitoring history available.")
        return

    history_df = pd.read_csv(HISTORY_PATH)

    if history_df.empty:
        st.info("Monitoring history is empty.")
        return

    history_df["retraining_required"] = (
        history_df["retraining_required"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    history_df = history_df.sort_values("timestamp")

    latest = history_df.iloc[-1]

    # --------------------------------------
    # OVERALL MONITORING METRICS
    # --------------------------------------

    total_batches = len(history_df)

    drifted_batches = int(
        history_df["retraining_required"].sum()
    )

    normal_batches = total_batches - drifted_batches

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Batches", total_batches)
    col2.metric("Normal Batches", normal_batches)
    col3.metric("Drifted Batches", drifted_batches)

    col4.metric(
        "Latest Drift",
        f"{latest['drift_percentage']:.2f}%"
    )

    st.divider()

    # --------------------------------------
    # LATEST BATCH STATUS
    # --------------------------------------

    st.subheader("Latest Incoming Batch")

    st.write("Batch:", latest["batch_name"])
    st.write("Timestamp:", latest["timestamp"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Drifted Features",
        f"{int(latest['drifted_features'])} / "
        f"{int(latest['total_features'])}"
    )

    col2.metric(
        "Drift Percentage",
        f"{latest['drift_percentage']:.2f}%"
    )

    col3.metric(
        "Configured Threshold",
        f"{latest['drift_threshold']:.2f}%"
    )

    if latest["retraining_required"]:
        st.error(
            "Drift threshold exceeded. "
            "Candidate training was triggered."
        )
    else:
        st.success(
            "Normal batch. Drift threshold was not exceeded."
        )

    st.divider()

    # --------------------------------------
    # BATCH HISTORY
    # --------------------------------------

    st.subheader("Incoming Batch History")

    display_df = history_df.sort_values(
        "timestamp",
        ascending=False
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------
    # DRIFT CHART
    # --------------------------------------

    st.subheader("Data Drift Across Incoming Batches")

    chart_df = history_df[
        ["batch_name", "drift_percentage", "drift_threshold"]
    ].set_index("batch_name")

    st.line_chart(chart_df)

    # --------------------------------------
    # CANDIDATE MODEL RESULTS
    # --------------------------------------

    st.divider()

    st.subheader("Candidate Model Performance")

    if CANDIDATE_PATH.exists():

        candidate_df = pd.read_csv(CANDIDATE_PATH)

        st.dataframe(
            candidate_df,
            use_container_width=True,
            hide_index=True
        )

        if {"model_name", "f1_score"}.issubset(candidate_df.columns):

            f1_df = candidate_df[
                ["model_name", "f1_score"]
            ].set_index("model_name")

            st.bar_chart(f1_df)

    else:
        st.info("No candidate model metrics available yet.")

    # --------------------------------------
    # CURRENT PRODUCTION MODEL
    # --------------------------------------

    st.divider()

    st.subheader("Production Model Status")

    if REGISTRY_PATH.exists():

        with open(REGISTRY_PATH, "r") as file:
            registry = json.load(file)

        col1, col2 = st.columns(2)

        col1.metric(
            "Production Model",
            registry.get("production_model", "Unknown")
        )

        col2.metric(
            "Approval Status",
            registry.get("approval_status", "Unknown")
        )

        st.write(
            "Approved By:",
            registry.get("approved_by", "Unknown")
        )

        st.write(
            "Model Version:",
            registry.get("model_version", "Unknown")
        )

    else:
        st.info("Production model registry not found.")

    # --------------------------------------
    # EVIDENTLY REPORT
    # --------------------------------------

    st.divider()

    st.subheader("Latest Evidently Drift Report")

    report_path = Path(str(latest["report_path"]))

    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    if report_path.exists():

        with open(report_path, "rb") as report_file:

            st.download_button(
                "Download Latest Drift Report",
                data=report_file.read(),
                file_name=report_path.name,
                mime="text/html"
            )

    else:
        st.info("Latest drift report not found.")


with tab_monitoring:
    render_live_monitoring()


# ==========================================
# MODEL COMPARISON
# ==========================================

@st.fragment(run_every="15s")
def render_model_comparison():

    st.header("Model Performance Comparison")

    st.caption(
        "Compare the current production model against newly trained "
        "candidate models. Data refreshes automatically every 15 seconds."
    )

    try:
        # Fetch the latest production and candidate metrics.
        response = requests.get(
            f"{API_URL}/model-comparison",
            timeout=10
        )

        response.raise_for_status()
        comparison_data = response.json()

        production_df = pd.DataFrame(
            comparison_data.get("production", [])
        )

        candidates_df = pd.DataFrame(
            comparison_data.get("candidates", [])
        )

        if production_df.empty and candidates_df.empty:
            st.warning("No model comparison data is available.")
            return

        # Identify production and candidate models.
        if not production_df.empty:
            production_df["model_type"] = "Production"

        if not candidates_df.empty:
            candidates_df["model_type"] = "Candidate"

        # Combine all model results.
        comparison_df = pd.concat(
            [production_df, candidates_df],
            ignore_index=True
        )

        st.subheader("Model Performance Metrics")

        display_columns = [
            "model_type",
            "model_name",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]

        st.dataframe(
            comparison_df[display_columns].round(4),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Compare models using a selected evaluation metric.
        st.subheader("Visual Model Comparison")

        selected_metric = st.selectbox(
            "Select Evaluation Metric",
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc"
            ],
            index=3,
            format_func=lambda value: value.replace("_", " ").title(),
            key="model_comparison_metric"
        )

        chart_df = comparison_df[
            ["model_name", selected_metric]
        ].set_index("model_name")

        st.bar_chart(chart_df)

        # Display the highest-performing model for the selected metric.
        best_model = comparison_df.loc[
            comparison_df[selected_metric].idxmax()
        ]

        st.success(
            f"Best model by {selected_metric.replace('_', ' ').upper()}: "
            f"{best_model['model_name']} "
            f"({best_model[selected_metric]:.4f})"
        )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Unable to retrieve model comparison data: {error}"
        )

    except (KeyError, ValueError) as error:

        st.error(
            f"Unable to process model comparison data: {error}"
        )


# Display the Model Comparison interface.
with tab_comparison:
    render_model_comparison()



# ==========================================
# HUMAN MODEL APPROVAL
# ==========================================

@st.fragment(run_every="15s")
def render_human_approval():

    st.header("👤 Human Approval")

    st.caption(
        "Automatically checks for pending approval requests every 15 seconds."
    )

    try:
        # --------------------------------------------------
        # 1. Fetch the current pending approval request
        # --------------------------------------------------
        pending_response = requests.get(
            f"{API_URL}/pending-approval",
            timeout=10
        )

        pending_response.raise_for_status()
        pending_data = pending_response.json()

        # --------------------------------------------------
        # 2. No pending request
        # --------------------------------------------------
        if pending_data.get("status") == "no_pending_request":
            st.info(
                "No pending approval request. "
                "Waiting for the Event Watcher or Scheduler."
            )
            return

        approval_type = pending_data.get(
            "approval_type",
            "production"
        )

        trigger_source = pending_data.get(
            "trigger_source",
            "Unknown"
        )

        request_id = pending_data.get(
            "request_id",
            "Unknown"
        )

        # ==================================================
        # RETRAINING APPROVAL
        # ==================================================

        if approval_type == "retraining":

            st.warning(
                "⚠️ Data drift has triggered a retraining approval request."
            )

            st.subheader("🔄 Retraining Approval")

            col1, col2 = st.columns(2)

            col1.metric(
                "Approval Type",
                "Retraining"
            )

            col2.metric(
                "Triggered By",
                trigger_source
            )

            st.write(
                "**Request ID:**",
                request_id
            )

            st.write(
                "**Created At:**",
                pending_data.get("created_at", "Unknown")
            )

            st.divider()

            st.info(
                "The incoming batch exceeded the configured drift threshold. "
                "Candidate model training will begin only after human approval."
            )

            st.subheader("Human Decision")

            retraining_decision = st.radio(
                "Select retraining decision",
                [
                    "Approve Retraining",
                    "Reject Retraining"
                ],
                key="retraining_decision"
            )

            reviewer = st.text_input(
                "Reviewer Name",
                placeholder="Enter your name",
                key="retraining_reviewer"
            )

            decision_mapping = {
                "Approve Retraining": "approve_retraining",
                "Reject Retraining": "reject_retraining"
            }

            selected_decision = decision_mapping[
                retraining_decision
            ]

            if st.button(
                "Submit Retraining Decision",
                type="primary",
                use_container_width=True,
                key="submit_retraining_decision"
            ):

                if not reviewer.strip():

                    st.error(
                        "Please enter the reviewer name."
                    )

                else:

                    approval_payload = {
                        "decision": selected_decision,
                        "reviewer": reviewer.strip(),
                        "confirm_lower_f1": False
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

                            st.success(
                                result["message"]
                            )

                            st.write(
                                "Approval Request ID:",
                                result["request_id"]
                            )

                            st.info(
                                "Retraining has been approved. "
                                "Candidate model training is now starting."
                            )

                            st.rerun()

                        elif result["status"] == "rejected":

                            st.warning(
                                result["message"]
                            )

                            st.write(
                                "Approval Request ID:",
                                result["request_id"]
                            )

                            st.rerun()

                        else:

                            st.info(
                                result.get(
                                    "message",
                                    str(result)
                                )
                            )

                    except requests.exceptions.RequestException as error:

                        st.error(
                            f"Retraining approval request failed: {error}"
                        )

            return

        # ==================================================
        # PRODUCTION APPROVAL
        # ==================================================

        st.warning(
            "A trained candidate model is waiting for production approval."
        )

        # --------------------------------------------------
        # Fetch production and candidate model metrics
        # --------------------------------------------------
        comparison_response = requests.get(
            f"{API_URL}/model-comparison",
            timeout=10
        )

        comparison_response.raise_for_status()
        comparison_data = comparison_response.json()

        production_df = pd.DataFrame(
            comparison_data.get("production", [])
        )

        candidates_df = pd.DataFrame(
            comparison_data.get("candidates", [])
        )

        # --------------------------------------------------
        # Pending request details
        # --------------------------------------------------
        st.subheader("Production Approval")

        col1, col2 = st.columns(2)

        col1.metric(
            "Approval Type",
            "Production"
        )

        col2.metric(
            "Triggered By",
            trigger_source
        )

        st.write(
            "**Request ID:**",
            request_id
        )

        st.write(
            "**Created At:**",
            pending_data.get("created_at", "Unknown")
        )

        st.divider()

        # --------------------------------------------------
        # Current production model
        # --------------------------------------------------
        st.subheader("Current Production Model")

        current_production_f1 = None

        if not production_df.empty:

            current_production_f1 = float(
                production_df.iloc[0]["f1_score"]
            )

            st.dataframe(
                production_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No production metrics are available."
            )

        # --------------------------------------------------
        # Candidate model comparison
        # --------------------------------------------------
        st.subheader("Candidate Model Comparison")

        if candidates_df.empty:

            st.error(
                "No candidate model metrics are available."
            )

            return

        st.dataframe(
            candidates_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # Human production approval decision
        # --------------------------------------------------
        st.divider()

        st.subheader(
            "Select Production Approval Decision"
        )

        decision_options = {
            "Approve Logistic Regression": "logistic_regression",
            "Approve Decision Tree": "decision_tree",
            "Reject Both Candidates": "reject_all"
        }

        selected_label = st.selectbox(
            "Model Approval Decision",
            list(decision_options.keys()),
            key="production_approval_decision"
        )

        selected_decision = decision_options[
            selected_label
        ]

        reviewer = st.text_input(
            "Reviewer Name",
            placeholder="Enter your name",
            key="production_approval_reviewer"
        )

        candidate_f1 = None
        lower_f1_confirmation = False

        # --------------------------------------------------
        # Lower F1 confirmation
        # --------------------------------------------------
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
                        f"Selected candidate F1-score "
                        f"({candidate_f1:.4f}) is lower than "
                        f"production F1-score "
                        f"({current_production_f1:.4f})."
                    )

                    lower_f1_confirmation = st.checkbox(
                        "I understand the performance difference "
                        "and still want to approve this model.",
                        key="production_lower_f1_confirmation"
                    )

        approval_blocked = (
            selected_decision != "reject_all"
            and current_production_f1 is not None
            and candidate_f1 is not None
            and candidate_f1 < current_production_f1
            and not lower_f1_confirmation
        )

        # --------------------------------------------------
        # Submit production decision
        # --------------------------------------------------
        if st.button(
            "Submit Production Decision",
            type="primary",
            use_container_width=True,
            disabled=approval_blocked,
            key="submit_production_decision"
        ):

            if not reviewer.strip():

                st.error(
                    "Please enter the reviewer name."
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

                        st.success(
                            result["message"]
                        )

                        st.write(
                            "Approval Request ID:",
                            result["request_id"]
                        )

                        st.info(
                            "The approved candidate has been "
                            "promoted to production."
                        )

                        st.rerun()

                    elif result["status"] == "rejected":

                        st.warning(
                            result["message"]
                        )

                        st.write(
                            "Approval Request ID:",
                            result["request_id"]
                        )

                        st.rerun()

                    elif result["status"] == "confirmation_required":

                        st.warning(
                            result["message"]
                        )

                    else:

                        st.info(
                            result.get(
                                "message",
                                str(result)
                            )
                        )

                except requests.exceptions.RequestException as error:

                    st.error(
                        f"Production approval request failed: {error}"
                    )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Unable to load approval information: {error}"
        )


# Display the Human Approval interface.

# Display the Human Approval interface.
with tab_approval:
    render_human_approval()