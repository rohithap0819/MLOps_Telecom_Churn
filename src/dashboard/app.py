import os
import json
import streamlit as st
import requests
from pathlib import Path
import pandas as pd


# ============================================================
# FASTAPI BACKEND CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom Churn MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

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

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        line-height: 1.3 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR - PRODUCTION STATUS
# ============================================================

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

    st.sidebar.error(
        f"Unable to connect to FastAPI: {error}"
    )


st.sidebar.metric(
    "API Status",
    api_status
)

st.sidebar.metric(
    "Production Model",
    model_data.get("production_model") or "Not Available"
)

st.sidebar.metric(
    "Approval Status",
    model_data.get("approval_status") or "Not Available"
)

st.sidebar.metric(
    "Model Version",
    model_data.get("model_version") or "Not Available"
)


if api_status == "Offline":
    st.sidebar.error("Unable to connect to FastAPI.")


st.sidebar.divider()


if st.sidebar.button(
    "🔄 Refresh Status",
    use_container_width=True
):
    st.cache_data.clear()
    st.rerun()


st.sidebar.caption(
    "Telecom Customer Churn | MLOps Platform"
)


# ============================================================
# MAIN DASHBOARD HEADER
# ============================================================

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


# ============================================================
# HORIZONTAL DASHBOARD NAVIGATION
# ============================================================

tab_prediction, tab_monitoring, tab_comparison, tab_approval = st.tabs([
    "🔍 Customer Prediction",
    "📈 Model Monitoring",
    "⚖️ Model Comparison",
    "👤 Human Approval"
])


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

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
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure_months = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                value=12
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

        with col2:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )

        with col3:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
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
                st.metric(
                    "Prediction",
                    result["churn_label"]
                )

            with col7:
                st.metric(
                    "Churn Probability",
                    f"{result['churn_probability'] * 100:.2f}%"
                )

            st.progress(result["churn_probability"])

        except requests.exceptions.RequestException as error:

            st.error(
                f"Prediction request failed: {error}"
            )


# ============================================================
# LIVE MLOPS MONITORING
# ============================================================

@st.fragment(run_every="15s")
def render_live_monitoring():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    HISTORY_PATH = (
        PROJECT_ROOT / "logs" / "monitoring_history.csv"
    )

    CANDIDATE_PATH = (
        PROJECT_ROOT / "logs" / "candidate_metrics.csv"
    )

    REGISTRY_PATH = (
        PROJECT_ROOT / "models" / "production_registry.json"
    )

    st.header("Live MLOps Monitoring")

    st.caption(
        "Automatically refreshes monitoring information every 15 seconds."
    )

    if not HISTORY_PATH.exists():

        st.info(
            "No monitoring history available."
        )

        return

    history_df = pd.read_csv(
        HISTORY_PATH
    )

    if history_df.empty:

        st.info(
            "Monitoring history is empty."
        )

        return

    history_df["retraining_required"] = (
        history_df["retraining_required"]
        .astype(str)
        .str.lower()
        .isin([
            "true",
            "1",
            "yes"
        ])
    )

    history_df = history_df.sort_values(
        "timestamp"
    )

    latest = history_df.iloc[-1]


    # ========================================================
    # OVERALL MONITORING METRICS
    # ========================================================

    total_batches = len(history_df)

    drifted_batches = int(
        history_df["retraining_required"].sum()
    )

    normal_batches = (
        total_batches - drifted_batches
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Batches",
        total_batches
    )

    col2.metric(
        "Normal Batches",
        normal_batches
    )

    col3.metric(
        "Drifted Batches",
        drifted_batches
    )

    col4.metric(
        "Latest Drift",
        f"{latest['drift_percentage']:.2f}%"
    )


    st.divider()


    # ========================================================
    # LATEST BATCH STATUS
    # ========================================================

    st.subheader(
        "Latest Incoming Batch"
    )

    st.write(
        "Batch:",
        latest["batch_name"]
    )

    st.write(
        "Timestamp:",
        latest["timestamp"]
    )

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


    # ========================================================
    # BATCH HISTORY
    # ========================================================

    st.subheader(
        "Incoming Batch History"
    )

    display_df = history_df.sort_values(
        "timestamp",
        ascending=False
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DRIFT CHART
    # ========================================================

    st.subheader(
        "Data Drift Across Incoming Batches"
    )

    chart_df = history_df[
        [
            "batch_name",
            "drift_percentage",
            "drift_threshold"
        ]
    ].set_index("batch_name")

    st.line_chart(
        chart_df
    )


    # ========================================================
    # CANDIDATE MODEL RESULTS
    # ========================================================

    st.divider()

    st.subheader(
        "Candidate Model Performance"
    )

    if CANDIDATE_PATH.exists():

        candidate_df = pd.read_csv(
            CANDIDATE_PATH
        )

        st.dataframe(
            candidate_df,
            use_container_width=True,
            hide_index=True
        )

        if {
            "model_name",
            "f1_score"
        }.issubset(candidate_df.columns):

            f1_df = candidate_df[
                [
                    "model_name",
                    "f1_score"
                ]
            ].set_index("model_name")

            st.bar_chart(
                f1_df
            )

    else:

        st.info(
            "No candidate model metrics available yet."
        )


    # ========================================================
    # CURRENT PRODUCTION MODEL
    # ========================================================

    st.divider()

    st.subheader(
        "Production Model Status"
    )

    if REGISTRY_PATH.exists():

        with open(
            REGISTRY_PATH,
            "r"
        ) as file:

            registry = json.load(
                file
            )

        col1, col2 = st.columns(2)

        col1.metric(
            "Production Model",
            registry.get(
                "production_model",
                "Unknown"
            )
        )

        col2.metric(
            "Approval Status",
            registry.get(
                "approval_status",
                "Unknown"
            )
        )

        st.write(
            "Approved By:",
            registry.get(
                "approved_by",
                "Unknown"
            )
        )

        st.write(
            "Model Version:",
            registry.get(
                "model_version",
                "Unknown"
            )
        )

    else:

        st.info(
            "Production model registry not found."
        )


    # ========================================================
    # SCHEDULER WORKFLOW
    # ========================================================

    # IMPORTANT:
    # This entire Scheduler section is intentionally INSIDE
    # render_live_monitoring() so it can access PROJECT_ROOT
    # and HISTORY_PATH defined above.

    st.divider()

    st.subheader(
        "📅 Scheduler Workflow"
    )

    st.caption(
        "Time-based monitoring of manually uploaded CSV batches. "
        "Any drift of 1% or more requires human approval for retraining."
    )

    scheduler_history_path = (
        PROJECT_ROOT
        / "logs"
        / "scheduler_processed_files.json"
    )

    approval_history_path = (
        PROJECT_ROOT
        / "logs"
        / "approval_requests.json"
    )


    # ========================================================
    # SCHEDULER BATCH HISTORY
    # ========================================================

    scheduler_batches = []

    # The Scheduler keeps its own processed-file history.
    # This prevents Event Watcher batches from appearing
    # inside the Scheduler dashboard.
    if scheduler_history_path.exists() and HISTORY_PATH.exists():

        try:

            with open(
                scheduler_history_path,
                "r"
            ) as scheduler_file:

                scheduler_processed_data = json.load(
                    scheduler_file
                )

            # scheduler_processed_files.json may contain either:
            #   - a simple list of processed file names
            #   - a dictionary containing a "processed_files" list
            if isinstance(
                scheduler_processed_data,
                list
            ):

                processed_files = scheduler_processed_data

            elif isinstance(
                scheduler_processed_data,
                dict
            ):

                processed_files = (
                    scheduler_processed_data.get(
                        "processed_files",
                        []
                    )
                )

            else:

                processed_files = []

            processed_files = {
                Path(str(file_name)).stem
                for file_name in processed_files
            }

            scheduler_history_df = pd.read_csv(
                HISTORY_PATH
            )

            # Only use monitoring records whose batch was
            # actually processed by the Scheduler.
            if (
                processed_files
                and "batch_name" in scheduler_history_df.columns
            ):

                scheduler_history_df = (
                    scheduler_history_df[
                        scheduler_history_df[
                            "batch_name"
                        ].apply(
                            lambda batch_name:
                                any(
                                    str(batch_name).startswith(
                                        processed_name
                                    )
                                    for processed_name
                                    in processed_files
                                )
                        )
                    ].copy()
                )

            else:

                scheduler_history_df = (
                    pd.DataFrame()
                )

            if not scheduler_history_df.empty:

                scheduler_history_df = (
                    scheduler_history_df
                    .sort_values("timestamp")
                    .drop_duplicates(
                        subset=["batch_name"],
                        keep="last"
                    )
                )

                for _, row in scheduler_history_df.iterrows():

                    batch_name = str(
                        row["batch_name"]
                    )

                    if "batch_1_" in batch_name:

                        display_batch = "Batch 1"

                    elif "batch_2_" in batch_name:

                        display_batch = "Batch 2"

                    elif "batch_3_" in batch_name:

                        display_batch = "Batch 3"

                    else:

                        display_batch = batch_name

                    # Use the recorded number of new records when
                    # available. Otherwise retain the learner
                    # dataset batch size of 581 rows.
                    new_rows = int(
                        row.get(
                            "new_records",
                            581
                        )
                    )

                    drift = float(
                        row["drift_percentage"]
                    )

                    # Scheduler policy:
                    # <1%  -> No Action
                    # >=1% -> Human Approval
                    if drift < 1:

                        action = "No Action"

                    else:

                        action = "Human Approval"

                    scheduler_batches.append({
                        "Batch": display_batch,
                        "New Rows": f"+{new_rows}",
                        "Drift": f"{drift:.2f}%",
                        "Action": action
                    })

        except (
            json.JSONDecodeError,
            OSError,
            ValueError,
            TypeError
        ) as error:

            st.warning(
                f"Unable to load Scheduler history: {error}"
            )


    if scheduler_batches:

        scheduler_display_df = pd.DataFrame(
            scheduler_batches
        )

        st.dataframe(
            scheduler_display_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # DATASET PROGRESSION
        # ====================================================

        st.subheader(
            "Dataset Progression"
        )

        # Start from the original 5,300-row learner dataset
        # and add the new records processed by the Scheduler.
        dataset_versions = [
            5300
        ]

        for new_rows_text in scheduler_display_df[
            "New Rows"
        ]:

            new_rows = int(
                str(new_rows_text).replace(
                    "+",
                    ""
                )
            )

            dataset_versions.append(
                dataset_versions[-1] + new_rows
            )

        progression_columns = min(
            len(dataset_versions),
            4
        )

        progression_cols = st.columns(
            progression_columns
        )

        for index in range(
            progression_columns
        ):

            progression_cols[index].metric(
                f"V{index}",
                f"{dataset_versions[index]:,} rows"
            )


        # ====================================================
        # LATEST SCHEDULER STATUS
        # ====================================================

        latest_scheduler_batch = (
            scheduler_display_df.iloc[-1]
        )

        latest_drift = float(
            latest_scheduler_batch[
                "Drift"
            ].replace(
                "%",
                ""
            )
        )

        st.subheader(
            "Latest Scheduler Event"
        )

        latest_col1, latest_col2, latest_col3 = (
            st.columns(3)
        )

        latest_col1.metric(
            "Batch",
            latest_scheduler_batch[
                "Batch"
            ]
        )

        latest_col2.metric(
            "Drift",
            latest_scheduler_batch[
                "Drift"
            ]
        )

        latest_col3.metric(
            "Action",
            latest_scheduler_batch[
                "Action"
            ]
        )

        if latest_drift < 1:

            st.success(
                "Scheduler detected less than 1% drift. "
                "No retraining approval was required."
            )

        else:

            st.warning(
                f"Scheduler detected {latest_drift:.2f}% drift. "
                "Human approval was required before retraining."
            )

    else:

        st.info(
            "No Scheduler batch history is available yet."
        )


    # ========================================================
    # SCHEDULER APPROVAL ACTIVITY
    # ========================================================

    if approval_history_path.exists():

        try:

            with open(
                approval_history_path,
                "r"
            ) as approval_file:

                approval_history = json.load(
                    approval_file
                )

            approval_df = pd.DataFrame(
                approval_history
            )

            if not approval_df.empty:

                # Sort the approval decisions by time.
                if "timestamp" in approval_df.columns:

                    approval_df["timestamp"] = (
                        pd.to_datetime(
                            approval_df[
                                "timestamp"
                            ],
                            errors="coerce"
                        )
                    )

                    approval_df = (
                        approval_df
                        .sort_values(
                            "timestamp",
                            ascending=False
                        )
                    )

                # Display only decisions made by the
                # Scheduler test reviewer.
                scheduler_approvals = approval_df[
                    approval_df.get(
                        "reviewer",
                        pd.Series(
                            dtype=str
                        )
                    ).astype(str).eq(
                        "Scheduler Test"
                    )
                ].copy()

                if not scheduler_approvals.empty:

                    st.subheader(
                        "Human Approval Activity"
                    )

                    approval_columns = [
                        column
                        for column in [
                            "request_id",
                            "decision",
                            "reviewer",
                            "timestamp"
                        ]
                        if column in scheduler_approvals.columns
                    ]

                    st.dataframe(
                        scheduler_approvals[
                            approval_columns
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.info(
                        "The Scheduler test approval decisions above "
                        "were recorded by the FastAPI approval workflow."
                    )

        except (
            json.JSONDecodeError,
            OSError,
            ValueError
        ) as error:

            st.warning(
                f"Unable to load Scheduler approval history: {error}"
            )


    # ========================================================
    # EVIDENTLY REPORT
    # ========================================================

    st.divider()

    st.subheader(
        "Latest Evidently Drift Report"
    )

    report_path = Path(
        str(latest["report_path"])
    )

    if not report_path.is_absolute():

        report_path = (
            PROJECT_ROOT / report_path
        )

    if report_path.exists():

        with open(
            report_path,
            "rb"
        ) as report_file:

            st.download_button(
                "Download Latest Drift Report",
                data=report_file.read(),
                file_name=report_path.name,
                mime="text/html"
            )

    else:

        st.info(
            "Latest drift report not found."
        )


# ============================================================
# DISPLAY MONITORING INTERFACE
# ============================================================

with tab_monitoring:
    render_live_monitoring()


# ============================================================
# MODEL COMPARISON
# ============================================================
@st.fragment(run_every="15s")
def render_model_comparison():

    st.header("Model Performance Comparison")

    st.caption(
        "Compare model performance across the MLOps lifecycle: "
        "initial selection, current monitoring, retrained candidates, "
        "and production."
    )

    try:

        # --------------------------------------------------------
        # FETCH MODEL COMPARISON DATA FROM FASTAPI
        # --------------------------------------------------------

        # FastAPI provides four lifecycle stages:
        #
        # 1. Initial model selection
        # 2. Current monitoring
        # 3. Retrained candidates
        # 4. Production model
        #
        # Candidate and production data are optional during the
        # initial project stage.

        response = requests.get(
            f"{API_URL}/model-comparison",
            timeout=10
        )

        response.raise_for_status()

        comparison_data = response.json()

        # Convert each API section into a DataFrame.
        initial_df = pd.DataFrame(
            comparison_data.get(
                "initial_model_selection",
                []
            )
        )

        monitoring_df = pd.DataFrame(
            comparison_data.get(
                "monitoring",
                []
            )
        )

        candidates_df = pd.DataFrame(
            comparison_data.get(
                "candidates",
                []
            )
        )

        production_df = pd.DataFrame(
            comparison_data.get(
                "production",
                []
            )
        )


        # ========================================================
        # 1. INITIAL MODEL SELECTION
        # ========================================================

        st.subheader(
            "Initial Model Selection"
        )

        st.caption(
            "Results generated by model_selection.py before "
            "any retraining workflow."
        )

        if not initial_df.empty:

            # Keep the original model-selection metric names.
            # These are CV mean/std metrics rather than
            # current-batch metrics.

            initial_display_columns = [
                column
                for column in [
                    "model",
                    "model_name",
                    "mlflow_run_id",
                    "f1_mean",
                    "f1_std",
                    "precision_mean",
                    "precision_std",
                    "recall_mean",
                    "recall_std",
                    "roc_auc_mean",
                    "roc_auc_std"
                ]
                if column in initial_df.columns
            ]

            initial_display_df = initial_df[
                initial_display_columns
            ].copy()

            # Add a clear lifecycle label.
            initial_display_df["stage"] = (
                "Initial Selection"
            )

            st.dataframe(
                initial_display_df,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # INITIAL MODEL F1 COMPARISON
            # ----------------------------------------------------

            if {
                "model",
                "f1_mean"
            }.issubset(initial_df.columns):

                initial_chart_df = (
                    initial_df[
                        [
                            "model",
                            "f1_mean"
                        ]
                    ]
                    .set_index("model")
                )

                st.subheader(
                    "Initial Model F1 Comparison"
                )

                st.bar_chart(
                    initial_chart_df
                )

                best_initial_model = initial_df.loc[
                    initial_df[
                        "f1_mean"
                    ].idxmax()
                ]

                st.success(
                    f"Selected initial model by CV F1: "
                    f"{best_initial_model['model']} "
                    f"({best_initial_model['f1_mean']:.4f})"
                )

        else:

            st.info(
                "Initial model-selection metrics are not available yet. "
                "Run model_selection.py first."
            )


        st.divider()


        # ========================================================
        # 2. CURRENT MONITORING
        # ========================================================

        st.subheader(
            "Current Monitoring"
        )

        st.caption(
            "Performance measured on the latest monitored batch. "
            "These are current-batch metrics, not cross-validation metrics."
        )

        if not monitoring_df.empty:

            # Convert timestamp to datetime when available so that
            # the latest monitoring record can be identified reliably.
            if "timestamp" in monitoring_df.columns:

                monitoring_df["timestamp"] = pd.to_datetime(
                    monitoring_df["timestamp"],
                    errors="coerce"
                )

                monitoring_df = (
                    monitoring_df
                    .sort_values("timestamp")
                )

            # The API contains historical monitoring records.
            # For Model Comparison, show the latest monitoring
            # result rather than displaying all 33 historical rows.
            latest_monitoring = (
                monitoring_df.iloc[-1:].copy()
            )

            monitoring_display_columns = [
                column
                for column in [
                    "timestamp",
                    "batch_name",
                    "model_name",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "drift_percentage",
                    "drift_threshold",
                    "retraining_required"
                ]
                if column in latest_monitoring.columns
            ]

            latest_monitoring_display = (
                latest_monitoring[
                    monitoring_display_columns
                ].copy()
            )

            latest_monitoring_display["stage"] = (
                "Current Monitoring"
            )

            st.dataframe(
                latest_monitoring_display,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # CURRENT MONITORING METRICS
            # ----------------------------------------------------

            metric_columns = [
                column
                for column in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ]
                if column in latest_monitoring.columns
            ]

            if metric_columns:

                metric_cols = st.columns(
                    len(metric_columns)
                )

                latest_row = (
                    latest_monitoring.iloc[0]
                )

                for metric_col, metric_container in zip(
                    metric_columns,
                    metric_cols
                ):

                    metric_container.metric(
                        metric_col.replace(
                            "_",
                            " "
                        ).title(),
                        f"{float(latest_row[metric_col]):.4f}"
                    )

            # ----------------------------------------------------
            # CURRENT MONITORING F1 VS INITIAL SELECTION
            # ----------------------------------------------------

            comparison_rows = []

            if (
                "model_name" in latest_monitoring.columns
                and "f1_score" in latest_monitoring.columns
            ):

                comparison_rows.append({
                    "Stage": "Current Monitoring",
                    "Model": latest_row["model_name"],
                    "F1 Score": float(
                        latest_row["f1_score"]
                    )
                })

            if (
                "model" in initial_df.columns
                and "f1_mean" in initial_df.columns
            ):

                selected_initial = initial_df.loc[
                    initial_df[
                        "f1_mean"
                    ].idxmax()
                ]

                comparison_rows.append({
                    "Stage": "Initial Selection",
                    "Model": selected_initial["model"],
                    "F1 Score": float(
                        selected_initial["f1_mean"]
                    )
                })

            if comparison_rows:

                st.subheader(
                    "Initial vs Current F1"
                )

                lifecycle_f1_df = pd.DataFrame(
                    comparison_rows
                )

                lifecycle_f1_chart = (
                    lifecycle_f1_df[
                        [
                            "Stage",
                            "F1 Score"
                        ]
                    ]
                    .set_index("Stage")
                )

                st.bar_chart(
                    lifecycle_f1_chart
                )

        else:

            st.info(
                "No monitoring history is available yet. "
                "Run main.py to generate current monitoring results."
            )


        st.divider()


        # ========================================================
        # 3. RETRAINED CANDIDATE MODELS
        # ========================================================

        st.subheader(
            "Retrained Candidate Models"
        )

        st.caption(
            "These results are generated only after the retraining "
            "workflow has been approved and candidate_training.py "
            "has trained new candidate models."
        )

        if not candidates_df.empty:

            candidates_display_columns = [
                column
                for column in [
                    "model_name",
                    "mlflow_run_id",
                    "model_path",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ]
                if column in candidates_df.columns
            ]

            candidates_display_df = (
                candidates_df[
                    candidates_display_columns
                ].copy()
            )

            candidates_display_df["stage"] = (
                "Retraining Candidate"
            )

            st.dataframe(
                candidates_display_df,
                use_container_width=True,
                hide_index=True
            )

            candidate_metrics = [
                metric
                for metric in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ]
                if metric in candidates_df.columns
            ]

            if candidate_metrics:

                selected_candidate_metric = st.selectbox(
                    "Select Candidate Evaluation Metric",
                    candidate_metrics,
                    index=(
                        candidate_metrics.index(
                            "f1_score"
                        )
                        if "f1_score" in candidate_metrics
                        else 0
                    ),
                    format_func=lambda value: (
                        value.replace(
                            "_",
                            " "
                        ).title()
                    ),
                    key="candidate_model_comparison_metric"
                )

                if "model_name" in candidates_df.columns:

                    candidate_chart_df = (
                        candidates_df[
                            [
                                "model_name",
                                selected_candidate_metric
                            ]
                        ]
                        .set_index("model_name")
                    )

                    st.bar_chart(
                        candidate_chart_df
                    )

                    best_candidate = candidates_df.loc[
                        candidates_df[
                            selected_candidate_metric
                        ].idxmax()
                    ]

                    st.success(
                        f"Best retrained candidate by "
                        f"{selected_candidate_metric.replace('_', ' ').upper()}: "
                        f"{best_candidate['model_name']} "
                        f"({best_candidate[selected_candidate_metric]:.4f})"
                    )

        else:

            st.info(
                "No retrained candidate model metrics are available yet. "
                "This is expected during the initial project stage."
            )


        st.divider()


        # ========================================================
        # 4. PRODUCTION MODEL
        # ========================================================

        st.subheader(
            "Production Model"
        )

        st.caption(
            "The model currently approved and deployed for production predictions."
        )

        if not production_df.empty:

            production_display_columns = [
                column
                for column in [
                    "model_name",
                    "mlflow_run_id",
                    "model_path",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "model_version",
                    "approval_status",
                    "approved_by"
                ]
                if column in production_df.columns
            ]

            production_display_df = (
                production_df[
                    production_display_columns
                ].copy()
            )

            production_display_df["stage"] = (
                "Production"
            )

            st.dataframe(
                production_display_df,
                use_container_width=True,
                hide_index=True
            )

            production_row = (
                production_df.iloc[0]
            )

            production_metric_columns = [
                column
                for column in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ]
                if column in production_df.columns
            ]

            if production_metric_columns:

                production_metric_cols = st.columns(
                    len(production_metric_columns)
                )

                for metric_col, metric_container in zip(
                    production_metric_columns,
                    production_metric_cols
                ):

                    metric_container.metric(
                        f"Production {metric_col.replace('_', ' ').title()}",
                        f"{float(production_row[metric_col]):.4f}"
                    )

            if "model_name" in production_df.columns:

                st.success(
                    f"Current production model: "
                    f"{production_row['model_name']}"
                )

        else:

            st.info(
                "No production model metrics are available yet."
            )


    except requests.exceptions.RequestException as error:

        st.error(
            f"Unable to retrieve model comparison data: {error}"
        )

    except (
        KeyError,
        ValueError,
        TypeError
    ) as error:

        st.error(
            f"Unable to process model comparison data: {error}"
        )


# Display the Model Comparison interface.
with tab_comparison:
    render_model_comparison()


# ============================================================
# HUMAN MODEL APPROVAL
# ============================================================

@st.fragment(run_every="15s")
def render_human_approval():

    st.header(
        "👤 Human Approval"
    )

    st.caption(
        "Automatically checks for pending approval requests every 15 seconds."
    )

    try:

        # ----------------------------------------------------
        # FETCH PENDING APPROVAL
        # ----------------------------------------------------

        pending_response = requests.get(
            f"{API_URL}/pending-approval",
            timeout=10
        )

        pending_response.raise_for_status()

        pending_data = pending_response.json()


        # ----------------------------------------------------
        # NO PENDING REQUEST
        # ----------------------------------------------------

        if pending_data.get(
            "status"
        ) == "no_pending_request":

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


        # ====================================================
        # RETRAINING APPROVAL
        # ====================================================

        if approval_type == "retraining":

            st.warning(
                "⚠️ Data drift has triggered a retraining approval request."
            )

            st.subheader(
                "🔄 Retraining Approval"
            )

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
                pending_data.get(
                    "created_at",
                    "Unknown"
                )
            )

            st.divider()

            st.info(
                "The incoming batch exceeded the configured drift threshold. "
                "Candidate model training will begin only after human approval."
            )

            st.subheader(
                "Human Decision"
            )

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
                "Approve Retraining":
                    "approve_retraining",
                "Reject Retraining":
                    "reject_retraining"
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
                        "decision":
                            selected_decision,
                        "reviewer":
                            reviewer.strip(),
                        "confirm_lower_f1":
                            False
                    }

                    try:

                        result_response = requests.post(
                            f"{API_URL}/approval",
                            json=approval_payload,
                            timeout=30
                        )

                        result_response.raise_for_status()

                        result = (
                            result_response.json()
                        )

                        if result[
                            "status"
                        ] == "approved":

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

                        elif result[
                            "status"
                        ] == "rejected":

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


        # ====================================================
        # PRODUCTION APPROVAL
        # ====================================================

        st.warning(
            "A trained candidate model is waiting for production approval."
        )

        comparison_response = requests.get(
            f"{API_URL}/model-comparison",
            timeout=10
        )

        comparison_response.raise_for_status()

        comparison_data = (
            comparison_response.json()
        )

        production_df = pd.DataFrame(
            comparison_data.get(
                "production",
                []
            )
        )

        candidates_df = pd.DataFrame(
            comparison_data.get(
                "candidates",
                []
            )
        )


        st.subheader(
            "Production Approval"
        )

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
            pending_data.get(
                "created_at",
                "Unknown"
            )
        )

        st.divider()


        # ----------------------------------------------------
        # CURRENT PRODUCTION MODEL
        # ----------------------------------------------------

        st.subheader(
            "Current Production Model"
        )

        current_production_f1 = None

        if not production_df.empty:

            current_production_f1 = float(
                production_df.iloc[0][
                    "f1_score"
                ]
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


        # ----------------------------------------------------
        # CANDIDATE MODEL COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "Candidate Model Comparison"
        )

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


        # ----------------------------------------------------
        # PRODUCTION APPROVAL DECISION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Select Production Approval Decision"
        )

        decision_options = {
            "Approve Logistic Regression":
                "logistic_regression",

            "Approve Decision Tree":
                "decision_tree",

            "Reject Both Candidates":
                "reject_all"
        }

        selected_label = st.selectbox(
            "Model Approval Decision",
            list(
                decision_options.keys()
            ),
            key="production_approval_decision"
        )

        selected_decision = (
            decision_options[
                selected_label
            ]
        )

        reviewer = st.text_input(
            "Reviewer Name",
            placeholder="Enter your name",
            key="production_approval_reviewer"
        )

        candidate_f1 = None
        lower_f1_confirmation = False


        # ----------------------------------------------------
        # LOWER F1 CONFIRMATION
        # ----------------------------------------------------

        if (
            selected_decision != "reject_all"
            and current_production_f1 is not None
        ):

            selected_model = candidates_df[
                candidates_df[
                    "model_name"
                ] == selected_decision
            ]

            if not selected_model.empty:

                candidate_f1 = float(
                    selected_model.iloc[0][
                        "f1_score"
                    ]
                )

                if candidate_f1 < current_production_f1:

                    st.warning(
                        f"Selected candidate F1-score "
                        f"({candidate_f1:.4f}) is lower than "
                        f"production F1-score "
                        f"({current_production_f1:.4f})."
                    )

                    lower_f1_confirmation = (
                        st.checkbox(
                            "I understand the performance difference "
                            "and still want to approve this model.",
                            key="production_lower_f1_confirmation"
                        )
                    )


        approval_blocked = (
            selected_decision != "reject_all"
            and current_production_f1 is not None
            and candidate_f1 is not None
            and candidate_f1 < current_production_f1
            and not lower_f1_confirmation
        )


        # ----------------------------------------------------
        # SUBMIT PRODUCTION DECISION
        # ----------------------------------------------------

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
                    "decision":
                        selected_decision,
                    "reviewer":
                        reviewer.strip(),
                    "confirm_lower_f1":
                        lower_f1_confirmation
                }

                try:

                    result_response = requests.post(
                        f"{API_URL}/approval",
                        json=approval_payload,
                        timeout=30
                    )

                    result_response.raise_for_status()

                    result = (
                        result_response.json()
                    )

                    if result[
                        "status"
                    ] == "approved":

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

                    elif result[
                        "status"
                    ] == "rejected":

                        st.warning(
                            result["message"]
                        )

                        st.write(
                            "Approval Request ID:",
                            result["request_id"]
                        )

                        st.rerun()

                    elif result[
                        "status"
                    ] == "confirmation_required":

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


# ============================================================
# DISPLAY HUMAN APPROVAL INTERFACE
# ============================================================

with tab_approval:
    render_human_approval()


# ============================================================
# SCRIPT SUMMARY

# MODEL COMPARISON NOTE:
# The Model Comparison function is explicitly called inside
# tab_comparison above. Without that call, Streamlit creates the tab
# but displays no content inside it.
#
# Initial comparison:
# model_selection.py -> logs/model_selection_metrics.csv -> dashboard
#
# Retraining comparison:
# candidate_training.py -> logs/candidate_metrics.csv -> dashboard
#
# Candidate metrics are optional during the initial project setup.

# ============================================================
#
# This Streamlit dashboard provides:
#
# 1. Customer churn prediction through FastAPI.
# 2. Live MLOps monitoring using monitoring_history.csv.
# 3. Incoming batch drift visualization.
# 4. Candidate model performance comparison.
# 5. Production model status.
# 6. Scheduler workflow visualization.
# 7. Scheduler batch progression from V0 to V3.
# 8. Scheduler drift policy visualization:
#       <1%  -> No Action
#       >=1% -> Human Approval
# 9. Scheduler approval activity.
# 10. Human approval for retraining.
# 11. Human approval for production model promotion.
# ============================================================