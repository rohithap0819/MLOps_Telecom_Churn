# MLOps Telecom Churn Prediction

An end-to-end educational MLOps project that demonstrates how machine learning models can be developed, tracked, monitored, retrained, evaluated, and served as part of a complete machine learning lifecycle.

In this project, you will build a Telecom Customer Churn Prediction system using Logistic Regression and Decision Tree models. You will explore how MLOps tools and practices help manage machine learning applications beyond simply training a model.

The project follows a practical scenario: a telecom company wants to identify customers who are likely to leave its services. As new customer data becomes available, the system monitors changes in the data, trains new candidate models when required, and allows a human reviewer to approve a model before it becomes the production model.

By completing this project, you will understand how individual machine learning components work together to form an end-to-end MLOps pipeline.

---

## 1. Project Overview

### What problem are we solving?

Customer churn occurs when customers stop using a company's products or services.

For a telecom company, losing customers can affect revenue and business growth. Predicting which customers are likely to leave can help the company identify potential churners and take appropriate retention measures.

In this project, we use historical customer information to train machine learning models that predict whether a customer is likely to churn.

However, building a prediction model is only one part of a real-world machine learning application.

We also need to:
- Validate incoming datasets before using them.
- Track experiments and model performance.
- Maintain dataset versions.
- Monitor incoming data for changes.
- Retrain models when required.
- Compare newly trained models with the existing production model.
- Obtain human approval before promoting a new model.
- Serve predictions through an API.
- Provide an interface for users to interact with the application.

These requirements form the foundation of this MLOps project.

## 2. Learning Objectives

By completing this project, you will learn how to:

1. Build and evaluate classification models using Scikit-learn.
2. Compare Logistic Regression and Decision Tree models.
3. Track experiments, metrics, and model artifacts using MLflow.
4. Validate datasets using Pandera.
5. Version datasets using DVC.
6. Automate testing using pytest.
7. Implement Continuous Integration using GitHub Actions.
8. Monitor data drift using Evidently.
9. Implement scheduled and event-triggered workflows.
10. Train and evaluate candidate models.
11. Implement human approval for production model selection.
12. Serve predictions using FastAPI.
13. Build an interactive application using Streamlit.
14. Containerize and run the complete application using Docker.

## 3. Understanding the Project Workflow

The project demonstrates the following machine learning lifecycle:

**Step 1: Data Preparation**

Load the telecom customer dataset and prepare the data for model development.

**Step 2: Model Development**

Train Logistic Regression and Decision Tree models and establish baseline performance.

**Step 3: Model Evaluation**

Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.

**Step 4: Experiment Tracking**

Use MLflow to record experiment parameters, evaluation metrics, and trained model artifacts.

**Step 5: Data Validation and Versioning**

Use Pandera to validate incoming datasets and DVC to track dataset versions.

**Step 6: Model Monitoring**

Use Evidently to compare incoming customer data against historical training data and identify feature drift.

**Step 7: Automated Workflows**

Use scheduled and event-triggered processes to initiate monitoring and candidate model training.

**Step 8: Candidate Model Evaluation**

Train both supported models and compare their performance against the existing production model.

**Step 9: Human Approval**

Allow a human reviewer to approve Logistic Regression, approve Decision Tree, or reject both candidates.

**Step 10: Model Promotion**

Promote an approved candidate to production while retaining the existing production model if candidates are rejected.

**Step 11: Model Serving**

Use FastAPI to expose prediction and model-management endpoints.

**Step 12: User Interface and Containerization**

Use Streamlit to interact with the application and Docker to run the complete system in a single container.

## 4. System Architecture

The following diagram represents the main workflow demonstrated in this project.

```text
                  TELECOM CUSTOMER DATA
                           |
                           v
                  DATA VALIDATION
                       Pandera
                           |
                           v
                    MODEL TRAINING
                           |
                 +---------+---------+
                 |                   |
                 v                   v
           Logistic Regression   Decision Tree
                 |                   |
                 +---------+---------+
                           |
                           v
                  MODEL EVALUATION
                           |
                           v
                   MLflow Tracking
                           |
                           v
                  MODEL MONITORING
                       Evidently
                           |
                           v
                  DRIFT DETECTION
                           |
              +------------+------------+
              |                         |
          No Drift                  Drift Detected
              |                         |
              v                         v
       Continue Monitoring       Candidate Training
                                         |
                                         v
                                  MODEL COMPARISON
                                         |
                                         v
                                   HUMAN APPROVAL
                                         |
                         +---------------+---------------+
                         |               |               |
                         v               v               v
                      Approve          Approve         Reject
                         |               |               |
                         v               v               v
                    Logistic         Decision Tree   Keep Existing
                    Regression                         Production
                         |               |
                         +-------+-------+
                                 |
                                 v
                          MODEL PROMOTION
                                 |
                                 v
                            PRODUCTION
                                 |
                                 v
                              FastAPI
                                 |
                                 v
                             Streamlit
```

The diagram provides a conceptual overview. The actual execution is controlled by the individual scripts and workflows in the repository.

## 5. Technology Stack

| Technology | Purpose in this project |
|---|---|
| Python | Main programming language |
| pandas and NumPy | Data manipulation and processing |
| Scikit-learn | Model training and evaluation |
| MLflow | Experiment tracking and model artifact logging |
| Pandera | Dataset schema validation |
| DVC | Dataset and pipeline versioning |
| Evidently | Data drift detection and monitoring reports |
| FastAPI | Prediction API and model-management endpoints |
| Streamlit | Interactive application interface |
| pytest | Automated testing |
| Git and GitHub | Source code version control |
| GitHub Actions | Continuous Integration |
| Docker | Containerization and application execution |

## 6. Understanding the Dataset

The project uses a Telecom Customer Churn dataset.

The dataset contains customer information such as:

- Customer demographics.
- Customer tenure.
- Phone and internet services.
- Subscription details.
- Contract information.
- Monthly charges.
- Total charges.
- Customer churn status.

The target column is `churn`, which indicates whether a customer has left the telecom service.

### Dataset Files

| Dataset | Rows | Purpose |
|---|---:|---|
| `telco_churn_full.csv` | 7,043 | Complete dataset |
| `train.csv` | 5,300 | Historical training dataset |
| `current.csv` | 943 | Current monitoring dataset |
| `current_stress.csv` | 800 | Dataset used to demonstrate data drift |

Additional incoming and stress datasets are used by the automated monitoring workflows.

The historical training dataset acts as the reference dataset for drift monitoring.

## 7. Project Folder Structure

```text
MLOps_Telecom_Churn/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── telco_churn_full.csv
│   ├── train.csv
│   ├── current.csv
│   ├── current_stress.csv
│   ├── stress_event.csv
│   ├── stress_scheduled.csv
│   └── incoming/
│
├── logs/
│
├── mlruns/
│
├── models/
│   ├── candidates/
│   └── production/
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── dashboard/
│   │   └── app.py
│   │
│   └── data/
│       └── validate.py
│
├── tests/
│   └── test_validation.py
│
├── baseline.py
├── model_selection.py
├── main.py
├── candidate_training.py
├── approval.py
├── monitor.py
├── scheduler.py
├── event_watcher.py
├── config.py
├── split_stress_data.py
│
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

**Important directories:**

- `data/`: Contains historical, current, stress, and incoming datasets.
- `src/`: Contains the API, dashboard, and data validation modules.
- `models/`: Stores candidate and production model artifacts at runtime.
- `logs/`: Stores monitoring reports, candidate metrics, approval history, and other execution records.
- `mlruns/`: Stores local MLflow experiment tracking data.
- `tests/`: Contains automated tests.
- `.github/workflows/`: Contains GitHub Actions workflow definitions.

Some runtime files and model artifacts are generated when you execute the project rather than being committed to GitHub.

## 8. Prerequisites

Before starting, ensure that you have:

- Python 3.12
- Git
- Docker Desktop
- Windows Command Prompt or VS Code terminal
- Internet access for installing Python dependencies

For the Docker workflow, Docker Desktop must be running.

## 9. Project Setup

If you want to explore the individual components locally before running the complete application in Docker, follow these steps.

### Step 1: Clone the repository

```bash
git clone https://github.com/rohithap0819/MLOps_Telecom_Churn.git
```

Navigate into the project:

```bash
cd MLOps_Telecom_Churn
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

For Windows Command Prompt:

```bat
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

You are now ready to execute the individual project components.

---

## 10. Step-by-Step Project Execution

This section explains the individual MLOps components and how to execute them.

### 10.1 Baseline Model Training

**Concept: Establishing a baseline**

A baseline provides an initial performance measurement that can be used to understand how well a model performs before introducing more advanced MLOps workflows.

In this project, we train two classification algorithms:

- Logistic Regression
- Decision Tree

Both models use the same dataset and evaluation approach.

**Run the script:**

```bash
python baseline.py
```

**What happens?**

1. Loads the historical training dataset.
2. Separates input features and the target variable.
3. Prepares numerical and categorical features.
4. Splits the data into training and holdout subsets.
5. Trains both classification models.
6. Evaluates their performance.
7. Displays the evaluation metrics and confusion matrices.

**Metrics covered:**

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### 10.2 Model Comparison Using Cross-Validation

**Concept: Cross-validation**

A single train-test split may not provide enough information about how consistently a model performs.

Cross-validation divides the dataset into multiple folds and evaluates the model across different training and validation combinations.

**Run the script:**

```bash
python model_selection.py
```

**What happens?**

1. Loads the training dataset.
2. Defines Logistic Regression and Decision Tree.
3. Applies stratified five-fold cross-validation.
4. Calculates evaluation metrics for each fold.
5. Compares the mean performance of both models.
6. Identifies the better-performing candidate based on mean F1-score.

This comparison is used to understand model performance. It does not automatically promote a model to production.

### 10.3 Main Training and Evaluation Pipeline

**Concept: Building a repeatable ML pipeline**

A machine learning pipeline organizes data preparation, training, evaluation, and experiment tracking into a repeatable workflow.

**Run the default pipeline:**

```bash
python main.py
```

The default configuration uses Logistic Regression and the current monitoring dataset.

To run Decision Tree:

```bash
python main.py --model decision_tree
```

To use the stress monitoring dataset:

```bash
python main.py --batch stress
```

**What happens?**

1. Loads the training and monitoring datasets.
2. Validates the data.
3. Prepares the input features.
4. Splits the training data into training and holdout subsets.
5. Trains the selected model.
6. Evaluates model performance.
7. Logs experiment parameters and metrics to MLflow.
8. Logs the trained model artifact.
9. Generates a data drift report using Evidently.

### 10.4 Data Validation Using Pandera

**Concept: Data quality validation**

Machine learning models depend on the quality and structure of their input data.

Unexpected columns, missing fields, duplicate customer IDs, or invalid target values can cause errors or produce unreliable results.

Pandera helps define and enforce a data schema before the data enters the pipeline.

**Implementation:**

The validation logic is available in:

```text
src/data/validate.py
```

The project validates incoming datasets against the expected schema.

The validation checks include:

- Required columns.
- Expected data types.
- Valid target values.
- Customer ID uniqueness.
- Nullable fields where appropriate.

Data validation is also used in the monitoring and incoming-data workflows.

### 10.5 Dataset Versioning Using DVC

**Concept: Data version control**

Git is commonly used to track source code, but machine learning projects also depend on datasets.

DVC helps track dataset versions and maintain reproducibility.

In this project, DVC is used alongside Git to manage dataset and pipeline versioning.

**Useful commands:**

Check the current DVC status:

```bash
dvc status
```

View the pipeline dependency graph:

```bash
dvc dag
```

These commands help you understand which data and pipeline dependencies have changed.

### 10.6 Experiment Tracking Using MLflow

**Concept: Experiment tracking**

During model development, different models and configurations produce different results.

MLflow helps record experiments so that model parameters, evaluation metrics, and artifacts can be inspected later.

In this project, MLflow records:

- Model parameters.
- Evaluation metrics.
- Trained model artifacts.
- Experiment run information.

**Start the MLflow interface:**

```bash
python -m mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

Use the interface to explore the recorded runs and compare model performance.

Press `Ctrl+C` in the terminal to stop the MLflow interface.

### 10.7 Data Drift Monitoring Using Evidently

**Concept: Data drift**

Data drift occurs when the distribution of incoming data changes compared with the reference data used during model development.

For example, customer behavior, subscription patterns, and monthly charges may change over time.

If the incoming data differs significantly from the training data, the model may need to be investigated and potentially retrained.

This project uses Evidently to generate data drift reports.

**Monitor the current dataset:**

```bash
python main.py --batch current
```

Open the generated report:

```bat
start logs\drift_report_current.html
```

**Monitor the stress dataset:**

```bash
python main.py --batch stress
```

Open:

```bat
start logs\drift_report_stress.html
```

**Observed results from the current configuration:**

| Monitoring dataset | Drifted features | Drift percentage |
|---|---:|---:|
| Current batch | 0 of 19 | 0% |
| Stress batch | 16 of 19 | 84.21% |

These are observed results from the current datasets and configuration.

**Important:** Data drift indicates changes in feature distributions. It does not automatically prove that prediction performance has decreased.

### 10.8 Candidate Model Training

**Concept: Candidate models**

When retraining is required, the system trains both supported algorithms and compares their performance.

**Run:**

```bash
python candidate_training.py
```

**What happens?**

1. Loads the training dataset.
2. Trains Logistic Regression.
3. Trains Decision Tree.
4. Evaluates both models.
5. Saves candidate model artifacts.
6. Records candidate metrics.
7. Logs the experiments to MLflow.
8. Compares candidate performance with the existing production model when one is available.

Candidate models are stored under:

```text
models/candidates/
```

Candidate training does not automatically promote a model.

### 10.9 Scheduled and Event-Triggered Workflows

**Concept: Workflow automation**

A monitoring workflow should not always require someone to execute a script manually.

This project demonstrates two ways to initiate automated workflows.

**Scheduled execution**

The scheduler periodically triggers candidate training using the configured scheduled dataset.

Run locally:

```bash
python scheduler.py
```

The current scheduler interval is 24 hours.

**Event-triggered execution**

The event watcher monitors:

```text
data/incoming/
```

When a new CSV file is added, the watcher:

1. Detects the incoming file.
2. Waits for the file copy to complete.
3. Validates the dataset.
4. Checks for data drift.
5. Initiates candidate training if the configured drift threshold is exceeded.

Run locally:

```bash
python event_watcher.py
```

To demonstrate the event workflow, copy a test dataset into the incoming directory using a new filename.

For example:

```bat
copy data\stress_event.csv data\incoming\event_test.csv
```

The incoming dataset is then processed by the event watcher.

### 10.10 Human Approval and Model Promotion

**Concept: Human-in-the-loop MLOps**

Automatically training a model does not mean that it should automatically become the production model.

A newly trained model must be evaluated before it replaces the existing production model.

This project introduces a human approval mechanism.

The reviewer can choose:

1. Approve Logistic Regression.
2. Approve Decision Tree.
3. Reject both candidates.

The system compares candidate performance with the current production model and warns when a candidate has a lower F1-score.

If a candidate is approved, the system promotes it and updates the production model registry.

If both candidates are rejected, the existing production model remains unchanged.

Approval history is recorded for tracking decisions.

**Important:** Candidate training and production promotion are separate operations. Training does not automatically replace the production model.

### 10.11 FastAPI Prediction Service

**Concept: Model serving**

After a model has been approved, it can be used to generate predictions through an API.

FastAPI exposes endpoints for interacting with the MLOps system.

The API implementation is available in:

```text
src/api/main.py
```

The main endpoints include:

| Endpoint | Purpose |
|---|---|
| `GET /` | API information |
| `GET /health` | Service health |
| `GET /model-info` | Current production model information |
| `GET /model-comparison` | Candidate and production metrics |
| `POST /approval` | Submit a human approval decision |
| `POST /predict` | Generate a churn prediction |

When running locally, start FastAPI using:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Open the interactive API documentation:

```text
http://127.0.0.1:8001/docs
```

You can use the Swagger interface to explore the endpoints and submit prediction requests.

### 10.12 Streamlit Dashboard

**Concept: Interactive ML application**

Streamlit provides a graphical interface for interacting with the MLOps system.

The dashboard connects to the FastAPI backend and provides four main sections:

**Customer Prediction**

Submit customer information and view the churn prediction and probability.

**Model Monitoring**

Explore monitoring results and drift reports.

**Model Comparison**

Compare the performance of the candidate models and the existing production model.

**Human Approval**

Review candidate models and submit approval or rejection decisions.

The dashboard implementation is available in:

```text
src/dashboard/app.py
```

The dashboard is included in the complete Docker application.

---

## 11. Automated Testing and Continuous Integration

**Concept: Testing and CI**

Machine learning projects require testing to ensure that changes do not break existing functionality.

This project uses pytest for automated testing and GitHub Actions for Continuous Integration.

The current test suite includes data validation tests such as:

- Valid dataset passes validation.
- Duplicate customer IDs are rejected.
- Invalid churn values are rejected.
- Missing required columns are rejected.

**Run tests locally:**

```bash
python -m pytest -v
```

The GitHub Actions workflow is located at:

```text
.github/workflows/ci.yml
```

When code changes are pushed to GitHub, the workflow installs dependencies and runs the automated tests.

---

## 12. Run the Complete Project Using Docker

**Concept: Containerization**

So far, we have explored the individual components of the MLOps workflow.

Docker allows us to package the application and its dependencies into a container so that the complete application can be started consistently.

This project uses a single Docker container that runs:

- FastAPI backend.
- Streamlit dashboard.
- Incoming data event watcher.
- Scheduled candidate training.

Docker Compose manages the application startup and persistent data volumes.

### Step 1: Start Docker Desktop

Ensure Docker Desktop is running.

### Step 2: Build and start the application

From the project root, execute:

```bash
docker compose up --build
```

This builds the Docker image and starts the application.

The initial startup also generates candidate models if the required candidate artifacts are not available.

### Step 3: Open the application

Once the services have started, access the following addresses.

| Application | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI | http://localhost:8001 |
| FastAPI Swagger Documentation | http://localhost:8001/docs |

### Step 4: Stop the application

Press:

```text
Ctrl+C
```

Alternatively, from another terminal:

```bash
docker compose down
```

The named Docker volumes preserve model, log, and MLflow runtime data when the container is stopped without removing the volumes.

### Useful Docker commands

Check running containers:

```bash
docker ps
```

View application logs:

```bash
docker logs telecom-churn-mlops
```

Follow application logs:

```bash
docker logs -f telecom-churn-mlops
```

Restart the application:

```bash
docker compose restart
```

Stop the application:

```bash
docker compose down
```

---

## 13. Suggested Learner Activities

After understanding the complete workflow, try the following activities.

**Activity 1: Compare the models**

Run the baseline and cross-validation scripts. Compare Logistic Regression and Decision Tree using different evaluation metrics.

**Activity 2: Explore MLflow**

Run the main training pipeline multiple times and inspect the experiment runs in MLflow.

**Activity 3: Validate incoming data**

Introduce an invalid value or remove a required column from a test dataset. Observe how Pandera handles the validation error.

**Activity 4: Demonstrate data drift**

Use the stress dataset and compare its drift report with the current dataset.

**Activity 5: Trigger candidate training**

Place a new CSV file into the incoming data directory and observe the event-triggered workflow.

**Activity 6: Test human approval**

Reject both candidate models and observe whether the existing production model remains active.

Then approve a candidate and inspect the updated production model information.

**Activity 7: Generate predictions**

Use the Streamlit dashboard or FastAPI Swagger documentation to submit customer information and observe the prediction.

**Activity 8: Explore Docker**

Start the complete application using Docker Compose and explore the dashboard, API, monitoring, and approval workflow.

---

## 14. Expected Learning Outcomes

After completing this project, you should be able to explain:

- Why machine learning projects require MLOps practices.
- How datasets and experiments can be versioned and tracked.
- How data validation helps maintain data quality.
- How model performance is evaluated.
- How data drift can be detected.
- How automated workflows can trigger candidate training.
- Why candidate models should be evaluated before promotion.
- Why human approval is useful in production model management.
- How a trained model can be exposed through an API.
- How an interactive dashboard can communicate with a backend service.
- How Docker simplifies application execution.

## 15. Conclusion

This project demonstrates how a machine learning application can move beyond model development into a complete MLOps workflow.

Starting with a telecom customer churn prediction problem, you explored model training, experiment tracking, dataset versioning, validation, monitoring, automated workflows, human approval, model serving, testing, and containerization.

The final application brings these components together into one system that can be started using Docker Compose.

The objective is not just to build a churn prediction model, but to understand how machine learning applications can be managed throughout their lifecycle.

---

## License

For educational use.
