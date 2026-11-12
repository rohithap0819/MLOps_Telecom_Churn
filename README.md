# MLOps Telecom Churn Prediction

An educational end-to-end MLOps project for predicting telecom customer churn and demonstrating model training, model selection, monitoring, data ingestion, retraining, human approval, API deployment, testing, and Docker.

> **Note:** This project is designed for learning and classroom demonstrations. It is not intended for production use.

---

## 1. Project Overview

The project predicts whether a telecom customer is likely to churn.

Instead of only training an ML model, the project demonstrates an MLOps lifecycle:

```text
Training Data
      |
      v
Data Validation
      |
      v
Model Training
      |
      v
Model Selection
      |
      v
Monitoring
      |
      v
New Incoming Data
      |
      v
Drift Detection
      |
      v
Retraining Decision
      |
      v
Human Approval
      |
      v
Candidate Model
      |
      v
Production Approval
```

---

## 2. What This Project Demonstrates

- Data validation using Pandera
- Logistic Regression and Decision Tree
- Model comparison and selection
- Cross-validation
- MLflow experiment tracking
- Data drift monitoring using Evidently
- DVC-based dataset versioning
- Event-driven data processing
- Scheduled data processing
- Candidate model retraining
- Human approval
- FastAPI backend
- Streamlit dashboard
- Pytest
- GitHub Actions
- Docker

---

## 3. Project Structure

```text
MLOps_Telecom_Churn/
│
├── .github/
│   └── workflows/
│
├── data/
│   ├── train.csv
│   └── incoming/
│
├── models/
│   ├── candidates/
│   └── production/
│
├── logs/
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
│
├── approval.py
├── baseline.py
├── candidate_training.py
├── config.py
├── event_watcher.py
├── main.py
├── model_selection.py
├── monitor.py
├── scheduler.py
├── simulate_incoming.py
├── split_stress_data.py
├── update_training_data.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Main Python Files

| File | Purpose |
|---|---|
| `baseline.py` | Trains the initial Logistic Regression and Decision Tree models |
| `model_selection.py` | Compares models using cross-validation and selects the best model |
| `main.py` | Runs monitoring using the selected model |
| `candidate_training.py` | Trains candidate models when retraining is required |
| `approval.py` | Handles human approval and rejection workflow |
| `event_watcher.py` | Detects incoming CSV files and starts the event-driven workflow |
| `scheduler.py` | Checks for incoming data at scheduled intervals |
| `simulate_incoming.py` | Simulates incoming data for the Event Watcher demo |
| `update_training_data.py` | Updates the training dataset with incoming customer records |
| `monitor.py` | Supports monitoring-related operations |
| `config.py` | Stores project configuration |
| `src/data/validate.py` | Validates datasets using Pandera |
| `src/api/main.py` | FastAPI backend |
| `src/api/schemas.py` | API request/response schemas |
| `src/dashboard/app.py` | Streamlit approval dashboard |

---

## 5. Machine Learning Workflow

The project uses two classification models:

- Logistic Regression
- Decision Tree

The original V0 dataset contains:

```text
Rows    : 5300
Columns : 21
Churn   : approximately 23.4%
```

The baseline evaluation is followed by 5-fold Stratified Cross-Validation for model selection.

### Model Selection

The main selection metric is:

```text
F1 Score
```

Observed V0 cross-validation results:

| Model | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.599 | 0.480 | 0.797 | 0.854 |
| Decision Tree | 0.584 | 0.457 | 0.809 | 0.832 |

Therefore, Logistic Regression is selected as the preferred model for the V0 workflow.

---

## 6. Model Selection vs Monitoring

These are two different stages.

### `model_selection.py`

Answers:

> Which model should be selected?

```text
Logistic Regression
        |
Decision Tree
        |
        v
Model Comparison
        |
        v
Best Model Selected
```

### `main.py`

Answers:

> How is the selected model behaving on current data?

```text
Selected Model
      |
      v
Current Data
      |
      v
Predictions
      |
      v
Metrics + Drift Monitoring
```

---

## 7. MLflow

MLflow is used for experiment tracking.

It records model parameters, metrics, and experiment runs.

Start MLflow with:

```bash
python -m mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

---

## 8. Monitoring

`main.py` runs the monitoring workflow using the selected model.

It:

1. Loads the selected model.
2. Loads reference and current data.
3. Validates the data.
4. Generates predictions.
5. Calculates metrics.
6. Logs metrics to MLflow.
7. Generates an Evidently drift report.

Example monitoring metrics from a project run:

```text
Accuracy  : 0.7519
Precision : 0.4814
Recall    : 0.8227
F1 Score  : 0.6074
ROC-AUC   : 0.8465
```

---

## 9. Data Drift

Evidently compares reference data with current data to identify changes in feature distributions.

```text
Reference Data
      |
      v
Current Data
      |
      v
Drift Detection
      |
      v
Policy Decision
```

Important:

```text
Data Drift != Model Performance Degradation
```

Drift indicates that the input data distribution has changed. Actual model performance requires labelled outcomes.

---

## 10. Event Watcher

`event_watcher.py` implements the event-driven workflow.

```text
Incoming CSV
      |
      v
Event Detected
      |
      v
Data Validation
      |
      v
Update Training Data
      |
      v
DVC
      |
      v
Drift Detection
      |
      v
Policy Decision
```

### Run Event Watcher

```bash
python event_watcher.py
```

For the classroom demonstration, the incoming data can be generated using:

```bash
python simulate_incoming.py
```

The simulator creates incoming CSV data which the Event Watcher detects.

---

## 11. Scheduler

`scheduler.py` implements scheduled data processing.

For the classroom demonstration, the Scheduler checks for incoming data approximately every **1 minute**.

### Run Scheduler

```bash
python scheduler.py
```

Then manually place a CSV file inside:

```text
data/incoming/
```

Example:

```text
data/incoming/batch_1.csv
```

The Scheduler detects and processes the file during its next check.

---

## 12. Event Watcher vs Scheduler

| Feature | Event Watcher | Scheduler |
|---|---|---|
| Trigger | File event | Scheduled check |
| Demo input | Simulator | Manually placed CSV |
| Simulator required | Yes | No |
| Data validation | Yes | Yes |
| Data update | Yes | Yes |
| Drift check | Yes | Yes |
| Retraining policy | Yes | Yes |

**Do not run the Event Watcher and Scheduler simultaneously during the demonstration.**

### Event Watcher Demo

```bash
python event_watcher.py
```

and:

```bash
python simulate_incoming.py
```

### Scheduler Demo

```bash
python scheduler.py
```

Then manually place the CSV in:

```text
data/incoming/
```

---

## 13. Retraining Policy

### Event Watcher

```text
Drift < 1%
    |
    v
No Action

1% - 50%
    |
    v
Human Retraining Approval

> 50%
    |
    v
Automatic Candidate Retraining
```

### Scheduler

```text
Drift < 1%
    |
    v
No Action

Drift >= 1%
    |
    v
Human Retraining Approval
```

---

## 14. Candidate and Production Models

A retrained model first becomes a **candidate model**.

It does not automatically become the production model.

```text
Retraining
    |
    v
Candidate Model
    |
    v
Production Approval
    |
    v
Production Model
```

This demonstrates human-in-the-loop model governance.

---

## 15. FastAPI

FastAPI provides the backend API used by the project.

Start FastAPI:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Swagger documentation:

```text
http://127.0.0.1:8001/docs
```

---

## 16. Streamlit

Streamlit provides the user interface for the human approval workflow.

Start Streamlit:

```bash
python -m streamlit run src/dashboard/app.py
```

Open:

```text
http://localhost:8501
```

For the human approval demonstration, both FastAPI and Streamlit should be running.

```text
User
 |
 v
Streamlit
 |
 v
FastAPI
 |
 v
Approval / Model Workflow
```

---

## 17. Testing

The project uses Pytest.

Run:

```bash
python -m pytest -v
```

Tests are located in:

```text
tests/
```

---

## 18. GitHub Actions

GitHub Actions is used for Continuous Integration.

The workflow runs automated checks such as tests when changes are pushed to the repository.

```text
git push
    |
    v
GitHub Actions
    |
    v
Install Dependencies
    |
    v
Run Tests
    |
    v
Pass / Fail
```

---

## 19. Docker

The project includes Docker configuration for running the application services.

Build and start the services with:

```bash
docker compose up --build
```

Typical services:

```text
Streamlit
http://localhost:8501

FastAPI
http://localhost:8001

FastAPI Swagger
http://localhost:8001/docs
```

---

# 20. Setup from ZIP

## Step 1: Extract the Project

Extract the ZIP to a convenient location.

Example:

```text
C:\Users\<username>\Desktop\MLOps_Telecom_Churn
```

## Step 2: Open Terminal

```bash
cd C:\Users\<username>\Desktop\MLOps_Telecom_Churn
```

## Step 3: Check Python

Python 3.12 is recommended.

```bash
python --version
```

## Step 4: Create Virtual Environment

```bash
python -m venv .venv
```

## Step 5: Activate Virtual Environment

```bash
.venv\Scripts\activate
```

## Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 7: Run Tests

```bash
python -m pytest -v
```

---

# 21. Recommended Execution Order

For a complete classroom demonstration:

### 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Test the Project

```bash
python -m pytest -v
```

### 3. Train Baseline Models

```bash
python baseline.py
```

### 4. Select the Best Model

```bash
python model_selection.py
```

### 5. Start MLflow

```bash
python -m mlflow ui --backend-store-uri ./mlruns
```

### 6. Run Monitoring

```bash
python main.py
```

### 7. Start FastAPI

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

### 8. Start Streamlit

```bash
python -m streamlit run src/dashboard/app.py
```

### 9. Demonstrate Event Watcher

```bash
python event_watcher.py
```

In another terminal:

```bash
python simulate_incoming.py
```

### 10. Demonstrate Scheduler

Stop the Event Watcher and simulator first.

Then:

```bash
python scheduler.py
```

Manually place a CSV inside:

```text
data/incoming/
```

### 11. Demonstrate Human Approval

Use the Streamlit dashboard to review and approve/reject the retraining workflow.

---

# 22. Important URLs

| Service | URL |
|---|---|
| MLflow | http://127.0.0.1:5000 |
| FastAPI | http://127.0.0.1:8001 |
| FastAPI Swagger | http://127.0.0.1:8001/docs |
| Streamlit | http://localhost:8501 |

---

# 23. Troubleshooting

### Missing Python Package

```bash
pip install -r requirements.txt
```

### Streamlit Not Found

Use:

```bash
python -m streamlit run src/dashboard/app.py
```

### MLflow Not Found

Use:

```bash
python -m mlflow ui --backend-store-uri ./mlruns
```

### DVC Not Found

Use:

```bash
python -m dvc
```

### FastAPI Not Starting

Make sure the terminal is opened from the project root and run:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

### Scheduler Does Not Process Immediately

The Scheduler checks periodically. For the classroom version, the interval is approximately 1 minute.

### No New Customer Records

If the incoming CSV contains customer IDs that already exist in the training dataset, no new records may be added. This is expected.

---

# 24. Key MLOps Concepts

This project demonstrates:

- Model training
- Model selection
- Cross-validation
- Experiment tracking
- Data validation
- Data versioning
- Data drift monitoring
- Event-driven processing
- Scheduled processing
- Retraining
- Human-in-the-loop approval
- Candidate model management
- Production model approval
- API deployment
- Dashboard-based interaction
- Automated testing
- Continuous Integration
- Containerization

---

# 25. Important Notes

- This is an educational project.
- Event Watcher and Scheduler are separate demonstrations.
- The Event Watcher demonstration uses the incoming-data simulator.
- The Scheduler demonstration uses a manually placed CSV.
- FastAPI and Streamlit should be running for the human approval demonstration.
- Data drift does not automatically mean model performance has degraded.
- A candidate model does not automatically become the production model.
- Runtime-generated logs, MLflow data, model artifacts, and temporary incoming files should not normally be committed to Git.

---

# 26. Quick Command Reference

| Task | Command |
|---|---|
| Activate environment | `.venv\Scripts\activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Run tests | `python -m pytest -v` |
| Train baseline | `python baseline.py` |
| Model selection | `python model_selection.py` |
| Monitoring | `python main.py` |
| MLflow | `python -m mlflow ui --backend-store-uri ./mlruns` |
| FastAPI | `uvicorn src.api.main:app --host 127.0.0.1 --port 8001` |
| Streamlit | `python -m streamlit run src/dashboard/app.py` |
| Event Watcher | `python event_watcher.py` |
| Simulator | `python simulate_incoming.py` |
| Scheduler | `python scheduler.py` |
| Docker | `docker compose up --build` |

---

# 27. Final Workflow

The complete project lifecycle is:

```text
                    TELECOM CHURN MLOPS

                         Training Data
                              |
                              v
                       Data Validation
                              |
                              v
                       Model Training
                              |
                              v
                       Model Selection
                              |
                              v
                       MLflow Tracking
                              |
                              v
                          Monitoring
                              |
                              v
                      Incoming New Data
                              |
                    +---------+---------+
                    |                   |
                    v                   v
              Event Watcher         Scheduler
                    |                   |
                    +---------+---------+
                              |
                              v
                       Data Validation
                              |
                              v
                    Update Training Data
                              |
                              v
                       Drift Detection
                              |
                              v
                        Policy Decision
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                 No Action          Retraining
                                        |
                                        v
                                Human Approval
                                        |
                                        v
                                Candidate Model
                                        |
                                        v
                              Production Approval
                                        |
                                        v
                               Production Model
```

This project provides a complete educational example of how machine learning can be managed as an operational lifecycle rather than as a one-time model training exercise.