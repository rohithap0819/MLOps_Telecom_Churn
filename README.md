# MLOps Telecom Churn Prediction

An educational end-to-end MLOps project demonstrating how to build, evaluate, track, monitor, and eventually retrain and deploy a telecom customer churn prediction model.

The project uses a telecom churn dataset to demonstrate reproducible ML workflows, experiment tracking, data drift detection, human-approved retraining, API serving, and automation.

## Project Objectives

- Establish baseline performance using Logistic Regression and Decision Tree.
- Compare models using appropriate classification metrics.
- Track experiments and model artifacts using MLflow.
- Validate and version datasets using Pandera and DVC.
- Monitor incoming data for distribution drift.
- Schedule monitoring and trigger workflows when configured events occur.
- Require explicit human approval before retraining.
- Evaluate candidate models before promotion.
- Serve predictions through FastAPI.
- Containerize the API using Docker.
- Automate testing with pytest and GitHub Actions.

## Technology Stack

- Python, pandas, NumPy
- scikit-learn
- MLflow
- DVC
- Pandera
- Evidently
- FastAPI and Uvicorn
- pytest
- Git and GitHub
- GitHub Actions
- Docker

## Project Workflow

1. Data loading and validation
2. Baseline model training
3. Model comparison and selection
4. Main training and evaluation pipeline
5. Experiment tracking and model artifact logging
6. Batch inference
7. Data drift monitoring
8. Scheduled and event-triggered checks
9. Human approval before retraining
10. Candidate model evaluation and promotion
11. API deployment
12. Automated testing and continuous integration

## Project Structure

```text
MLOps_Telecom_Churn/
├── data/
│   ├── telco_churn_full.csv
│   ├── train.csv
│   ├── current.csv
│   └── current_stress.csv
├── models/
├── logs/
├── src/
├── tests/
├── .github/
│   └── workflows/
├── baseline.py
├── model_selection.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

Additional files will be added as the remaining MLOps components are implemented.

## Dataset

The project currently uses the following datasets:

| Dataset | Rows | Purpose |
|---|---:|---|
| `telco_churn_full.csv` | 7,043 | Full telecom churn dataset |
| `train.csv` | 5,300 | Historical model training data |
| `current.csv` | 943 | Current monitoring batch |
| `current_stress.csv` | 800 | Stress batch for demonstrating feature drift |

The monitoring workflow compares feature distributions against the historical training dataset. The stress batch is used to demonstrate how deliberately different data distributions can trigger drift detection.

## Prerequisites

- Python 3.12
- Git
- A Windows terminal or VS Code terminal
- Internet access for installing dependencies

## Setup

Run these commands from the project root.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

Windows Command Prompt:

```bat
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run the following scripts from the project root with the virtual environment activated.

### 1. Run the baseline models

```bash
python baseline.py
```

This script trains and evaluates Logistic Regression and Decision Tree using a stratified training/holdout split.

### 2. Compare models

```bash
python model_selection.py
```

This script compares Logistic Regression and Decision Tree using stratified cross-validation and selects a candidate based on mean F1-score.

### 3. Run the main pipeline

```bash
python main.py
```

The main pipeline currently performs these stages:

1. Load and validate the training and monitoring datasets.
2. Prepare numerical and categorical features.
3. Split the training data into training and holdout subsets.
4. Train and evaluate the selected model.
5. Log model parameters, metrics, and the model artifact to MLflow.
6. Generate a feature drift report with Evidently.
7. Print a pipeline summary.

The default model is Logistic Regression, and the default monitoring batch is `current.csv`.

To choose the Decision Tree model:

```bash
python main.py --model decision_tree
```

## Data Drift Monitoring

The project includes two monitoring batches to demonstrate different drift outcomes.

### Current batch

Generate the current-batch drift report:

```bash
python main.py --batch current
```

Open the report in your default browser on Windows:

```bat
start logs\drift_report_current.html
```

**Observed result:** 0 of 19 monitored features were detected as drifted in the current batch.

### Stress batch

Generate the stress-batch drift report:

```bash
python main.py --batch stress
```

Open the report:

```bat
start logs\drift_report_stress.html
```

**Observed result:** 16 of 19 monitored features were detected as drifted, corresponding to approximately 84.21% of monitored features.

These results reflect the current datasets and Evidently's configured drift detection settings. Results may change if the datasets, library versions, or detection configuration change.

### Understanding the drift results

The report compares feature distributions in the selected monitoring batch against the historical training data.

- **No drift detected:** The configured detection criteria did not flag the monitored dataset or features.
- **Drift detected:** One or more feature distributions differ sufficiently to cross the configured detection criteria.

Feature drift does not, by itself, prove that model performance has degraded. Model performance should also be evaluated when reliable ground-truth labels become available.

## MLflow Experiment Tracking

The main pipeline logs model parameters, evaluation metrics, and the trained model artifact to MLflow.

The local tracking data is stored in the `mlruns/` directory.

To launch the MLflow interface from the project root:

```bash
python -m mlflow ui --backend-store-uri ./mlruns
```

Then open the address displayed in the terminal, usually:

```text
http://127.0.0.1:5000
```

Use `Ctrl+C` in the terminal to stop the MLflow UI.

## Current Model Evaluation

The most recently completed default run used Logistic Regression with a stratified 80/20 holdout split.

| Metric | Observed value |
|---|---:|
| Accuracy | 0.757 |
| Precision | 0.489 |
| Recall | 0.863 |
| F1-score | 0.624 |
| ROC-AUC | 0.868 |

These are observed results from the current dataset and configuration, not guaranteed results for every run.

## Human Approval and Retraining

Human-approved retraining is a planned component of this project.

The intended workflow is:

1. Detect a configured event, such as feature drift.
2. Record the monitoring result.
3. Request explicit human approval.
4. Stop without retraining if approval is declined.
5. If approved, train and evaluate a candidate model.
6. Compare the candidate with the existing model.
7. Promote the candidate only when the configured evaluation criteria are satisfied.

Retraining, scheduling, and model promotion should not be considered implemented until their respective workflows have been added and tested.

## Development Status

| Component | Status |
|---|---|
| Project setup and dataset preparation | Completed |
| Baseline model training | Implemented and tested |
| Cross-validation model comparison | Implemented and tested |
| Main training and evaluation pipeline | Implemented and tested |
| MLflow model and metric logging | Tested |
| Evidently drift reports | Generated and inspected |
| Batch inference | Planned |
| DVC dataset versioning | Planned |
| Pandera schema validation | Planned |
| Scheduled monitoring | Planned |
| Event-triggered workflow | Planned |
| Human approval before retraining | Planned |
| Candidate model promotion | Planned |
| FastAPI prediction service | Planned |
| Docker deployment | Planned |
| pytest and GitHub Actions CI | Planned |

## Troubleshooting

### Drift report not found

Run the corresponding pipeline command before opening the report:

```bash
python main.py --batch current
```

Then:

```bat
start logs\drift_report_current.html
```

For the stress report, use `--batch stress` and open `logs\drift_report_stress.html`.

### MLflow model logging error

Make sure the `mlflow.sklearn.log_model()` call uses arguments supported by your installed MLflow version. The current implementation uses `artifact_path="model"`.

### Missing dependencies

Activate the project virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

## License

For educational use.
