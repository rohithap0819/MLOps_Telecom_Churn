# MLOps Telecom Churn Prediction

An educational end-to-end MLOps project that demonstrates how to build, evaluate, monitor, retrain, and deploy a telecom customer churn prediction model.

## Project Objectives

- Establish baseline performance using Logistic Regression and Decision Tree.
- Compare models using appropriate classification metrics.
- Track experiments and model versions using MLflow.
- Validate and version datasets using Pandera and DVC.
- Monitor incoming data for distribution drift.
- Schedule monitoring and trigger workflows when configured events occur.
- Require human approval before retraining.
- Serve predictions through FastAPI.
- Containerize the API using Docker.
- Automate testing with pytest and GitHub Actions.

## Technology Stack

- Python and pandas
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
5. Experiment tracking and model registration
6. Batch inference
7. Data drift monitoring
8. Scheduled and event-triggered checks
9. Human approval before retraining
10. Candidate model evaluation and promotion
11. API deployment
12. Automated testing and CI

## Project Status

This project is under development. Commands, implementation details, evaluation results, and expected outputs will be documented as each stage is completed.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## License

For educational use.