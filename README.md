# MLOps Telecom Churn Prediction
An educational end-to-end MLOps project for predicting telecom customer churn and demonstrating model training, model selection, monitoring, data ingestion, drift detection, retraining, human approval, API deployment, testing, CI, DVC dataset versioning, and Docker.



> ****Note:**** This project is designed for learning and classroom demonstrations. It is not intended for production use.



---
## 1. Project Overview
The project predicts whether a telecom customer is likely to churn.



Instead of only training an ML model, the project demonstrates an MLOps lifecycle:



```text

Project_setup

    |

    v

Data Validation

    |

    v

Baseline Model Training

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

    +------------------------+

    |                        |

    v                        v

Event Watcher            Scheduler

    |                        |

    +-----------+------------+

                |

                v

        Data Validation

                |

                v

        Update Training Data

                |

                v

         DVC Versioning

                |

                v

         Drift Detection

                |

                v

         Policy Decision

                |

        +-------+-------+

        |               |

        v               v

     No Action       Retraining

                        |

                        v

                Human Approval

                        |

                        v

                Candidate Model

                        |

                        v

                Model Approval

                        |

                        v

                Production Model

```



---
## 2. What This Project Demonstrates
- Data validation using Pandera

- Logistic Regression and Decision Tree

- Model comparison and selection

- Stratified cross-validation

- MLflow experiment tracking

- Data drift monitoring using Evidently

- DVC-based dataset versioning

- Event-driven data processing

- Scheduled data processing

- Candidate model retraining

- Human approval for retraining

- Human approval for candidate models

- FastAPI backend

- Streamlit dashboard

- Pytest

- GitHub Actions

- Docker

- Reproducible classroom demonstrations



---
## 3. Project Structure
```text

MLOps_Telecom_Churn/

├── src/

│   ├── \_\_init\_\_.py

│   ├── main.py

│   ├── fastapi_schema.py

│   └── streamlit_app.py

│

├── validate_pipeline.py

│

├── data/

│   ├── validation.py

│   └── data_sources/

│       ├── train.csv

│       ├── current.csv

│       ├── current_stress.csv

│       ├── telco_churn_full.csv

│       ├── event_watcher_data/

│       │   ├── train_trigger.csv

│       │   └── train_trigger.csv.dvc

│       ├── scheduler_data/

│       │   ├── train_scheduler.csv

│       │   └── train_scheduler.csv.dvc

│       ├── incoming/

│       └── incoming_batches/

│

├── modelling/

│   ├── train.py

│   └── model_selection.py

│

├── deployment/

│   ├── monitoring.py

│   ├── drift_monitor_and_trigger.py

│   ├── scheduler.py

│   ├── data_simulator_source.py

│   ├── data_simulator_stream.py

│   ├── data_simulator_version.py

│   ├── model_human_approval.py

│   ├── retraining.py

│   ├── config.py

│   └── docker/

│       ├── Dockerfile

│       ├── docker-compose.yml

│       ├── start.sh

│       └── .dockerignore

│── project_setup.py

├── models/

├── logs/

├── .dvc/

├── mlruns/

├── requirements.txt

├── README.md

├── .gitignore

├── .dvcignore

└── .github/

    └── workflows/

        └── ci.yml

```



### Important data locations
All project CSV files used by the workflows are under:



```text

data/data_sources/

```



The Event Watcher uses:



```text

data/data_sources/event_watcher_data/train_trigger.csv

```



The Scheduler uses:



```text

data/data_sources/scheduler_data/train_scheduler.csv

```



Incoming files are generated under:



```text

data/data_sources/incoming/

```



Generated incoming batches are stored under:



```text

data/data_sources/incoming_batches/

```



---
## 3.1 Initial Project Setup

The project includes `setup_project.py` so learners can prepare the local Git/DVC environment without manually performing the initial Git/DVC initialization steps.

Run:

```cmd
python setup_project.py
```

The script supports both:

- **GitHub clone:** preserves the existing repository and DVC configuration.
- **ZIP download:** initializes the local Git/DVC environment and adds the baseline dataset to DVC when required.

After setup, continue with validation, baseline training, model selection, MLflow, FastAPI/Streamlit, Event Watcher, Scheduler, and Docker.

---

## 4. Main Python Files
| File | Purpose |

|---|---|

| `modelling/train.py` | Trains the baseline model used by the project |

| `modelling/model_selection.py` | Compares candidate models using cross-validation and selects the preferred model |

| `deployment/monitoring.py` | Supports monitoring-related operations |

| `deployment/drift_monitor_and_trigger.py` | Implements the Event Watcher workflow, including incoming-data processing, drift detection, policy decisions, and retraining workflow |

| `deployment/scheduler.py` | Checks for incoming data on a scheduled interval and runs the Scheduler workflow |

| `deployment/data_simulator_source.py` | Prepares the datasets required for the Event Watcher and Scheduler demonstrations |

| `deployment/data_simulator_stream.py` | Generates incoming data for the demonstrations |

| `deployment/data_simulator_version.py` | Supports the data-versioning demonstration |

| `deployment/model_human_approval.py` | Handles human approval decisions |

| `deployment/retraining.py` | Handles candidate model retraining |

| `deployment/config.py` | Stores project configuration |

| `data/validation.py` | Validates datasets |

| `src/main.py` | FastAPI backend and monitoring-related API endpoints |

| `src/fastapi_schema.py` | FastAPI request/response schemas |

| `src/streamlit_app.py` | Streamlit dashboard for monitoring and approval workflow |

| `validate_pipeline.py` | Pytest validation checks for the project |
| `setup_project.py` | Checks Git/DVC and prepares the local Git/DVC environment for GitHub clones or ZIP-based setups |



---
## 5. Machine Learning Workflow
The project uses two classification models:



- Logistic Regression

- Decision Tree



The original V0 training dataset contains:



```text

Rows    : 5300

Columns : 21

```



The baseline training and model-selection workflow is designed to establish the initial model used by the rest of the MLOps demonstration.



### Model Selection
The project uses cross-validation for model comparison.



The primary model-selection metric is:



```text

F1 Score

```



The model-selection workflow evaluates the candidate models and stores the metrics used by the Streamlit dashboard.



---
## 6. Model Selection vs Monitoring
These are two different stages.



### `model_selection.py`
Answers:



> Which model should be selected?



```text

Logistic Regression

        |

        v

Decision Tree

        |

        v

Model Comparison

        |

        v

Best Model Selected

```



### `src/main.py`
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



Model selection chooses the model.



Monitoring evaluates the selected model and current data during the MLOps workflow.



---
## 7. MLflow
MLflow is used for experiment tracking.



It records model parameters, metrics, and experiment runs.



Start the MLflow UI with:



```cmd

mlflow ui --host 127.0.0.1 --port 5000

```



Open:



```text

http://localhost:5000

```



Keep the MLflow terminal running while viewing the UI.



---
## 8. Monitoring
The project includes monitoring for the selected model and current data.



The monitoring workflow can:



1. Load the selected model.

2. Load reference and current data.

3. Validate the data.

4. Generate predictions.

5. Calculate metrics.

6. Log metrics to MLflow.

7. Generate an Evidently drift report.



Monitoring results are also surfaced through the project dashboard and logs.



---
## 9. Data Drift
Evidently is used to compare reference data with current/incoming data and identify changes in feature distributions.



```text

Reference Data

      |

      v

Current / Incoming Data

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



The Scheduler and Event Watcher use their configured drift policies to decide whether retraining action is required.



---
## 10. Event Watcher
The Event Watcher implements the event-driven workflow.



The main script is:



```text

deployment/drift_monitor_and_trigger.py

```



Its workflow is:



```text

Incoming CSV

      |

      v

Event Watcher Detects Data

      |

      v

Data Validation

      |

      v

Update Training Data

      |

      v

DVC Versioning

      |

      v

Drift Detection

      |

      v

Policy Decision

      |

      v

Human Approval / Retraining Workflow

```



### Start the Event Watcher
Open a new VS Code terminal and run:



```cmd

python deployment\drift_monitor_and_trigger.py

```



Keep this terminal running.



### Start the Event Watcher data simulator
Open another VS Code terminal and run:



```cmd

python deployment\data_simulator_stream.py

```



The simulator generates incoming data for the Event Watcher.



Observe:



- Event Watcher terminal

- Streamlit dashboard

- Generated logs

- Drift reports

- Human approval workflow



### Event Watcher workflow
```text

Incoming data

    |

    v

Event Watcher detects the data

    |

    v

Data processing

    |

    v

Drift detection

    |

    v

Retraining condition

    |

    v

Human approval

    |

    v

Candidate model training

    |

    v

Model evaluation / approval

    |

    v

Monitoring

```



---
## 11. Scheduler
The Scheduler implements scheduled data processing.



The main script is:



```text

deployment/scheduler.py

```



For the classroom demonstration, the Scheduler runs on a configured ****1-minute cycle****.



### Start the Scheduler
Open a new VS Code terminal and run:



```cmd

python deployment\scheduler.py

```



Keep this terminal running.



### Start the Scheduler data simulator
Open another VS Code terminal and run:



```cmd

python deployment\data_simulator_stream.py

```



Observe the Scheduler terminal and Streamlit dashboard.



### Important: Wait for the Scheduler cycle
The Scheduler runs on a 1-minute cycle.



After each approval/retraining action, ****wait for the next Scheduler cycle**** before expecting the next stage.



Do not immediately assume that the workflow is stuck if the next action does not appear.



Typical approval flow:



```text

Approve Retraining

        |

        v

WAIT \~1 MINUTE

        |

        v

Candidate Model Training / Model Approval

        |

        v

Approve Model

        |

        v

WAIT \~1 MINUTE

        |

        v

Next Drift Check

        |

        v

Approve Retraining

        |

        v

WAIT \~1 MINUTE

        |

        v

Model Approval

        |

        v

WAIT \~1 MINUTE

        |

        v

Next Scheduler Cycle

```



---
## 12. Event Watcher vs Scheduler
| Feature | Event Watcher | Scheduler |

|---|---|---|

| Trigger | Incoming file event | Scheduled check |

| Main script | `deployment/drift_monitor_and_trigger.py` | `deployment/scheduler.py` |

| Demo data | Data simulator | Data simulator |

| Processing | Event-driven | Scheduled |

| Data validation | Yes | Yes |

| Data update | Yes | Yes |

| DVC versioning | Yes | Yes |

| Drift check | Yes | Yes |

| Retraining policy | Yes | Yes |

| Human approval | Yes | Yes |



### Important
****Do not run the Event Watcher and Scheduler simultaneously during the same classroom demonstration.****



Complete one workflow, clean its generated incoming state, and then start the other workflow.



---
## 13. Retraining Policy
The Event Watcher and Scheduler demonstrate human-in-the-loop retraining decisions.



### Event Watcher
The Event Watcher uses the configured drift policy to determine whether a retraining action is required.



Conceptually:



```text

Drift detected

      |

      v

Policy Decision

      |

      +-------------------+

      |                   |

      v                   v

No Retraining        Retraining Required

                          |

                          v

                  Human Approval

```



### Scheduler
The Scheduler uses its configured policy for drift-triggered retraining.



```text

Drift < configured threshold

          |

          v

      No Action



Drift >= configured threshold

          |

          v

Human Retraining Approval

```



The project deliberately demonstrates human approval before the retraining workflow continues.



---
## 14. Candidate and Production Models
A retrained model first becomes a ****candidate model****.



It does not automatically become the production model.



```text

Retraining

    |

    v

Candidate Model

    |

    v

Human / Production Approval

    |

    v

Production Model

```



This demonstrates human-in-the-loop model governance.



---
## 15. Human Approval
Human approval is surfaced through the Streamlit dashboard.



The workflow can require approval for:



- Retraining

- Candidate model / production promotion



A rejection does not automatically remove the dataset version or lineage created during the workflow.



The dashboard is used to observe and interact with the approval state.



---
## 16. FastAPI
FastAPI provides the backend API used by the project.



Start FastAPI from the project root:



```cmd

uvicorn src.main:app --reload

```



Open the Swagger documentation:



```text

http://localhost:8001/docs

```



The API is also used by the Streamlit application where applicable.



---
## 17. Streamlit
Streamlit provides the user interface for the MLOps workflow.



Start Streamlit from the project root:



```cmd

streamlit run src/streamlit_app.py

```



Open:



```text

http://localhost:8501

```



The dashboard can surface:



- Model-selection metrics

- Monitoring information

- Drift information

- Retraining approval requests

- Candidate model approval information

- Workflow state



For the human approval demonstration, FastAPI and Streamlit should be running.



---
## 18. Testing
The project uses Pytest for validation.



Run:



```cmd

python -m pytest -v validate_pipeline.py

```



Expected result:



```text

4 passed

```



The validation checks are defined in:



```text

validate_pipeline.py

```



GitHub Actions also runs the validation tests.



---
## 19. GitHub Actions
GitHub Actions is used for Continuous Integration.



The workflow runs automated checks such as tests when changes are pushed to the repository.



Conceptually:



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
## 20. Docker
Docker is included as a ****containerization demonstration****.



Docker starts:



- FastAPI

- Streamlit

- Event Watcher



The Scheduler is intentionally ****NOT**** started by Docker.



### Start Docker
From the project root:



```cmd

docker compose -f deployment\docker\docker-compose.yml up --build

```



Docker exposes:



```text

Streamlit

http://localhost:8501



FastAPI

http://localhost:8001



FastAPI Swagger

http://localhost:8001/docs

```



### Docker architecture
```text

Docker

   |

   +----------------------+

   |          |           |

   v          v           v

FastAPI   Streamlit   Event Watcher

   |          |           |

   +----------+-----------+

              |

              v

       Shared Project Data

```



The Docker setup is intended to demonstrate how the application services can be containerized together. It is not presented as a production/cloud deployment architecture.



### Run the simulator with Docker
Keep Docker running.



Open another VS Code terminal at the project root:



```cmd

python deployment\data_simulator_stream.py

```



The simulator generates incoming data while the Event Watcher running inside Docker processes it.



Observe:



- Docker terminal

- Event Watcher processing

- Streamlit dashboard

- FastAPI, where applicable



### Stop Docker
When the Docker workflow is complete:



```cmd

docker compose -f deployment\docker\docker-compose.yml down

```



---
# 21. Complete Setup and Execution Flow
This section provides the recommended learner workflow from a fresh ZIP extraction.



## Step 1: Extract and Open the Project
1. Extract the project ZIP file.

2. Open the extracted `MLOps_Telecom_Churn` folder in VS Code.

3. Open the VS Code terminal at the project root.



---
## Step 2: Create and Activate the Virtual Environment
Create the virtual environment:



```cmd

python -m venv .venv

```



Activate it on Windows:



```cmd

.venv\Scripts\activate

```



---
## Step 3: Install Dependencies
Run:



```cmd

pip install -r requirements.txt

```



---
## Step 5: Prepare the Local Git/DVC Environment

Run:

```cmd
python setup_project.py
```

This script checks Git and DVC and prepares the local project environment.

### If the project was downloaded as a ZIP

The script initializes the local Git repository and adds the baseline training dataset to DVC when required.

### If the project was cloned from GitHub

The script preserves the existing Git/DVC configuration and does not re-add the baseline dataset.

### Important

You do **not** need to manually run:

```cmd
git init
dvc init
python -m dvc add data/data_sources/train.csv
```

during the initial project setup.

The main baseline dataset remains available locally so the initial training and model-selection workflow can run without requiring a DVC remote.

---

## Step 6: Run Validation
Run:



```cmd

python -m pytest -v validate_pipeline.py

```



Expected:



```text

4 passed

```



---
## Step 6: Run Baseline Training
Run:



```cmd

python modelling\train.py

```



This trains the baseline model used by the project.



---
## Step 7: Run Model Selection
Run:



```cmd

python modelling\model_selection.py

```



This evaluates the candidate models and stores the model-selection metrics used by the project.



---
## Step 8: Start MLflow
Run:



```cmd

mlflow ui --host 127.0.0.1 --port 5000

```



Open:



```text

http://localhost:5000

```



Keep this terminal running.



---
## Step 9: Start FastAPI and Streamlit
Start FastAPI:



```cmd

uvicorn src.main:app --reload

```



Start Streamlit in another terminal:



```cmd

streamlit run src/streamlit_app.py

```



Open:



```text

http://localhost:8001/docs

```



and:



```text

http://localhost:8501

```



Keep these services running while demonstrating the workflow.



---
## Step 10: Generate the Workflow Datasets
Before starting the Event Watcher:



```cmd

python deployment\data_simulator_source.py

```



This prepares the datasets required for the Event Watcher and Scheduler demonstrations.



---
# 22. Event Watcher Demonstration
## Step 1: Start the Event Watcher
Open a new VS Code terminal:



```cmd

python deployment\drift_monitor_and_trigger.py

```



Keep this terminal running.



## Step 2: Start the Data Simulator
Open another terminal:



```cmd

python deployment\data_simulator_stream.py

```



Observe the Event Watcher terminal and Streamlit dashboard.



## Step 3: Complete the Workflow
Follow the incoming-data workflow through:



```text

Incoming Data

      |

      v

Validation

      |

      v

Training Data Update

      |

      v

DVC Versioning

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

Candidate Training

      |

      v

Model Approval

      |

      v

Monitoring

```



## Step 4: Stop the Event Watcher
After the Event Watcher workflow has completed:



```text

Ctrl + C

```



Stop it in the terminal where the Event Watcher is running.



---
# 23. Prepare for the Scheduler Demonstration
After finishing the Event Watcher workflow:



### 1. Stop the Event Watcher
Make sure the Event Watcher process is stopped.



### 2. Clean the generated incoming folder
Run:



```cmd

del /q data\data_sources\incoming\\*

```



### 3. Make sure the incoming folder is empty
This prevents previously processed Event Watcher data from being picked up again.



---
# 24. Scheduler Demonstration
## Step 1: Start the Scheduler
Open a new VS Code terminal:



```cmd

python deployment\scheduler.py

```



Keep this terminal running.



## Step 2: Start the Data Simulator
Open another VS Code terminal:



```cmd

python deployment\data_simulator_stream.py

```



Observe the Scheduler terminal and Streamlit dashboard.



## Step 3: Wait for Scheduler Cycles
The Scheduler runs on a 1-minute cycle.



After an approval or retraining action, wait for the next scheduler cycle before expecting the next stage.



## Step 4: Stop the Scheduler
After the Scheduler workflow has completed:



```text

Ctrl + C

```



Stop it in the terminal where the Scheduler is running.



---
# 25. Reset Event Watcher and Scheduler Before Docker
Before starting Docker, clean the generated Event Watcher and Scheduler workflow state.



This reset ensures Docker starts with:



- Event Watcher baseline dataset = 5,300 rows

- Scheduler baseline dataset = 5,300 rows

- No old generated `.dvc` files

- Empty `incoming/` and `incoming_batches/` folders



## Step 1: Stop Event Watcher and Scheduler
Stop both Python processes before continuing.



## Step 2: Empty the generated incoming folder
Run:



```cmd

del /q data\data_sources\incoming\\*

```



## Step 3: Reset the Event Watcher dataset
Restore the original 5,300-row dataset:



```cmd

copy /Y data\data_sources\train.csv data\data_sources\event_watcher_data\train_trigger.csv

```



Remove existing generated DVC metadata:



```cmd

del /q data\data_sources\event_watcher_data\train_trigger.csv.dvc

```



Verify:



```cmd

python -c "import pandas as pd; df=pd.read_csv('data/data_sources/event_watcher_data/train_trigger.csv'); print('Rows:', len(df)); print('Columns:', len(df.columns))"

```



Expected:



```text

Rows: 5300

Columns: 21

```



## Step 4: Reset the Scheduler dataset
Restore the original 5,300-row dataset:



```cmd

copy /Y data\data_sources\train.csv data\data_sources\scheduler_data\train_scheduler.csv

```



Remove existing generated DVC metadata:



```cmd

del /q data\data_sources\scheduler_data\train_scheduler.csv.dvc

```



Verify:



```cmd

python -c "import pandas as pd; df=pd.read_csv('data/data_sources/scheduler_data/train_scheduler.csv'); print('Rows:', len(df)); print('Columns:', len(df.columns))"

```



Expected:



```text

Rows: 5300

Columns: 21

```



## Step 5: Remove Scheduler Processing History
Run:



```cmd

del /q logs\scheduler_processed_files.json

```



## Important DVC Note
Do ****not**** delete `.dvc/cache`.



The workflow-generated `.dvc` metadata files can be removed during this reset, but the DVC cache is separate from the `.dvc` metadata directory.



---
# 26. Docker Demonstration
After completing the Event Watcher and Scheduler demonstrations and cleaning the generated workflow state:



## Step 1: Start Docker
From the project root:



```cmd

docker compose -f deployment\docker\docker-compose.yml up --build

```



Docker starts:



- FastAPI

- Streamlit

- Event Watcher



The Scheduler is intentionally ****NOT**** started by Docker.



## Step 2: Open Streamlit
```text

http://localhost:8501

```



Use Streamlit to observe the MLOps workflow and monitoring/approval information.



## Step 3: Open FastAPI
```text

http://localhost:8001/docs

```



This opens the FastAPI Swagger documentation.



## Step 4: Run the Data Simulator
Keep Docker running.



Open another VS Code terminal at the project root:



```cmd

python deployment\data_simulator_stream.py

```



The simulator generates incoming data while the Event Watcher running inside Docker processes it.



Observe:



- Docker terminal

- Event Watcher processing

- Streamlit dashboard

- FastAPI, where applicable



## Step 5: Stop Docker
```cmd

docker compose -f deployment\docker\docker-compose.yml down

```



---
# 27. Running the Simulation Again
Before running the simulation again, reset the generated workflow state.



This is important because previously generated records can cause duplicate `CUSTOMER_ID` problems during another simulation run.



## Clean Generated CSV Files
Remove generated CSV files from:



```text

data\data_sources\event_watcher_data\\

data\data_sources\scheduler_data\\

```



Do not delete the original baseline/project CSV files.



## Clean Generated DVC Files
Remove the generated Event Watcher and Scheduler DVC files created by the previous simulation.



Do not delete DVC files that are part of the original project/repository unless the workflow specifically instructs you to restore or regenerate them.



## Empty the Incoming Directory
Make sure this directory is empty:



```text

data\data_sources\incoming\\

```



Remove previously generated incoming files before starting another simulation.



## Restore the Event Watcher Baseline
Restore:



```cmd

copy /Y data\data_sources\train.csv data\data_sources\event_watcher_data\train_trigger.csv

```



Remove old generated DVC metadata:



```cmd

del /q data\data_sources\event_watcher_data\train_trigger.csv.dvc

```



Verify:



```cmd

python -c "import pandas as pd; df=pd.read_csv('data/data_sources/event_watcher_data/train_trigger.csv'); print('Rows:', len(df)); print('Columns:', len(df.columns))"

```



## Restore the Scheduler Baseline
Restore:



```cmd

copy /Y data\data_sources\train.csv data\data_sources\scheduler_data\train_scheduler.csv

```



Remove old generated DVC metadata:



```cmd

del /q data\data_sources\scheduler_data\train_scheduler.csv.dvc

```



Verify:



```cmd

python -c "import pandas as pd; df=pd.read_csv('data/data_sources/scheduler_data/train_scheduler.csv'); print('Rows:', len(df)); print('Columns:', len(df.columns))"

```



Remove Scheduler processing history:



```cmd

del /q logs\scheduler_processed_files.json

```



Verify both datasets:



```cmd

python -c "import pandas as pd; print('Event Watcher rows:', len(pd.read_csv('data/data_sources/event_watcher_data/train_trigger.csv'))); print('Scheduler rows:', len(pd.read_csv('data/data_sources/scheduler_data/train_scheduler.csv')))"

```



Expected:



```text

Event Watcher rows: 5300

Scheduler rows: 5300

```



After restoring the required baseline datasets, cleaning generated CSV/DVC files, and emptying `incoming/`, run the required workflow again.



This clean reset helps avoid duplicate `CUSTOMER_ID` errors and keeps repeated demonstrations reproducible.



---
# 28. Quick Command Reference
| Task | Command |

|---|---|

| Create environment | `python -m venv .venv` |

| Activate environment | `.venv\Scripts\activate` |

| Install dependencies | `pip install -r requirements.txt` |

| Prepare Git/DVC setup | `python setup_project.py` |
| Run validation | `python -m pytest -v validate_pipeline.py` |

| Train baseline | `python modelling\train.py` |

| Model selection | `python modelling\model_selection.py` |

| MLflow | `mlflow ui --host 127.0.0.1 --port 5000` |

| FastAPI | `uvicorn src.main:app --reload` |

| Streamlit | `streamlit run src/streamlit_app.py` |

| Generate workflow datasets | `python deployment\data_simulator_source.py` |

| Event Watcher | `python deployment\drift_monitor_and_trigger.py` |

| Data simulator | `python deployment\data_simulator_stream.py` |

| Scheduler | `python deployment\scheduler.py` |

| Docker | `docker compose -f deployment\docker\docker-compose.yml up --build` |

| Stop Docker | `docker compose -f deployment\docker\docker-compose.yml down` |



---
# 29. Important URLs
| Service | URL |

|---|---|

| MLflow | http://localhost:5000 |

| FastAPI | http://localhost:8001 |

| FastAPI Swagger | http://localhost:8001/docs |

| Streamlit | http://localhost:8501 |



---
# 30. Troubleshooting
### Missing Python Package
Run:



```cmd

pip install -r requirements.txt

```



### Streamlit Not Found
Use:



```cmd

streamlit run src\streamlit_app.py

```



### MLflow Not Found
Use:



```cmd

mlflow ui --host 127.0.0.1 --port 5000

```



### DVC Not Found
Use:



```cmd

python -m dvc

```



### FastAPI Not Starting
Make sure the terminal is opened from the project root and run:



```cmd

uvicorn src.main:app --reload

```



### Scheduler Does Not Process Immediately
The Scheduler checks periodically.



For the classroom version, the interval is approximately ****1 minute****.



Wait for the next Scheduler cycle before assuming that the workflow is stuck.



### No New Customer Records
If the incoming CSV contains customer IDs that already exist in the training dataset, no new records may be added.



This is expected.



### Duplicate Customer ID During a New Demonstration
Reset the generated Event Watcher and Scheduler datasets and clear the generated incoming files before starting another simulation.



Use the reset procedure in ****Section 27 — Running the Simulation Again****.



---
# 31. Key MLOps Concepts
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
# 32. Important Notes
- This is an educational project.

- Event Watcher and Scheduler are separate demonstrations.

- Complete the Event Watcher demonstration before starting the Scheduler demonstration.

- Clean the generated incoming files before switching from Event Watcher to Scheduler.

- The Event Watcher demonstration uses the incoming-data simulator.

- The Scheduler demonstration also uses the data simulator to generate incoming data.

- FastAPI and Streamlit should be running for the human approval demonstration.

- Data drift does not automatically mean model performance has degraded.

- A candidate model does not automatically become the production model.

- Human approval is required at the configured approval stages.

- Runtime-generated logs, MLflow data, model artifacts, and temporary incoming files should not normally be committed to Git.

- Docker starts FastAPI, Streamlit, and Event Watcher.

- The Scheduler is intentionally not started by Docker.

- Do not delete the DVC cache during normal workflow resets.



---
# 33. Final Workflow
The complete project lifecycle is:



```text

                         TELECOM CHURN MLOPS

                                |

                                v

                         Project_setup

                                |

                                v

                        Data Validation

                                |

                                v

                     Baseline Model Training

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

                    +-----------+-----------+

                    |                       |

                    v                       v

              Event Watcher             Scheduler

                    |                       |

                    +-----------+-----------+

                                |

                                v

                         Data Validation

                                |

                                v

                      Update Training Data

                                |

                                v

                         DVC Versioning

                                |

                                v

                        Drift Detection

                                |

                                v

                         Policy Decision

                                |

                    +-----------+-----------+

                    |                       |

                    v                       v

                 No Action             Retraining

                                            |

                                            v

                                   Human Approval

                                            |

                                            v

                                    Candidate Model

                                            |

                                            v

                                     Model Approval

                                            |

                                            v

                                    Production Model

                                            |

                                            v

                                         Docker

                                            |

                                            v

                              FastAPI + Streamlit +

                                  Event Watcher

```



This project provides a complete educational example of how machine learning can be managed as an operational lifecycle rather than as a one-time model training exercise.



For the exact learner execution sequence, use the companion `execution_flow.md` file. That file is the detailed step-by-step operational guide; this README provides the project overview, architecture, component responsibilities, setup guidance, and command reference.
