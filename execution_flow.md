# 🚀 MLOps Telecom Churn — Execution Flow


This file is a step-by-step execution guide for running the project after extracting the ZIP file.

IMPORTANT:
- Run the commands in the order shown.
- Some components must remain running while another command is executed in a separate terminal.
- Do not close the Event Watcher or Scheduler terminal until that workflow is complete.
- Detailed explanation of each script and the project architecture is available in README.md.


### 1. EXTRACT AND OPEN THE PROJECT


1. Extract the project ZIP file.
2. Open the extracted MLOps_Telecom_Churn folder in VS Code.
3. Open the VS Code terminal at the project root.


### 2. CREATE AND ACTIVATE VIRTUAL ENVIRONMENT


Create the virtual environment:

```cmd
python -m venv .venv
```


Activate it on Windows:

.venv\Scripts\activate


### 3. INSTALL DEPENDENCIES


Run:

```cmd
pip install -r requirements.txt
```


### 4. PREPARE THE LOCAL GIT/DVC ENVIRONMENT**

Run the project setup script once after installing the dependencies:

```cmd
python setup_project.py
```

This step supports both ways of obtaining the project:

- **GitHub clone:** the existing Git and DVC configuration is preserved.
- **GitHub ZIP:** the script creates the required local Git context and prepares the baseline dataset for local DVC usage.

Learners do **not** need to manually run `git init`, `dvc init`, or `dvc add` for the initial project setup.

Do not create a GitHub repository or configure a GitHub remote for the ZIP workflow.


### 4. RUN VALIDATION


Run:

```cmd
python -m pytest -v validate_pipeline.py
```


Expected result:

4 passed

This checks that the project's data-validation tests are working before continuing.


### 5. RUN BASELINE TRAINING


Run:

```cmd
python modelling\train.py
```


This trains the baseline model used by the project.


### 6. RUN MODEL SELECTION


Run:

```cmd
python modelling\model_selection.py
```


This evaluates the candidate models and stores the model-selection metrics used by the project.


### 7. START MLflow UI


Start the MLflow UI so learners can view the experiment runs, parameters,
metrics, and model information.

Run:

```cmd
mlflow ui --host 127.0.0.1 --port 5000
```


Open the MLflow UI in the browser at:

http://localhost:5000


Keep the MLflow terminal running while viewing the UI.


### 8. OPEN FASTAPI AND STREAMLIT


Start the FastAPI application and Streamlit dashboard before continuing
with the data and drift workflows.

FastAPI:

```cmd
uvicorn src.main:app --reload
```


Streamlit:

```cmd
streamlit run src/streamlit_app.py
```


Open:

http://localhost:8001/docs


and:

http://localhost:8501


### 9. GENERATE THE WORKFLOW DATASETS


Before starting the Event Watcher, run:

```cmd
python deployment\data_simulator_source.py
```


This prepares the datasets required for the Event Watcher and Scheduler demonstrations.


=========================================================
## 🟦 EVENT WATCHER FLOW


### 10. START THE EVENT WATCHER


Open a NEW VS Code terminal.

Run:

```cmd
python deployment\drift_monitor_and_trigger.py
```


Keep this terminal running.

The Event Watcher waits for incoming data and processes the workflow when new data arrives.


### 11. START THE DATA SIMULATOR


Open ANOTHER VS Code terminal.

Run:

```cmd
python deployment\data_simulator_stream.py
```


The simulator generates incoming data for the Event Watcher.

Observe the Event Watcher terminal and Streamlit dashboard to follow the workflow.


The flow is:

Incoming data
```text
    |
    v
```

Event Watcher detects the data
```text
    |
    v
```

Data processing
```text
    |
    v
```

Drift detection
```text
    |
    v
```

Retraining condition
```text
    |
    v
```

Human approval
```text
    |
    v
```

Candidate model training
```text
    |
    v
```

Model evaluation / approval
```text
    |
    v
```

Monitoring


### 10. STOP THE EVENT WATCHER


After the Event Watcher workflow has completed, stop it using:

```cmd
Ctrl + C
```


Stop it in the terminal where the Event Watcher is running.


---
### Before Starting the Scheduler


After finishing the Event Watcher workflow:

1. Stop the Event Watcher.

2. Clean the generated incoming folder before starting the Scheduler.

3. Delete the incoming folder.

Run:
```cmd
   del /q data\data_sources\incoming\*
```


4. Make sure the incoming folder is empty before starting the Scheduler.

This prevents previously processed Event Watcher data from being picked up again.


=========================================================
## 🟦 SCHEDULER FLOW


### 14. START THE SCHEDULER


Open a NEW VS Code terminal.

Run:

```cmd
python deployment\scheduler.py
```


Keep this terminal running.

The Scheduler checks for new data on its configured schedule.


### 15. START THE DATA SIMULATOR


Open ANOTHER VS Code terminal.

Run:

```cmd
python deployment\data_simulator_stream.py
```


Observe the Scheduler terminal and Streamlit dashboard.


### IMPORTANT — WAIT FOR THE SCHEDULER CYCLE


The Scheduler runs on a 1-minute cycle.

After each approval/retraining action, WAIT for the next scheduler cycle before expecting the next stage.

Do not immediately assume that the workflow is stuck if the next action does not appear.

Typical approval flow:

Approve Retraining
```text
        |
        v
```

WAIT ~1 MINUTE
```text
        |
        v
```

Candidate Model Training / Model Approval
```text
        |
        v
```

Approve Model
```text
        |
        v
```

WAIT ~1 MINUTE
```text
        |
        v
```

Next Drift Check
```text
        |
        v
```

Approve Retraining
```text
        |
        v
```

WAIT ~1 MINUTE
```text
        |
        v
```

Model Approval
```text
        |
        v
```

WAIT ~1 MINUTE
```text
        |
        v
```

Next Scheduler Cycle


### 16. STOP THE SCHEDULER


After the Scheduler workflow has completed, stop it using:

```cmd
Ctrl + C
```


Stop it in the terminal where the Scheduler is running.


IMPORTANT:
Before starting Docker, clean the generated Event Watcher and Scheduler workflow state.

---
### Reset Event Watcher and Scheduler Before Docker


After completing both the Event Watcher and Scheduler workflows, clean the generated workflow data before starting Docker.

This reset ensures Docker starts with:
- Event Watcher baseline dataset = 5,300 rows
- Scheduler baseline dataset = 5,300 rows
- No old generated `.dvc` files
- Empty `incoming/` and `incoming_batches/` folders


### 1. Stop the Event Watcher and Scheduler


Stop both Python processes before continuing.

### 2. Reset the Event Watcher Dataset


Restore the original 5,300-row dataset:

---
First Delete data/data_sources/incoming/
Run:
```cmd
    del /q data\data_sources\incoming\*
```


---
### For event_watcher


```cmd
copy /Y data\data_sources\train.csv data\data_sources\event_watcher_data\train_trigger.csv

:: Remove Existing DVC Metadata
del /q data\data_sources\event_watcher_data\train_trigger.csv.dvc

:: Verify
python -c "import pandas as pd; df=pd.read_csv('data/data_sources/event_watcher_data\train_trigger.csv'); print('Rows:', len(df)); print('Columns:', len(df.columns))"


-------------------------
For Scheduler
-------------------------

```cmd
copy /Y data\data_sources\train.csv data\data_sources\scheduler_data\train_scheduler.csv

:: Remove Existing DVC Metadata
del /q data\data_sources\scheduler_data\train_scheduler.csv.dvc

:: Verify
python -c "import pandas as pd; df=pd.read_csv('data/data_sources/scheduler_data/train_scheduler.csv'); print('Rows:', len(df)); print('Columns:', len(df.columns))"

:: Remove Scheduler Processing History
del /q logs\scheduler_processed_files.json


=========================================================
DOCKER FLOW
=========================================================

18. START DOCKER
----------------

From the project root, run:

docker compose -f deployment\docker\docker-compose.yml up --build

Docker starts:

- FastAPI
- Streamlit
- Event Watcher

The Scheduler is intentionally NOT started by Docker.


19. OPEN STREAMLIT
------------------

Open:

http://localhost:8501

Use Streamlit to observe the MLOps workflow and its monitoring/approval information.


20. OPEN FASTAPI
----------------

Open:

http://localhost:8001/docs

This opens the FastAPI Swagger documentation.


21. RUN THE DATA SIMULATOR WITH DOCKER
--------------------------------------

Keep Docker running.

Open ANOTHER VS Code terminal at the project root.

Run:

python deployment\data_simulator_stream.py

The simulator generates incoming data while the Event Watcher running inside Docker processes it.

Observe:

- Docker terminal
- Event Watcher processing
- Streamlit dashboard
- FastAPI, where applicable


22. STOP DOCKER
---------------

When the Docker workflow is complete, stop the containers with:

docker compose -f deployment\docker\docker-compose.yml down


=========================================================
RUNNING THE SIMULATION AGAIN
=========================================================

Before running the simulation again, reset the generated workflow state.

This is important because previously generated records can cause duplicate CUSTOMER_ID problems during another simulation run.


23. CLEAN GENERATED CSV FILES
-----------------------------

Remove the generated CSV files from:

data\data_sources\event_watcher_data\
data\data_sources\scheduler_data\

Do not delete the original baseline/project CSV files.


24. CLEAN GENERATED DVC FILES
-----------------------------

Remove the generated Event Watcher and Scheduler DVC files created by the previous simulation.

Do not delete the DVC files that are part of the original project/repository unless the README specifically instructs you to restore or regenerate them.


25. EMPTY THE INCOMING DIRECTORY
--------------------------------

Make sure this directory is empty:

data\data_sources\incoming\

Remove previously generated incoming files before starting another simulation.


### 26. RESTORE THE REQUIRED BASELINE DATA FOR ANOTHER RUN

If you want to run the Event Watcher simulation again, restore the Scheduler
training dataset from the main training dataset:

copy /Y data\data_sources\train.csv data\data_sources\scheduler_data\train_scheduler.csv

Then recreate the DVC file:

python -m dvc add data\data_sources\scheduler_data\train_scheduler.csv

Check the dataset:

python -c "import pandas as pd; df=pd.read_csv('data/data_sources/scheduler_data/train_scheduler.csv'); print('Rows:', len(df)); print('Columns:', len(df.columns))"

and delete logs/scheduler_processed_files.json

-----------------------------------------------------------------------------
For the Event Watcher workflow, restore the Event Watcher baseline from the
-----------------------------------------------------------------------------
main training dataset:

copy /Y data\data_sources\train.csv data\data_sources\event_watcher_data\train_trigger.csv

Then recreate the DVC file:

python -m dvc add data\data_sources\event_watcher_data\train_trigger.csv

Check the dataset:

python -c "import pandas as pd; df=pd.read_csv('data/data_sources/event_watcher_data/train_trigger.csv'); print('Rows:', len(df)); print('Columns:', len(df.columns))"


-----------------------
Verify
-----------------------
python -c "import pandas as pd; print('Event Watcher rows:', len(pd.read_csv('data/data_sources/event_watcher_data/train_trigger.csv'))); print('Scheduler rows:', len(pd.read_csv('data/data_sources/scheduler_data/train_scheduler.csv')))"

Expected:
    Event Watcher rows: 5300
    Scheduler rows: 5300

----------------------------
27. RUN THE SIMULATION AGAIN
----------------------------

After restoring the required baseline datasets, cleaning generated CSV/DVC
files, and emptying incoming/, run the required workflow again.

This clean reset helps avoid duplicate CUSTOMER_ID errors and keeps repeated
demonstrations reproducible.


=========================================================
QUICK COMPLETE FLOW
=========================================================

Extract ZIP
    |
    v
Open project in VS Code
    |
    v
Create / activate .venv
    |
    v
pip install -r requirements.txt
    |
    v
Run setup_project.py
    |
    v
Run validation
    |
    v
Run train.py
    |
    v
Run model_selection.py
    |
    v
Start MLflow UI
    |
    v
Open FastAPI + Streamlit
    |
    v
Run data_simulator_source.py
    |
    +------------------------------+
    |                              |
    v                              v
EVENT WATCHER                  SCHEDULER
    |                              |
    | Start Event Watcher         | Clean Event Watcher data
    | Start simulator             | Start Scheduler
    | Observe flow                | Start simulator
    | Stop Event Watcher          | Wait ~1 minute between cycles
    |                              | Observe flow
    |                              | Stop Scheduler
    +--------------+---------------+
                   |
                   v
          Clean generated data
                   |
                   v
            Empty incoming/
                   |
                   v
              DOCKER
                   |
                   v
       docker compose up --build
                   |
                   v
       FastAPI + Streamlit +
          Event Watcher
                   |
                   v
       Run simulator again
                   |
                   v
          Observe Docker flow
                   |
                   v
              Stop Docker


END

For detailed explanations of the project architecture, scripts, monitoring,
drift detection, retraining, approvals, DVC, FastAPI, Streamlit, Scheduler,
Event Watcher, and Docker, refer to README.md
