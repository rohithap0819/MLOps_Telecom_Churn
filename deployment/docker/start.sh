#!/bin/bash

set -e

echo "=========================================="
echo "Starting Telecom Churn MLOps Application"
echo "=========================================="

# Store process IDs so all background services can be stopped together.
PIDS=()


# --------------------------------------------------
# 1. Clean up background processes when container stops
# --------------------------------------------------

cleanup() {
    echo "Stopping Telecom Churn MLOps services..."

    # Stop every service started by this script.
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null || true
    done

    # Wait for all background processes to finish.
    wait
}

# Run cleanup when the container receives a stop/interrupt signal.
trap cleanup SIGTERM SIGINT EXIT


# --------------------------------------------------
# 2. Start FastAPI
# --------------------------------------------------

echo "Starting FastAPI backend..."

# Start the FastAPI application in the background.
# 0.0.0.0 allows the API to be accessed from outside the container.
uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8001 &

# Store the FastAPI process ID.
PIDS+=("$!")


# --------------------------------------------------
# 3. Start Streamlit
# --------------------------------------------------

echo "Starting Streamlit dashboard..."

# Start the Streamlit dashboard in the background.
# 0.0.0.0 allows the dashboard to be accessed from outside the container.
streamlit run src/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true &

# Store the Streamlit process ID.
PIDS+=("$!")


# --------------------------------------------------
# 4. Start Event Watcher
# --------------------------------------------------

echo "Starting incoming data event watcher..."

# Start the Event Watcher in the background.
# Event Watcher is the active automatic trigger inside Docker.
python deployment/drift_monitor_and_trigger.py &

# Store the Event Watcher process ID.
PIDS+=("$!")


# --------------------------------------------------
# 5. Scheduler - Alternative Trigger
# --------------------------------------------------

# The Scheduler is intentionally commented out.
# It is demonstrated separately during the local MLOps workflow.
#
# Docker uses Event Watcher as the active automatic trigger.
# Uncomment the following lines if the Scheduler needs to
# be used instead of Event Watcher.

# echo "Starting scheduled monitoring..."
#
# python deployment/scheduler.py &
#
# PIDS+=("$!")


# --------------------------------------------------
# 6. Keep the container running
# --------------------------------------------------

echo "=========================================="
echo "Docker services have been started."
echo "Streamlit: http://localhost:8501"
echo "FastAPI:   http://localhost:8001/docs"
echo "Event Watcher: Running"
echo "Scheduler: Disabled"
echo "=========================================="

# Keep the container alive while the background services run.
wait -n "${PIDS[@]}"


# --------------------------------------------------
# SCRIPT SUMMARY
# --------------------------------------------------
# Purpose:
#   Start the main services of the Telecom Churn MLOps
#   application inside the Docker container.
#
# Services:
#   1. FastAPI backend
#   2. Streamlit dashboard
#   3. Event Watcher - active automatic trigger
#   4. Scheduler - included as an alternative but disabled
#
# Important:
#   - Candidate training is NOT automatically executed.
#   - baseline.py and model_selection.py are run manually
#     before Docker as part of the learner workflow.
#   - Event Watcher is the active trigger inside Docker.
#   - Scheduler remains commented out because it is the
#     alternative trigger demonstrated separately.
#
# Inputs:
#   Existing project code, models, datasets, configuration,
#   and Python dependencies available inside the container.
#
# Outputs:
#   Running FastAPI, Streamlit, and Event Watcher services.
#
# Role in the project:
#   This script is the Docker container startup script.
#   It launches the services required for the final
#   containerized MLOps demonstration.