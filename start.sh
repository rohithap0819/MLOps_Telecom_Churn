
#!/bin/bash

set -e

echo "=========================================="
echo "Starting Telecom Churn MLOps Application"
echo "=========================================="

# Store process IDs
PIDS=()

# Clean up all processes when container stops
cleanup() {
    echo "Stopping Telecom Churn MLOps services..."

    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null || true
    done

    wait
}

trap cleanup SIGTERM SIGINT EXIT


# --------------------------------------------------
# 1. Generate initial candidate models if required
# --------------------------------------------------

if [ ! -f "models/candidates/logistic_regression.joblib" ] || \
   [ ! -f "models/candidates/decision_tree.joblib" ] || \
   [ ! -f "logs/candidate_metrics.csv" ]; then

    echo "Candidate models not found."
    echo "Generating initial candidate models..."

    python candidate_training.py

    echo "Initial candidate training completed."
fi


# --------------------------------------------------
# 2. Start FastAPI
# --------------------------------------------------

echo "Starting FastAPI backend..."

uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8001 &

PIDS+=("$!")


# --------------------------------------------------
# 3. Start Streamlit
# --------------------------------------------------

echo "Starting Streamlit dashboard..."

streamlit run src/dashboard/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true &

PIDS+=("$!")


# --------------------------------------------------
# 4. Start Event Watcher
# --------------------------------------------------

echo "Starting incoming data event watcher..."

python event_watcher.py &

PIDS+=("$!")


# --------------------------------------------------
# 5. Start Scheduled Monitoring
# --------------------------------------------------

echo "Starting scheduled candidate training..."

python scheduler.py &

PIDS+=("$!")


# --------------------------------------------------
# 6. Keep container running
# --------------------------------------------------

echo "=========================================="
echo "All four services have been started."
echo "Streamlit: http://localhost:8501"
echo "FastAPI:   http://localhost:8001/docs"
echo "=========================================="

wait -n "${PIDS[@]}"
