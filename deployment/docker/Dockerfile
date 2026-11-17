# --------------------------------------------------
# 1. Base Python image
# --------------------------------------------------

FROM python:3.12-slim

# Use /app as the working directory inside the container.
WORKDIR /app


# --------------------------------------------------
# 2. Install system dependencies
# --------------------------------------------------

# Install build tools required by some Python packages.
# --no-install-recommends keeps the image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# --------------------------------------------------
# 3. Install Python dependencies
# --------------------------------------------------

# Copy requirements first so Docker can reuse this layer
# when the Python dependency list has not changed.
COPY requirements.txt .

# Install all project Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt


# --------------------------------------------------
# 4. Copy the project
# --------------------------------------------------

# Copy the project source code, scripts, configuration,
# datasets, and other required project files into the container.
COPY . .


# --------------------------------------------------
# 5. Prepare the startup script
# --------------------------------------------------

# Windows may store the shell script with CRLF line endings.
# Convert CRLF to Unix LF and make the script executable.
RUN sed -i 's/\r$//' deployment/docker/start.sh && chmod +x deployment/docker/start.sh


# --------------------------------------------------
# 6. Create runtime directories
# --------------------------------------------------

# Create directories that the application may write to
# while running inside the container.
RUN mkdir -p \
    logs \
    models/production \
    models/candidates \
    data/data_sources/incoming


# --------------------------------------------------
# 7. Expose application ports
# --------------------------------------------------

# Streamlit dashboard
EXPOSE 8501

# FastAPI backend
EXPOSE 8001


# --------------------------------------------------
# 8. Start the application
# --------------------------------------------------

# start.sh launches:
#   - FastAPI
#   - Streamlit
#   - Event Watcher
#
# Scheduler remains commented out in start.sh.
CMD ["bash", "deployment/docker/start.sh"]


# --------------------------------------------------
# DOCKERFILE SUMMARY
# --------------------------------------------------
# Purpose:
#   Build the Docker image for the Telecom Churn MLOps
#   project.
#
# Steps:
#   1. Start with Python 3.12.
#   2. Install required system build tools.
#   3. Install Python dependencies from requirements.txt.
#   4. Copy the project into the container.
#   5. Prepare start.sh for Linux execution.
#   6. Create required runtime directories.
#   7. Expose FastAPI and Streamlit ports.
#   8. Start the application through start.sh.
#
# Important:
#   - baseline.py is NOT executed automatically.
#   - model_selection.py is NOT executed automatically.
#   - candidate_training.py is NOT executed automatically.
#   - Event Watcher is the active Docker trigger.
#   - Scheduler is available but remains disabled.
#
# Role in the project:
#   This Dockerfile defines how the learner's existing
#   MLOps project is packaged into a runnable container.