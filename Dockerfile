FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix script line endings and permissions
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Create required runtime directories
RUN mkdir -p logs models/production models/candidates data/incoming

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8001

# Start both applications
CMD ["bash", "start.sh"]
