FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# First upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install large packages separately with increased timeout and alternative index
RUN pip install --no-cache-dir --timeout=1000 --retries=10 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch>=2.9.0

RUN pip install --no-cache-dir --timeout=1000 --retries=10 xgboost==2.0.3
RUN pip install --no-cache-dir --timeout=1000 --retries=10 transformers>=4.37.0
RUN pip install --no-cache-dir --timeout=1000 --retries=10 neuralforecast>=3.1.0

# Install remaining packages
RUN pip install --no-cache-dir --timeout=1000 --retries=10 -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY .env .

# Create necessary directories
RUN mkdir -p logs data mlruns models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
