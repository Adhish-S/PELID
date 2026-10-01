# Production Dockerfile for Pelid AI Decision Gateway
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential for compiling native extensions, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and metadata
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Install pelid package
RUN pip install --no-cache-dir -e .

# Expose default proxy port
EXPOSE 8080

# Production container health probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Pre-warm decision engines and start server
CMD ["python", "-m", "uvicorn", "pelid.proxy:app", "--host", "0.0.0.0", "--port", "8080"]
