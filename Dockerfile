# ─────────────────────────────────────────────────────────────
# Pelid AI Decision Gateway — Hardened Production Container
# 
# Security & Privacy Hardening:
# - Zero baked secrets or credentials (API keys passed at runtime via env)
# - Zero personal/host metadata or IP addresses included
# - No database history or telemetry logs copied into the image
# - Runs as an unprivileged non-root user (UID 10001)
# - Installs CPU-only PyTorch to minimize image footprint (<800MB vs 3.5GB)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Prevent bytecode compilation and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

# 1. Install minimal system packages for native ONNX runtime & healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install CPU-only PyTorch first (drastically reduces image size)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Copy dependency definitions and install Python packages
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy ONLY necessary application code, models, and domain definitions
# (.dockerignore guarantees .env, *.db, *.log, tests/ are completely excluded)
COPY src/ ./src/
COPY models/ ./models/
COPY pelid.yaml ./pelid.yaml

# Copy only the schema/criteria definition file, NOT user databases
RUN mkdir -p /app/data
COPY data/intents.txt ./data/intents.txt

# Optional static dashboard (if present)
COPY static/ ./static/

# 5. Install pelid in editable / production mode
RUN pip install --no-cache-dir -e .

# 6. Create non-root user and grant ownership of working directory
RUN groupadd -g 10001 pelidgroup && \
    useradd -u 10001 -g pelidgroup -s /bin/bash -m peliduser && \
    chown -R peliduser:pelidgroup /app

# 7. Switch to unprivileged user for security compliance
USER peliduser

# 8. Expose gateway service port
EXPOSE 8080

# 9. Kubernetes & Docker healthcheck probe
HEALTHCHECK --interval=20s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# 10. Start high-concurrency gateway
CMD ["python", "-m", "uvicorn", "pelid.proxy:app", "--host", "0.0.0.0", "--port", "8080"]
