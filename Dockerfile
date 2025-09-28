# ---------- Base image ----------
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    UVICORN_WORKERS=1

# System deps (xgboost -> libgomp1; build tools for wheels; curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python deps ----------
# copy only requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# NOTE: includes tensorflow-cpu to ensure keras_text model runs the same everywhere.
RUN python -m pip install --upgrade pip \
 && python -m pip install -r /app/requirements.txt

# ---------- Project code ----------
COPY . /app

# Make helper scripts executable
RUN chmod +x /app/scripts/build_all.sh || true

EXPOSE 8000

# Default command: start API
# We print the demo URL on start for convenience
CMD bash -lc 'echo "===========================================" && \
              echo " API running at:      http://127.0.0.1:8000" && \
              echo " Demo page available: http://127.0.0.1:8000/demo/demo.html" && \
              echo "===========================================" && \
              exec uvicorn service.api:app --host 0.0.0.0 --port 8000'
