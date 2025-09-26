# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps (git for editable installs if needed; build tools minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirement files first for better layer caching
COPY requirements-test.txt requirements-local.txt requirements-dashboard.txt ./

# Install python deps (tests/dev covers Flask/plotly if present there)
#
RUN pip install --upgrade pip && \
    pip install -r requirements-test.txt && \
    pip install -r requirements-local.txt && \
    pip install -r requirements-dashboard.txt

# Copy source
COPY . .

RUN pip install ibx-0.1.1/.

# Default command runs the monitoring dashboard; override in compose
EXPOSE 9090
CMD ["python", "monitoring/dashboard.py", "--port", "9090"]
