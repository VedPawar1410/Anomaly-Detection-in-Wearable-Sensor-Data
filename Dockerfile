# API image: runtime requirements only (no torch, no training data).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY wearable_anomaly/ wearable_anomaly/
COPY api/ api/

# Don't run as root inside the container.
RUN useradd --create-home app
USER app

EXPOSE 8000
# Render (and most hosts) pass the port in $PORT.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
