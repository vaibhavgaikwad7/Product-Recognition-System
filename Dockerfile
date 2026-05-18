FROM python:3.11-slim

WORKDIR /app

# OpenCV runtime deps (libGL etc.) - using opencv-python-headless still needs libglib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source, trained model, and seeded catalog must exist before docker build
COPY src/ ./src/
COPY models/ ./models/
COPY catalog/ ./catalog/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
