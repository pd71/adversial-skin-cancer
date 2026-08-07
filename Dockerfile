# Production Dockerfile for deploying Skin Cancer CDSS Backend to Hugging Face Spaces
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (for OpenCV, ReportLab PDF generation, and fault handler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt-get/lists/*

# Copy backend requirements first for caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
    pip install --no-cache-dir gunicorn huggingface_hub

# Copy full application code
COPY backend /app/backend

# Set environment variables for Hugging Face Spaces
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PYTHONPATH=/app/backend:/app/backend/src

# Expose default Hugging Face Spaces port
EXPOSE 7860

WORKDIR /app/backend

# Execute model download & launch Flask API via Gunicorn worker
CMD ["sh", "-c", "python download_models.py && gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 120 app:app"]
