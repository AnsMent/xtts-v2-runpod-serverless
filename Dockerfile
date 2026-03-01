FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy renamed requirements to avoid any buildpack conflict
COPY requirements-docker.txt requirements.txt

# Upgrade pip and install packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
