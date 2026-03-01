FROM runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy renamed requirements to avoid any conflict
COPY requirements-docker.txt requirements.txt

# Upgrade pip and install packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
