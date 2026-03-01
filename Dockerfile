FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir coqui-tts runpod==1.6.2 numpy==1.26.4 scipy==1.13.1 librosa==0.10.2 noisereduce==3.0.3 soundfile==0.12.1

COPY handler.py .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["python3", "-u", "handler.py"]
