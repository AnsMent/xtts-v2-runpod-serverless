FROM runpod/pytorch:2.4.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure latest PyTorch with CUDA 12.1 (override if base has older)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install coqui-tts with CUDA support + other deps
RUN pip install --no-cache-dir coqui-tts[cuda] runpod requests numpy scipy librosa noisereduce torchaudio

COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
