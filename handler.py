import runpod
import torch
import base64
import os
import requests
import numpy as np
from io import BytesIO
from scipy.io.wavfile import write
from TTS.api import TTS

# Global model (ek baar load hota hai worker start hone par)
model = None

def load_model():
    global model
    if model is None:
        print("Loading XTTS v2 model...")
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", 
                   gpu=torch.cuda.is_available())
        print("XTTS v2 loaded successfully!")
    return model

def download_reference_audio(url: str, job_id: str) -> str:
    """Download speaker wav to temp file"""
    temp_path = f"/tmp/ref_{job_id}.wav"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(temp_path, "wb") as f:
        f.write(response.content)
    return temp_path

def handler(job):
    try:
        job_input = job["input"]
        
        text = job_input.get("text")
        language = job_input.get("language", "en")
        speaker_wav_url = job_input.get("speaker_wav_url")  # voice cloning ke liye reference audio URL
        
        if not text:
            return {"error": "text is required"}
        
        # Model load
        tts_model = load_model()
        
        # Reference audio download (voice cloning)
        ref_path = None
        if speaker_wav_url:
            ref_path = download_reference_audio(speaker_wav_url, job["id"])
        
        # Generate speech
        wav = tts_model.tts(
            text=text,
            speaker_wav=ref_path,
            language=language,
            split_sentences=True
        )
        
        # Convert to WAV bytes (24kHz, 16-bit)
        wav_np = np.array(wav, dtype=np.float32)
        wav_int16 = (wav_np * 32767).astype(np.int16)
        
        buffer = BytesIO()
        write(buffer, 24000, wav_int16)
        buffer.seek(0)
        
        # Base64
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        # Cleanup
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "sample_rate": 24000,
            "format": "wav"
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    load_model()  # preload on worker start
    runpod.serverless.start({"handler": handler})
