import runpod
import base64
import tempfile
import os
import torch
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import noisereduce as nr
from TTS.api import TTS

# Model load - delayed to handler first call (startup safe)
tts = None

def load_model():
    global tts
    if tts is None:
        print("Loading XTTS v2 model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
        print("XTTS v2 loaded on", device)
    return tts

def enhance_audio(audio, sr=24000):
    try:
        reduced = nr.reduce_noise(y=audio, sr=sr)
        trimmed, _ = librosa.effects.trim(reduced, top_db=25)
        fft = np.fft.rfft(trimmed)
        freqs = np.fft.rfftfreq(len(trimmed), 1/sr)
        fft[(freqs > 120) & (freqs < 300)] *= 1.1
        fft[(freqs > 3000) & (freqs < 6000)] *= 1.05
        shaped = np.fft.irfft(fft)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(shaped)
        normalized = pyln.normalize.loudness(shaped, loudness, -16.0)
        normalized = np.clip(normalized, -0.99, 0.99)
        return normalized
    except Exception as e:
        print("Enhance failed:", str(e))
        return audio

def handler(event):
    try:
        input_data = event.get("input", {})
        text = input_data.get("text")
        language = input_data.get("language", "en")
        speaker_b64 = input_data.get("speaker_wav_base64")

        if not text or not speaker_b64:
            return {"error": "Missing text or speaker_wav_base64"}

        speaker_bytes = base64.b64decode(speaker_b64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_s:
            tmp_s.write(speaker_bytes)
            speaker_path = tmp_s.name

        tts_model = load_model()

        wav = tts_model.tts(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            temperature=0.35,
            speed=1.0
        )

        audio_np = np.array(wav)
        enhanced = enhance_audio(audio_np, 24000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        sf.write(output_path, enhanced, 24000)

        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.remove(speaker_path)
        os.remove(output_path)

        return {"audio_base64": output_b64}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
