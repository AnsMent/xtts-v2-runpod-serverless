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
import traceback

# Global model - lazy load to avoid startup crash
tts = None

def load_model():
    global tts
    if tts is None:
        try:
            print("Loading XTTS v2 model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device detected: {device}")
            if device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
            else:
                print("WARNING: Running on CPU - very slow inference expected")
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
            print("XTTS v2 loaded successfully on", device)
        except Exception as load_error:
            print("CRITICAL: XTTS model loading failed!")
            print(traceback.format_exc())
            raise  # Crash worker with log for debugging
    return tts

def enhance_audio(audio, sr=24000):
    try:
        print("Starting audio enhancement...")
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
        print("Audio enhancement completed")
        return normalized
    except Exception as e:
        print("Enhancement failed:", str(e))
        traceback.print_exc()
        return audio  # Fallback to original audio

def handler(event):
    print("Job received:", event.get("id", "unknown"))
    try:
        input_data = event.get("input", {})
        text = input_data.get("text")
        language = input_data.get("language", "en")
        speaker_b64 = input_data.get("speaker_wav_base64")

        if not text or not speaker_b64:
            return {"error": "Missing text or speaker_wav_base64"}

        print(f"Processing text: '{text[:50]}...' | Language: {language}")

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

        print("Job completed successfully")
        return {"audio_base64": output_b64}

    except Exception as e:
        error_msg = f"Handler error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

runpod.serverless.start({"handler": handler})
