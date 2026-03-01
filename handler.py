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

# =============================
# MODEL INITIALIZATION
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
print("✅ XTTS Loaded on", DEVICE)

# Speaker embedding cache
speaker_cache = {}

# =============================
# AUDIO ENHANCEMENT PIPELINE
# =============================

def enhance_audio(audio, sr=24000):
    # Noise reduction
    reduced = nr.reduce_noise(y=audio, sr=sr)

    # Trim silence
    trimmed, _ = librosa.effects.trim(reduced, top_db=25)

    # FFT EQ shaping
    fft = np.fft.rfft(trimmed)
    freqs = np.fft.rfftfreq(len(trimmed), 1/sr)

    # Warmth boost (low mids)
    fft[(freqs > 120) & (freqs < 300)] *= 1.1

    # Clarity boost
    fft[(freqs > 3000) & (freqs < 6000)] *= 1.05

    shaped = np.fft.irfft(fft)

    # Loudness normalization (-16 LUFS)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(shaped)
    normalized = pyln.normalize.loudness(shaped, loudness, -16.0)

    # Peak limiting
    normalized = np.clip(normalized, -0.99, 0.99)

    return normalized

# =============================
# SPEAKER CACHING
# =============================

def get_embedding(wav_path):
    if wav_path in speaker_cache:
        return speaker_cache[wav_path]

    emb = tts.get_speaker_embedding(wav_path)
    speaker_cache[wav_path] = emb
    return emb

# =============================
# HANDLER
# =============================

def handler(event):
    try:
        input_data = event.get("input", {})
        language = input_data.get("language", "en")

        dialogue = input_data.get("dialogue")
        text = input_data.get("text")
        speakers = input_data.get("speakers")

        full_audio = []

        # -------------------------
        # MULTI-SPEAKER MODE
        # -------------------------
        if dialogue and speakers:
            for turn in dialogue:
                speaker_id = turn["speaker"]
                speaker_b64 = speakers[speaker_id]
                speaker_bytes = base64.b64decode(speaker_b64)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_s:
                    tmp_s.write(speaker_bytes)
                    speaker_path = tmp_s.name

                wav = tts.tts(
                    text=turn["text"],
                    speaker_wav=speaker_path,
                    language=language,
                    temperature=0.35,
                    speed=1.0
                )

                full_audio.extend(wav)

                os.remove(speaker_path)

        # -------------------------
        # SINGLE SPEAKER MODE
        # -------------------------
        else:
            speaker_b64 = input_data.get("speaker_wav_base64")
            if not text or not speaker_b64:
                return {"error": "Missing text or speaker_wav_base64"}

            speaker_bytes = base64.b64decode(speaker_b64)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_s:
                tmp_s.write(speaker_bytes)
                speaker_path = tmp_s.name

            wav = tts.tts(
                text=text,
                speaker_wav=speaker_path,
                language=language,
                temperature=0.35,
                speed=1.0
            )

            full_audio = wav
            os.remove(speaker_path)

        full_audio = np.array(full_audio)

        # Apply mastering
        enhanced = enhance_audio(full_audio, 24000)

        # Save output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        sf.write(output_path, enhanced, 24000)

        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.remove(output_path)

        return {
            "audio_base64": output_b64
        }

    except Exception as e:
        return {"error": str(e)}

# =============================
# START SERVERLESS WORKER
# =============================

runpod.serverless.start({"handler": handler})
