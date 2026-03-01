import runpod
import base64
import tempfile
import os
import torch
import numpy as np
import soundfile as sf
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
            print(f"Device: {device}")
            if device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
            print("XTTS v2 loaded successfully on", device)
        except Exception as e:
            print("CRITICAL: XTTS model loading failed!")
            print(traceback.format_exc())
            raise
    return tts

def handler(event):
    job_id = event.get("id", "unknown")
    print(f"Job started: {job_id}")
    try:
        input_data = event.get("input", {})
        text = input_data.get("text")
        language = input_data.get("language", "en")
        speaker_b64 = input_data.get("speaker_wav_base64")

        if not text:
            return {"error": "Text is required"}
        if not speaker_b64:
            return {"error": "speaker_wav_base64 is required"}

        print(f"Text: '{text[:50]}...' | Language: {language}")

        speaker_bytes = base64.b64decode(speaker_b64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_s:
            tmp_s.write(speaker_bytes)
            speaker_path = tmp_s.name

        tts_model = load_model()

        print("Starting TTS generation...")
        wav = tts_model.tts(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            temperature=0.65,
            speed=1.0
        )

        if len(wav) == 0:
            raise ValueError("TTS generated empty audio (0 samples)")

        print(f"TTS output samples: {len(wav)}")

        audio_np = np.array(wav, dtype=np.float32)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        sf.write(output_path, audio_np, 24000)

        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.remove(speaker_path)
        os.remove(output_path)

        print(f"Job {job_id} success")
        return {"audio_base64": output_b64, "status": "success"}

    except Exception as e:
        error_msg = f"Job {job_id} failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

runpod.serverless.start({"handler": handler})
