import runpod
import torch
import base64
import os
import requests
import numpy as np
from io import BytesIO
from scipy.io.wavfile import write
from TTS.api import TTS
import librosa
import noisereduce as nr
import torchaudio.functional as F
import torchaudio.effects as E
import re
import traceback
import sys

# Global model - None initially
model = None

def load_model():
    global model
    try:
        print("Attempting to load XTTS v2 model...")
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available! Falling back to CPU (slow).")
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", 
                    gpu=torch.cuda.is_available())
        print("XTTS v2 model loaded successfully on device:", "GPU" if torch.cuda.is_available() else "CPU")
    except Exception as e:
        print("CRITICAL: Model load failed!")
        print(traceback.format_exc())
        raise  # Re-raise to crash worker if needed, but log first

def download_reference_audio(url: str, job_id: str, index: int) -> str:
    try:
        temp_path = f"/tmp/ref_{job_id}_{index}.wav"
        print(f"Downloading reference audio from {url} to {temp_path}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(temp_path, "wb") as f:
            f.write(response.content)
        print("Reference audio downloaded successfully")
        return temp_path
    except Exception as e:
        print(f"Download failed for {url}: {str(e)}")
        raise

def parse_multispeaker_text(text: str) -> list:
    pattern = r'\[speaker(\d+)\](.*?)\[/speaker\1\]'
    matches = re.findall(pattern, text)
    if matches:
        return [(int(speaker_idx) - 1, segment) for speaker_idx, segment in matches]
    return [(0, text)]

def get_emotion_params(emotion: str):
    emotions = {
        "neutral": {"temperature": 0.85, "top_p": 0.85, "repetition_penalty": 5.0},
        "happy": {"temperature": 0.95, "top_p": 0.90, "repetition_penalty": 4.0},
        "sad": {"temperature": 0.70, "top_p": 0.75, "repetition_penalty": 6.0},
        "angry": {"temperature": 0.90, "top_p": 0.80, "repetition_penalty": 5.5},
        "excited": {"temperature": 1.00, "top_p": 0.95, "repetition_penalty": 3.5},
        "calm": {"temperature": 0.60, "top_p": 0.70, "repetition_penalty": 7.0},
        "surprised": {"temperature": 0.95, "top_p": 0.85, "repetition_penalty": 4.5},
        "fearful": {"temperature": 0.80, "top_p": 0.75, "repetition_penalty": 5.5},
        "disgusted": {"temperature": 0.85, "top_p": 0.80, "repetition_penalty": 6.0},
        "whisper": {"temperature": 0.50, "top_p": 0.60, "repetition_penalty": 8.0}
    }
    return emotions.get(emotion.lower(), emotions["neutral"])

def apply_emotion_effects(wav_tensor: torch.Tensor, sr: int, emotion: str) -> torch.Tensor:
    emotion = emotion.lower()
    try:
        if emotion == "happy":
            wav_tensor = E.pitch_shift(wav_tensor, sr, n_steps=4)
            wav_tensor = E.speed(wav_tensor, sr, factor=1.1)[0]
            wav_tensor = F.equalizer_biquad(wav_tensor, sr, center_freq=3000, gain=3.0, q=1.0)
        elif emotion == "sad":
            wav_tensor = E.pitch_shift(wav_tensor, sr, n_steps=-2)
            wav_tensor = E.speed(wav_tensor, sr, factor=0.9)[0]
            wav_tensor = F.equalizer_biquad(wav_tensor, sr, center_freq=500, gain=-2.0, q=1.0)
        elif emotion == "angry":
            wav_tensor = E.pitch_shift(wav_tensor, sr, n_steps=2)
            wav_tensor = E.gain(wav_tensor, gain_db=3.0)
            wav_tensor = F.equalizer_biquad(wav_tensor, sr, center_freq=2000, gain=4.0, q=0.5)
            wav_tensor = torch.clamp(wav_tensor * 1.2, -1.0, 1.0)
        elif emotion == "excited":
            wav_tensor = E.pitch_shift(wav_tensor, sr, n_steps=3)
            wav_tensor = E.speed(wav_tensor, sr, factor=1.15)[0]
            wav_tensor = F.equalizer_biquad(wav_tensor, sr, center_freq=4000, gain=2.5, q=1.0)
        elif emotion == "calm":
            wav_tensor = E.pitch_shift(wav_tensor, sr, n_steps=-1)
            wav_tensor = E.speed(wav_tensor, sr, factor=0.95)[0]
            wav_tensor = F.equalizer_biquad(wav_tensor, sr, center_freq=1000, gain=-1.5, q=1.0)
        elif emotion == "surprised":
            wav_tensor = E.pitch_shift(wav_tensor, sr, n_steps=5)
            wav_tensor = E.speed(wav_tensor, sr, factor=1.05)[0]
        elif emotion == "fearful":
            wav_tensor = E.pitch_shift(wav_tensor, sr, n_steps=-3)
            wav_tensor = E.tremolo(wav_tensor, sr, freq=8.0, depth=0.3)
        elif emotion == "disgusted":
            wav_tensor = E.pitch_shift(wav_tensor, sr, n_steps=-1)
            wav_tensor = F.equalizer_biquad(wav_tensor, sr, center_freq=800, gain=-3.0, q=1.0)
        elif emotion == "whisper":
            wav_tensor = E.gain(wav_tensor, gain_db=-6.0)
            wav_tensor = F.highpass_biquad(wav_tensor, sr, cutoff_freq=2000, q=0.707)
        return wav_tensor
    except Exception as e:
        print(f"Emotion effect failed for {emotion}: {str(e)}")
        return wav_tensor  # Fallback to original

def enhance_audio(wav_np: np.ndarray, sr: int = 24000, add_reverb: bool = False, normalize_volume: bool = True, emotion: str = "neutral") -> np.ndarray:
    try:
        wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
        wav_tensor = apply_emotion_effects(wav_tensor, sr, emotion)
        enhanced = wav_tensor.squeeze(0).numpy()
        reduced_noise = nr.reduce_noise(y=enhanced, sr=sr, stationary=True, prop_decrease=0.90, n_std_thresh_stationary=1.5)
        high_pass = librosa.effects.preemphasis(reduced_noise, coef=0.98)
        enhanced = F.equalizer_biquad(torch.from_numpy(high_pass).float(), sample_rate=sr, 
                                     center_freq=800, gain=3.0, q=1.2).numpy()
        enhanced = F.equalizer_biquad(torch.from_numpy(enhanced).float(), sample_rate=sr, 
                                     center_freq=3000, gain=2.5, q=1.0).numpy()
        if normalize_volume:
            enhanced = librosa.util.normalize(enhanced, norm=np.inf, threshold=-3.0)
        if add_reverb:
            reverb = np.convolve(enhanced, np.exp(-np.linspace(0, 1, int(sr * 0.3))) * 0.3, mode='same')
            enhanced = enhanced * 0.7 + reverb * 0.3
        return enhanced
    except Exception as e:
        print(f"Audio enhancement failed: {str(e)}")
        return wav_np  # Fallback

def handler(job):
    print(f"Job received: {job['id']}")
    try:
        job_input = job["input"]
        text = job_input.get("text")
        language = job_input.get("language", "en")
        speaker_wav_urls = job_input.get("speaker_wav_urls", [])
        if not speaker_wav_urls:
            speaker_wav_urls = [job_input.get("speaker_wav_url")] if job_input.get("speaker_wav_url") else []
        add_reverb = job_input.get("add_reverb", False)
        normalize_volume = job_input.get("normalize_volume", True)
        speed = job_input.get("speed", 1.0)
        emotion = job_input.get("emotion", "neutral")

        if not text:
            return {"error": "text is required"}

        if language not in ["en", "hi", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"]:
            language = "en"
            print("Language fallback to en")

        # Load model only when needed (prevents startup crash)
        global model
        if model is None:
            load_model()

        tts_model = model

        segments = parse_multispeaker_text(text)

        ref_paths = [None] * len(speaker_wav_urls)
        for i, url in enumerate(speaker_wav_urls):
            if url:
                ref_paths[i] = download_reference_audio(url, job["id"], i)

        full_wav = []
        emotion_params = get_emotion_params(emotion)

        for speaker_idx, segment in segments:
            if speaker_idx >= len(ref_paths):
                speaker_idx = 0
            ref_path = ref_paths[speaker_idx]
            print(f"Generating segment for speaker {speaker_idx+1}: {segment[:50]}...")
            wav_segment = tts_model.tts(
                text=segment.strip(),
                speaker_wav=ref_path,
                language=language,
                split_sentences=True,
                speed=speed,
                **emotion_params
            )
            full_wav.extend(wav_segment)

        wav_np = np.array(full_wav, dtype=np.float32)
        enhanced_wav = enhance_audio(wav_np, sr=24000, add_reverb=add_reverb, normalize_volume=normalize_volume, emotion=emotion)
        wav_int16 = (enhanced_wav * 32767).astype(np.int16)

        buffer = BytesIO()
        write(buffer, 24000, wav_int16)
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        # Cleanup
        for ref_path in ref_paths:
            if ref_path and os.path.exists(ref_path):
                os.remove(ref_path)
                print(f"Cleaned up {ref_path}")

        print(f"Job {job['id']} completed successfully")
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "sample_rate": 24000,
            "format": "wav"
        }

    except Exception as e:
        error_msg = f"Handler error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    try:
        # Do NOT load model here - delay to handler
        runpod.serverless.start({"handler": handler})
    except Exception as startup_error:
        print(f"Startup critical error: {str(startup_error)}")
        print(traceback.format_exc())
        sys.exit(1)
