# XTTS Serverless (RunPod) - 2026 Stable Version

Voice cloning TTS endpoint using XTTS v2.

## Features
- Zero-shot voice cloning
- 17+ languages
- Noise reduction, EQ, loudness normalization (-16 LUFS)
- Base64 input/output
- GPU acceleration

## Input JSON

```json
{
  "input": {
    "text": "Namaste bhai, yeh final setup se ban raha hai",
    "language": "hi",
    "speaker_wav_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
  }
}
Output
JSON{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "status": "success"
}
