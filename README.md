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
Deployment

Create GitHub repo xtts-serverless
Add above 3 files + this README.md
RunPod Serverless → New Endpoint → Import GitHub repo (main)
GPU: RTX 4090 / A6000 (24GB VRAM)
Container Disk: 30 GB
Active Workers: 1
Max Workers: 3
Deploy
First build 20-50 min lag sakta hai
Logs check karo: "XTTS v2 loaded successfully"

Troubleshooting

Build fail → Logs dekho (pip error common)
No audio → Reference base64 valid WAV ka ho (duration 6-10 sec)
Blank output → Logs mein "empty audio" dikhega

textBhai, ab repo banao, files daal do, push kar do aur RunPod par new endpoint create karo. Yeh
