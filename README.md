# XTTS Serverless (RunPod) - Stable Version

## Features
- XTTS v2 multilingual TTS with voice cloning
- Noise reduction + loudness normalization (-16 LUFS)
- GPU accelerated
- Serverless auto-scaling
- Input via base64 speaker audio (secure & easy from WP)

## Input JSON Example

```json
{
  "input": {
    "text": "Namaste bhai, yeh test audio hai",
    "language": "hi",
    "speaker_wav_base64": "UklGRiQAAABXQVZFZm10IBIAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="  // real base64 here
  }
