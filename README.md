# XTTS Serverless (RunPod)

## Features
- XTTS v2 multilingual TTS
- Voice cloning
- Studio-quality mastering
- Noise reduction
- Loudness normalization (-16 LUFS)
- GPU accelerated
- Serverless auto scaling

## Input JSON

{
  "input": {
    "text": "Hello world",
    "language": "hi",
    "speaker_wav_base64": "BASE64_WAV"
  }
}

## Output

{
  "audio_base64": "BASE64_ENCODED_WAV"
}

## Deployment

1. Push to GitHub
2. Create RunPod Serverless endpoint
3. Import GitHub repo
4. Build context: /
5. Dockerfile path: Dockerfile
6. Min workers = 0
7. Deploy
