# My XTTS Serverless (RunPod)

## Features

- XTTS v2 multilingual TTS
- Voice cloning
- Multi-speaker dialogue mode
- Studio mastering pipeline
- Noise reduction
- Loudness normalization (-16 LUFS)
- Lip-sync ready clean WAV
- GPU accelerated
- Auto scaling (Serverless)

---

## Input Format (Single Speaker)

POST JSON:

{
  "input": {
    "text": "Hello world",
    "language": "hi",
    "speaker_wav_base64": "BASE64_WAV"
  }
}

---

## Multi-Speaker Format

{
  "input": {
    "language": "hi",
    "dialogue": [
      {"speaker": "A", "text": "Hello"},
      {"speaker": "B", "text": "Kaise ho"}
    ],
    "speakers": {
      "A": "BASE64_WAV_A",
      "B": "BASE64_WAV_B"
    }
  }
}

---

## Output

{
  "audio_base64": "BASE64_ENCODED_WAV"
}

---

## Deployment

1. Push to GitHub
2. Create RunPod Serverless Endpoint
3. Connect GitHub repo
4. Deploy with GPU
5. Set Min Workers = 0 (Cost Saving)

---

Cold start may take ~20-40 seconds (model load).
