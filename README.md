# RunPod Serverless XTTS v2 - Advanced Multilingual Voice Cloning TTS

High-performance, serverless Text-to-Speech (TTS) endpoint powered by **Coqui XTTS v2** model with zero-shot voice cloning, 17+ languages, real emotion effects, noise reduction, multispeaker support, and ultra-natural audio post-processing.

This worker runs on **RunPod Serverless** infrastructure using a custom Docker image and RunPod Python SDK.

## Model Details

- **Model Name**: XTTS-v2 (from Coqui TTS)
- **Hugging Face Repo**: [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
- **Library Version**: TTS==0.22.0
- **Sampling Rate**: 24 kHz
- **Output Format**: WAV (16-bit PCM)
- **Voice Cloning**: Zero-shot – works with just 6-10 seconds reference audio clip
- **Cross-language Cloning**: Yes – clone English voice → generate in Hindi, Spanish, etc.
- **Supported Languages** (17 officially supported):
  - English (en)
  - Spanish (es)
  - French (fr)
  - German (de)
  - Italian (it)
  - Portuguese (pt)
  - Polish (pl)
  - Turkish (tr)
  - Russian (ru)
  - Dutch (nl)
  - Czech (cs)
  - Arabic (ar)
  - Chinese (zh-cn)
  - Japanese (ja)
  - Korean (ko)
  - Hungarian (hu)
  - Hindi (hi)

## Key Features & Enhancements

- Zero-shot multispeaker voice cloning (multiple reference URLs supported)
- Multispeaker switching in single request using `[speaker1]Text[/speaker1]` tags
- 10+ Emotion simulation with real audio effects (pitch shift, speed, EQ, distortion, tremolo, whisper, etc.)
  - Supported emotions: neutral, happy, sad, angry, excited, calm, surprised, fearful, disgusted, whisper
- Speed control (0.5× to 2.0×)
- Advanced post-processing:
  - Noise reduction (noisereduce – stationary + dynamic)
  - High-pass filter & pre-emphasis for clarity
  - Dynamic EQ (boost mids/highs for natural tone)
  - Volume normalization (-3 dB peak)
  - Optional subtle reverb
- Output: Base64-encoded WAV (ready for direct playback in browser/apps)
- Pre-loading of model on worker startup → fast inference after cold start
- GPU acceleration (CUDA 12.1)
- Error handling with detailed messages
- Fallback to English if unsupported language provided

## Prerequisites

- RunPod account[](https://www.runpod.io)
- GitHub repository (public or private)
- GPU: Minimum RTX 4090 / A6000 / A5000 (10GB+ VRAM recommended)
- Reference audio files publicly accessible via HTTPS (for cloning)

## Deployment Steps (RunPod Serverless)

1. Create a new GitHub repository.
2. Add the following files exactly as provided in this repo:
   - `Dockerfile`
   - `requirements.txt`
   - `handler.py`
3. Go to RunPod Dashboard → Serverless → New Endpoint
4. Select **Import from GitHub** → choose your repo
5. Configure:
   - **GPU Type**: RTX 4090 / A6000 / equivalent (10GB+ VRAM)
   - **Container Disk**: 30 GB (model ~2GB + temp files)
   - **Active Workers**: 0–3 (scale as needed)
   - **Max Workers**: 1–10 (budget dependent)
   - **Flashboot**: Enabled (faster cold starts, optional)
   - **Environment Variables**: (none required currently)
6. Deploy → wait for build & active status
7. Copy the generated **Endpoint URL**[](https://api.runpod.ai/v2/xxxxxx/runs)

## Input JSON Format (POST to /runs)

```json
{
  "input": {
    "text": "Namaste bhai, yeh ultra natural TTS hai!",
    "language": "hi",
    "speaker_wav_urls": [
      "https://example.com/voice1.wav",
      "https://example.com/voice2.wav"
    ],
    "emotion": "excited",
    "speed": 1.2,
    "add_reverb": true,
    "normalize_volume": true
  }
}

text (required): Text to speak. Supports multispeaker tags like [speaker1]Hello[/speaker1] [speaker2]World[/speaker2]
language (optional, default: "en"): One of the 17 supported codes
speaker_wav_urls (optional array): List of public WAV URLs for cloning (order matches speaker1, speaker2, ...)
If only one voice → use single URL or old speaker_wav_url key (backward compatible)

emotion (optional, default: "neutral"): happy, sad, angry, excited, calm, surprised, fearful, disgusted, whisper
speed (optional, default: 1.0): Float 0.5–2.0
add_reverb (optional, default: false): Add subtle room reverb
normalize_volume (optional, default: true): Consistent loudness

Output JSON Format
Success:
JSON{
  "status": "success",
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "sample_rate": 24000,
  "format": "wav"
}
Error:
JSON{
  "error": "Detailed error: No text provided"
}
To play audio in browser:
HTML<audio controls src="data:audio/wav;base64,{audio_base64}"></audio>
Local Testing (Optional)
Bash# Install deps
pip install -r requirements.txt

# Run handler locally (simulate RunPod input)
python handler.py
(Note: Local run needs GPU + CUDA setup)
Troubleshooting

Build fails on pip install: Ensure torch is installed with CUDA index before requirements
Cold start slow (~15-40s first request): Model loads on startup – subsequent requests <3s
No audio output: Check reference WAV is clean, mono/stereo, 16kHz+ sample rate
Language not working: Falls back to English – use exact code like "hi", "zh-cn"
VRAM out of memory: Use larger GPU or reduce concurrent workers
Noisy output: Increase noise reduction strength in code if needed

Performance Notes

Inference time: ~0.8–1.5× realtime on RTX 4090
Cold start: 15–40 seconds (first request only)
Model size: ~2 GB (auto-downloaded on first load)

Security & Privacy

Reference audio downloaded temporarily → deleted after generation
No data stored permanently
All processing on RunPod ephemeral instances

Credits & License

Core model: Coqui XTTS-v2 (Apache 2.0 / open weights)
TTS library: Coqui TTS (Mozilla Public License 2.0)
Worker base: RunPod Python SDK
Enhancements: Custom audio processing (librosa, noisereduce, torchaudio)

Future Improvements (Possible)

Streaming output support
Real-time WebSocket endpoint
Fine-tuned emotion classifier integration
Batch processing multiple texts
