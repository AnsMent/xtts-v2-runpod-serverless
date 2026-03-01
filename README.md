# XTTS v2 Serverless Deployment

This repository contains a production-ready Docker image for deploying XTTS v2 on RunPod Serverless.

## Features

- GPU acceleration (CUDA)
- FastAPI endpoint
- API key authentication
- Multilingual XTTS v2
- Speaker voice cloning support

## Endpoint

POST /generate

Headers:
x-api-key: YOUR_SECRET_KEY

Form-data:
text: Text to synthesize
language: Target language (e.g. "hi", "en")
speaker_wav: WAV file

Response:
audio/wav binary

## Environment Variable

XTTS_API_KEY=your_secure_key_here

## Deployment

1. Push this repo to GitHub
2. Create RunPod Serverless endpoint
3. Select GPU (RTX A4500)
4. Set environment variable XTTS_API_KEY
5. Deploy
