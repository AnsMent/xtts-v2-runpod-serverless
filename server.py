import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import StreamingResponse
from TTS.api import TTS

# ====== CONFIG ======
API_KEY = os.getenv("XTTS_API_KEY", "change_this_key")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== LOAD MODEL ======
print("Loading XTTS v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
print("Model loaded.")
print("GPU Available:", torch.cuda.is_available())

app = FastAPI()


@app.post("/generate")
async def generate_voice(
    text: str = Form(...),
    language: str = Form(...),
    speaker_wav: UploadFile = File(...),
    x_api_key: str = Header(None)
):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    speaker_audio = await speaker_wav.read()

    temp_speaker_path = "/tmp/speaker.wav"
    with open(temp_speaker_path, "wb") as f:
        f.write(speaker_audio)

    output_path = "/tmp/output.wav"

    tts.tts_to_file(
        text=text,
        speaker_wav=temp_speaker_path,
        language=language,
        file_path=output_path
    )

    def iterfile():
        with open(output_path, "rb") as f:
            yield from f

    return StreamingResponse(iterfile(), media_type="audio/wav")
