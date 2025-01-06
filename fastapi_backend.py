"""
fastapi whisper transcriber
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import uvicorn
import io
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Request, WebSocketDisconnect
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import logging
import numpy as np
import requests
import httpx
import soundfile as sf  # This will help in reading WAV files
from fastapi.middleware.cors import CORSMiddleware
# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

SAMPLE_RATE = 16000
SEGMENT_DURATION = 3  # 3 secondes
SEGMENT_SIZE = SAMPLE_RATE * SEGMENT_DURATION 


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Actions to perform at application startup
    """
    print("Starting up...")

    # Switch to a more reliable Whisper model
    model_id = "bofenghuang/whisper-large-v3-french"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        global model_pipeline
        model_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    except Exception as e:
        logging.error(f"Error loading model or processor: {e}")
        yield
        return

    yield  # The application runs while this generator is suspended
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict the allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get():
    with open("static/index.html", "r") as f:
        return f.read()

# WebSocket pour envoyer la transcription en temps réel
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    audio_buffer = bytearray()  # Buffer pour accumuler les morceaux audio

    try:
        while True:
            # Recevoir un morceau d'audio
            audio_chunk = await websocket.receive_bytes()
            audio_buffer.extend(audio_chunk)

            # Vérifier si le buffer contient un segment de 3 secondes
            if len(audio_buffer) >= SEGMENT_SIZE:
                # Extraire un segment de 3 secondes (SEGMENT_SIZE)
                segment = audio_buffer[:SEGMENT_SIZE]
                audio_buffer = audio_buffer[SEGMENT_SIZE:]  # Réinitialiser le buffer pour les prochains morceaux

                # Convertir le segment en numpy array de type int16
                audio_data = np.frombuffer(segment, dtype=np.int16)

                # Normalisation des données audio en float32
                audio_data = audio_data.astype(np.float32) / 32768.0

                # Transcrire l'audio
                transcription = model_pipeline(audio_data)["text"]

                # Envoyer la transcription au client WebSocket
                await websocket.send_text(transcription)

    except Exception as e:
        logging.error(f"Erreur WebSocket: {e}")
        await websocket.close()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Endpoint to transcribe uploaded audio chunk
    """
    try:
        # Read the uploaded file
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)

        # Log the file size to check if it's received correctly
        logging.debug(f"Received file of size {len(audio_bytes)} bytes")

        # Load the audio data using soundfile (sf)
        audio_data, sample_rate = sf.read(audio_file)

        # Log the audio sample rate and shape
        logging.debug(f"Audio sample rate: {sample_rate}, audio shape: {audio_data.shape}")

        # Preprocess the audio and use the model for transcription
        transcription = model_pipeline(audio_data)["text"]
        logging.debug(f"Transcription: {transcription}")

        # Return the transcription as part of the response
        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        logging.error("Error processing transcription: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
        
@app.post("/stream")
async def stream_transcribe(request: Request):
    audio_buffer = bytearray()
    async for chunk in request.stream():
        audio_buffer.extend(chunk)

    if len(audio_buffer) < SEGMENT_SIZE:
        return {"message": "Segment audio trop court"}

    # Convertir en numpy array et normaliser
    audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalisation
    transcription = model_pipeline(audio_data)["text"]
    return {"transcription": transcription}

def main():
    """Run the FastAPI app using uvicorn."""
    uvicorn.run("fastapi_backend:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
