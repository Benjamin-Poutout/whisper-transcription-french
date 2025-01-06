"""
fastapi whisper transcriber
"""

import logging
import tempfile
import os
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import soundfile as sf 
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.DEBUG)

# Configuration de base du modèle ASR
model_id = "bofenghuang/whisper-large-v3-french"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Lifespan context pour charger le modèle à la création de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")

    try:
        # Charger le modèle Whisper avec la gestion de la mémoire optimisée
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        
        # Charger le processeur pour Whisper
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
        logging.error(f"Error loading model: {e}")
        yield
        return
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route pour servir index.html
@app.get("/", response_class=HTMLResponse)
async def get():
    with open("static/index.html", "r") as f:
        return f.read()

# WebSocket pour recevoir et traiter l'audio
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected.")
    
    transcription = ""  # Transcription progressive

    try:
        while True:
            # Réception de l'audio en chunks
            audio_data = await websocket.receive_bytes()
            print(f"Received audio chunk of size: {len(audio_data)}")

            # Traitement et transcription du chunk en temps réel
            partial_transcription = await process_audio_chunk(audio_data)
            transcription += partial_transcription + " "  # Ajouter la transcription partielle
            
            # Envoi de la transcription partielle au client
            await websocket.send_text(transcription)
    
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
        print(f"Final transcription: {transcription}")

# Fonction de conversion WebM en WAV
async def process_audio_chunk(audio_data: bytes):
    try:
        # Enregistrer les données audio dans un fichier temporaire .webm
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
            print(f"Temporary file created at {tmp_file_path}")
        
        # Convertir .webm en .wav
        wav_path = tmp_file_path.replace('.webm', '.wav')
        conversion_result = convert_webm_to_wav(tmp_file_path, wav_path)
        if conversion_result is None:
            raise ValueError("Conversion failed")
        
        # Transcrire le fichier .wav
        transcription = await transcribe_audio(wav_path)
        
        # Supprimer les fichiers temporaires après utilisation
        os.remove(tmp_file_path)
        os.remove(wav_path)
        
        return transcription
    
    except Exception as e:
        logging.error(f"Error processing audio chunk: {e}")
        return f"Error processing audio chunk: {e}"

def convert_webm_to_wav(input_path: str, output_path: str):
    try:
        # Charger le fichier WebM avec AudioSegment
        print(f"Converting {input_path} to {output_path}")
        audio = AudioSegment.from_file(input_path, format="webm")
        
        # Convertir la fréquence d'échantillonnage à 16000 Hz
        audio = audio.set_frame_rate(16000)
        
        # Exporter au format WAV
        audio.export(output_path, format="wav")
        print(f"Conversion successful: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error converting WebM to WAV: {e}")
        return None

# Fonction pour effectuer la transcription de l'audio
async def transcribe_audio(file_path: str):
    try:
        # Transcrire l'audio avec le modèle
        audio_data, sample_rate = sf.read(file_path)
        
        # Assurer que la fréquence d'échantillonnage est correcte
        if sample_rate != 16000:
            raise ValueError(f"Expected sample rate: 16000, but got: {sample_rate}")
        
        # Transcrire l'audio avec le modèle
        result = model_pipeline(audio_data)
        transcription = result["text"]
        return transcription
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return "Error during transcription"
