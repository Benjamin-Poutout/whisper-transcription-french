import pyaudio
import requests
import time

# Paramètres de l'enregistrement audio
RATE = 16000  # Fréquence d'échantillonnage
CHUNK_SIZE = 32000  # Taille des morceaux d'audio
CHANNELS = 1  # Mono
FORMAT = pyaudio.paInt16  # Format audio
DEVICE_INDEX = None  # Utiliser le périphérique par défaut

# Créer une instance PyAudio
p = pyaudio.PyAudio()

# Ouvrir le flux d'enregistrement
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=DEVICE_INDEX)

# URL du serveur FastAPI
url = "http://127.0.0.1:8000/stream"

# Envoyer les données en continu
try:
    while True:
        # Lire un morceau d'audio
        audio_chunk = stream.read(CHUNK_SIZE)
        
        # Envoyer l'audio en tant que flux multipart
        response = requests.post(url, files={'file': ('audio.wav', audio_chunk, 'audio/wav')})
        
        # Afficher la transcription reçue
        if response.status_code == 200:
            transcription = response.json().get('transcription')
            print(f"Transcription: {transcription}")
        else:
            print("Erreur:", response.status_code)
        
        time.sleep(0.1)  # Petit délai pour limiter le nombre de requêtes

except KeyboardInterrupt:
    print("Arrêt de l'enregistrement...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
