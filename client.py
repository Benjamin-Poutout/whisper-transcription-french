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

# Buffer pour accumuler les morceaux audio avant d'envoyer
audio_buffer = []

# Envoyer les données en continu
try:
    while True:
        # Lire un morceau d'audio
        audio_chunk = stream.read(CHUNK_SIZE)
        audio_buffer.append(audio_chunk)

        # Si le buffer atteint une certaine taille, envoyer l'audio en entier
        if len(audio_buffer) > 2:  # 2 morceaux accumulés, à ajuster
            audio_data = b''.join(audio_buffer)  # Joindre les morceaux audio
            response = requests.post(url, files={'file': ('audio.wav', audio_data, 'audio/wav')})

            # Afficher la transcription reçue
            if response.status_code == 200:
                transcription = response.json().get('transcription')
                print(f"Transcription: {transcription}")
            else:
                print(f"Erreur {response.status_code}: {response.text}")

            # Réinitialiser le buffer
            audio_buffer = []
        
        time.sleep(0.1)  # Petit délai pour limiter le nombre de requêtes

except KeyboardInterrupt:
    print("Arrêt de l'enregistrement...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
