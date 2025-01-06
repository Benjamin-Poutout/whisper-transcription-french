let socket;
let mediaRecorder;
let audioChunks = [];

// Connexion WebSocket
socket = new WebSocket("ws://localhost:8000/ws");

socket.onopen = () => {
    console.log("Connected to WebSocket");
};

socket.onmessage = (event) => {
    // Afficher la transcription reçue du serveur
    const transcription = event.data;
    document.getElementById("transcription").innerText = transcription;
};

socket.onclose = () => {
    console.log("Disconnected from WebSocket");
};

// Demander la permission pour accéder au micro et démarrer l'enregistrement
const startButton = document.getElementById("startRecording");
const stopButton = document.getElementById("stopRecording");

if (startButton) {
    startButton.onclick = async function () {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

        startButton.disabled = true;
        stopButton.disabled = false;

        // Capture audio en chunks et envoi en temps réel
        mediaRecorder.ondataavailable = (event) => {
            // Ajouter les morceaux à notre tableau
            audioChunks.push(event.data);

            // Envoyer chaque chunk d'audio au serveur en temps réel
            socket.send(event.data);
        };

        mediaRecorder.onstop = () => {
            console.log("Recording stopped.");
            // Lorsque l'enregistrement s'arrête, vider les morceaux d'audio
            audioChunks = [];
        };

        // Démarrer l'enregistrement
        mediaRecorder.start();
        console.log("Recording started...");
    };
} else {
    console.error("Start button not found.");
}

if (stopButton) {
    stopButton.onclick = () => {
        mediaRecorder.stop();
        startButton.disabled = false;
        stopButton.disabled = true;
        console.log("Recording stopped.");
    };
} else {
    console.error("Stop button not found.");
}
