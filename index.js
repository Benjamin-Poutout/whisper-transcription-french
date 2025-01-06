let mediaRecorder;
let audioChunks = [];
let socket;

// Fonction pour initialiser la connexion WebSocket
const startWebSocketConnection = () => {
    socket = new WebSocket("ws://127.0.0.1:8000/ws");

    socket.onopen = () => {
        console.log("WebSocket connecté.");
    };

    socket.onmessage = (event) => {
        console.log("Message reçu:", event.data);  // Log pour voir exactement ce qui est reçu
        document.getElementById("transcription").innerText = event.data; // Affiche le texte brut
    };

    socket.onerror = (error) => {
        console.error("Erreur WebSocket:", error);
    };

    socket.onclose = () => {
        console.log("Connexion WebSocket fermée.");
    };
};

// Fonction pour envoyer un chunk audio via WebSocket
const sendAudioChunk = (chunk) => {
    if (socket.readyState !== WebSocket.OPEN) {
        console.error("WebSocket n'est pas ouvert. Impossible d'envoyer le chunk.");
        return;
    }

    const sampleRate = 16000; // Exemple de taux d'échantillonnage
    const chunkDurationSeconds = 3; // Durée d'un chunk en secondes
    const chunkSize = sampleRate * chunkDurationSeconds * 2; // 2 octets par échantillon pour PCM 16 bits
    let start = 0;

    // Si le chunk est un Blob, le convertir en ArrayBuffer
    if (chunk instanceof Blob) {
        chunk.arrayBuffer().then(buffer => {
            console.log("Conversion du Blob en ArrayBuffer réussi, taille:", buffer.byteLength);

            // Diviser l'audio en morceaux et envoyer chaque morceau
            while (start < buffer.byteLength) {
                const end = Math.min(start + chunkSize, buffer.byteLength);
                const chunkToSend = buffer.slice(start, end);
                console.log("Envoi du chunk de taille:", chunkToSend.byteLength);
                socket.send(chunkToSend);
                start = end;
            }
        }).catch(error => {
            console.error("Erreur de conversion du Blob en ArrayBuffer:", error);
        });
    } else if (chunk instanceof ArrayBuffer) {
        // Si c'est déjà un ArrayBuffer
        while (start < chunk.byteLength) {
            const end = Math.min(start + chunkSize, chunk.byteLength);
            const chunkToSend = chunk.slice(start, end);
            console.log("Envoi du chunk de taille:", chunkToSend.byteLength);
            socket.send(chunkToSend);
            start = end;
        }
    } else {
        console.error("Le chunk n'est ni un Blob ni un ArrayBuffer, type:", typeof chunk);
    }
};

// Démarrer l'enregistrement et la connexion WebSocket
document.getElementById("startRecording").addEventListener("click", async () => {
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    // Démarrer la connexion WebSocket
    startWebSocketConnection();

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
        console.log("Chunk audio disponible:", event.data);
        // Envoyer chaque chunk audio à travers WebSocket
        sendAudioChunk(event.data);
    };

    mediaRecorder.onstop = () => {
        console.log("Enregistrement arrêté.");
        document.getElementById("startRecording").disabled = false;
        document.getElementById("stopRecording").disabled = true;
    };

    mediaRecorder.start(3000);  // Capture l'audio par morceaux de 500 ms
    console.log("Enregistrement démarré.");
    document.getElementById("startRecording").disabled = true;
    document.getElementById("stopRecording").disabled = false;
});

// Arrêter l'enregistrement
document.getElementById("stopRecording").addEventListener("click", () => {
    console.log("Arrêt de l'enregistrement...");
    mediaRecorder.stop();
    document.getElementById("startRecording").disabled = false;
    document.getElementById("stopRecording").disabled = true;
});
