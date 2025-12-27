const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-camera');
const stopBtn = document.getElementById('stop-camera');
const emotionLabel = document.getElementById('dominant-emotion');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const logList = document.getElementById('log-list');

let stream = null;
let intervalId = null;
let isRunning = false;

// Buttons
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

// Tab Switching
const webcamTab = document.getElementById('mode-webcam');
const uploadTab = document.getElementById('mode-upload');
const webcamSection = document.getElementById('webcam-section');
const uploadSection = document.getElementById('upload-section');

webcamTab.addEventListener('click', () => {
    webcamTab.classList.add('active');
    uploadTab.classList.remove('active');
    webcamSection.classList.add('active-section');
    uploadSection.classList.remove('active-section');
});

uploadTab.addEventListener('click', () => {
    uploadTab.classList.add('active');
    webcamTab.classList.remove('active');
    uploadSection.classList.add('active-section');
    webcamSection.classList.remove('active-section');
    stopCamera();
});

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        startBtn.disabled = true;
        stopBtn.disabled = false;

        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isRunning = true;
            processVideo();
        };
    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Could not access camera. Please allow permissions.");
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        isRunning = false;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

async function processVideo() {
    if (!isRunning) return;

    // Draw video frame to a temporary canvas to get base64
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);

    try {
        const response = await fetch('/predict-frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const result = await response.json();

        drawResults(result.faces);
    } catch (err) {
        console.error("Prediction error:", err);
    }

    if (isRunning) {
        // requestAnimationFrame(processVideo); // Too fast?
        setTimeout(processVideo, 200); // 5 FPS roughly
    }
}

function drawResults(faces) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings

    if (!faces || faces.length === 0) {
        emotionLabel.innerText = "No Face";
        confidenceBar.style.width = '0%';
        confidenceText.innerText = '0%';
        return;
    }

    // Process first face for main display (could be improved)
    const mainFace = faces[0];
    updateUI(mainFace);

    faces.forEach(face => {
        const [x, y, w, h] = face.box;
        const emotion = face.emotion;
        const conf = Math.round(face.confidence * 100);

        // Draw Box
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Draw Text Background
        ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
        ctx.fillRect(x, y - 30, w, 30);

        // Draw Text
        ctx.fillStyle = '#22d3ee';
        ctx.font = '16px Outfit';
        ctx.fillText(`${emotion} ${conf}%`, x + 5, y - 10);
    });
}

function updateUI(face) {
    emotionLabel.innerText = face.emotion;
    const conf = Math.round(face.confidence * 100);
    confidenceBar.style.width = `${conf}%`;
    confidenceText.innerText = `${conf}%`;

    addToLog(face.emotion, conf);
}

function addToLog(emotion, conf) {
    const li = document.createElement('li');
    const time = new Date().toLocaleTimeString();
    li.innerHTML = `[${time}] Detected <span>${emotion}</span> (${conf}%)`;
    logList.prepend(li);
    if (logList.children.length > 20) {
        logList.removeChild(logList.lastChild);
    }
}

// Upload Handling
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const imageOverlay = document.getElementById('image-overlay');
const previewContainer = document.getElementById('image-preview-container');

const changeImageBtn = document.getElementById('change-image-btn');

changeImageBtn.addEventListener('click', () => {
    previewContainer.classList.add('hidden');
    dropZone.style.display = 'block';
    fileInput.value = '';
    imagePreview.src = '';
    const ctxImg = imageOverlay.getContext('2d');
    ctxImg.clearRect(0, 0, imageOverlay.width, imageOverlay.height);
    document.getElementById('upload-loader').classList.add('hidden'); // Ensure loader is hidden

    // Reset UI results
    emotionLabel.innerText = "--";
    confidenceBar.style.width = '0%';
    confidenceText.innerText = '0%';
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#22d3ee';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

async function handleFile(file) {
    if (!file) return;

    // Show Preview
    previewContainer.classList.remove('hidden');
    dropZone.style.display = 'none';

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.onload = () => {
            imageOverlay.width = imagePreview.width; // This might be natural width?
            // Need to handle layout sizing. For simplicity, just set canvas to client dims
            // Actually, best to get rect
            imageOverlay.width = imagePreview.clientWidth;
            imageOverlay.height = imagePreview.clientHeight;
        };
    };
    reader.readAsDataURL(file);

    // Upload
    const formData = new FormData();
    formData.append('file', file);

    const loader = document.getElementById('upload-loader');
    loader.classList.remove('hidden'); // Show loader

    try {
        const response = await fetch('/predict-image', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        loader.classList.add('hidden'); // Hide loader

        // Draw on image overlay
        const ctxImg = imageOverlay.getContext('2d');
        ctxImg.clearRect(0, 0, imageOverlay.width, imageOverlay.height);

        // We need to scale the boxes if the image is displayed at a different size than original
        // Let's assume server returns boxes for original image coordinates.
        // We need the scale factor.
        // For this demo, let's just use the server result and assume 1:1 or handle scaling simply if possible
        // To do it right: get naturalWidth/Height vs clientWidth/Height.

        const scaleX = imagePreview.clientWidth / imagePreview.naturalWidth;
        const scaleY = imagePreview.clientHeight / imagePreview.naturalHeight;

        if (result.faces && result.faces.length > 0) {
            result.faces.forEach(face => {
                const [x, y, w, h] = face.box;

                const sx = x * scaleX;
                const sy = y * scaleY;
                const sw = w * scaleX;
                const sh = h * scaleY;

                ctxImg.strokeStyle = '#22d3ee';
                ctxImg.lineWidth = 3;
                ctxImg.strokeRect(sx, sy, sw, sh);

                ctxImg.fillStyle = 'rgba(15, 23, 42, 0.8)';
                ctxImg.fillRect(sx, sy - 30, sw, 30);

                ctxImg.fillStyle = '#22d3ee';
                ctxImg.font = '16px Outfit';
                ctxImg.fillText(`${face.emotion} ${Math.round(face.confidence * 100)}%`, sx + 5, sy - 10);
            });
            updateUI(result.faces[0]);
        }

    } catch (err) {
        console.error(err);
    }
}
