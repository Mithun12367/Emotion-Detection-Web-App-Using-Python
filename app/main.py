from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import base64
import io
from app.model import EmotionModel
from app.utils import detect_faces

app = FastAPI(title="Emotion Detection API", description="API for detecting emotions from images and video frames")

# CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
# We prioritize loading weights if they exist, otherwise the model will run with random weights (or user can train it)
emotion_model = EmotionModel()

# Mount Static Files (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        faces, original_img = detect_faces(image_bytes=contents)
        
        results = []
        for face_data in faces:
            face_roi = face_data['face'] # This is grayscale 48x48 or varying size
            
            # Predict
            emotion, confidence = emotion_model.predict(face_roi)
            
            x, y, w, h = face_data['box']
            results.append({
                "emotion": emotion,
                "confidence": float(confidence),
                "box": [x, y, w, h]
            })

        return JSONResponse(content={"faces": results})
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class FrameData(dict):
    # Depending on how we send data, usually JSON with base64 string
    pass

from pydantic import BaseModel

class ImagePayload(BaseModel):
    image: str # Base64 string

@app.post("/predict-frame")
async def predict_frame(payload: ImagePayload):
    try:
        # Decode base64
        image_data = base64.b64decode(payload.image.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces, _ = detect_faces(image_np=img)
        # print(f"Frame processed. Faces: {len(faces)}")
        
        results = []
        for face_data in faces:
            face_roi = face_data['face']
            emotion, confidence = emotion_model.predict(face_roi)
            
            x, y, w, h = face_data['box']
            results.append({
                "emotion": emotion,
                "confidence": float(confidence),
                "box": [x, y, w, h]
            })
            
        return JSONResponse(content={"faces": results})
    except Exception as e:
        # print(f"Error processing frame: {e}") # Reduce log noise
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
