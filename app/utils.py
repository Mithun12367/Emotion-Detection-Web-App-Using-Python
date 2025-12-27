import cv2
import numpy as np
import os

import sys

# face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# Move init inside function to avoid global state issues in server

def detect_faces(image_bytes=None, image_np=None):
    """
    Detects faces in an image.
    Args:
        image_bytes: Image data in bytes (from file upload)
        image_np: Image data as numpy array (from video frame)
    Returns:
        List of dictionaries: [{'face': numpy_array, 'box': (x, y, w, h)}]
    """
    # Re-init cascade to ensure thread safety
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        print(f"Error: Could not load Haar Cascade from {face_cascade_path}", file=sys.stderr)
        return [], None # Return empty

    if image_bytes:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_np is not None:
        img = image_np
    else:
        return []

    if img is None:
        print("Debug: Decoded image is None", file=sys.stderr)
        return []

    print(f"Debug: Processing image shape: {img.shape}", file=sys.stderr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Relaxed parameters: scaleFactor 1.1, minNeighbors 3
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    print(f"Debug: Detect faces found {len(faces)}", file=sys.stderr) 

    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        results.append({
            'face': face_roi,
            'box': (int(x), int(y), int(w), int(h))
        })
    
    return results, img
