import numpy as np
import cv2
import sys
import os

# Ensure app is in path
sys.path.append(os.getcwd())

from app.model import EmotionModel

def test():
    print("Initializing model...")
    try:
        model = EmotionModel()
    except Exception as e:
        print(f"Failed to init model: {e}")
        return

    print("Creating dummy face image (48x48)...")
    dummy_face = np.zeros((48, 48), dtype=np.uint8)
    
    print("Predicting...")
    emotion, confidence = model.predict(dummy_face)
    print(f"Result 1 (48x48): {emotion}, {confidence}")

    print("Creating dummy face image (100x100)...")
    dummy_face_2 = np.zeros((100, 100), dtype=np.uint8)
    emotion, confidence = model.predict(dummy_face_2)
    print(f"Result 2 (100x100): {emotion}, {confidence}")

if __name__ == "__main__":
    test()
