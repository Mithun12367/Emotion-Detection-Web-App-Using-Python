import cv2
import os
from app.utils import detect_faces

# Path to the user's uploaded image
image_path = r"C:/Users/rdxqy/.gemini/antigravity/brain/495cac78-b005-4317-bb19-ddaaf9e29753/uploaded_image_1766766140759.png"

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
else:
    print(f"Testing detection on {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image via cv2.")
    else:
        results, _ = detect_faces(image_np=image)
        print(f"Results (numpy): Found {len(results)} faces.")
        
        # Test bytes path
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        print(f"Read {len(img_bytes)} bytes.")
        results_bytes, _ = detect_faces(image_bytes=img_bytes)
        print(f"Results (bytes): Found {len(results_bytes)} faces.")

