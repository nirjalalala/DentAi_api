# yolov_model.py

from ultralytics import YOLO
import cv2
import base64
import numpy as np
from pathlib import Path

# --- Robust Model Loading ---
# Construct an absolute path to the model file relative to this script's location.
# This ensures the model is found regardless of the current working directory.
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "best.pt"

# Load the model once at startup using the absolute path
model = YOLO(MODEL_PATH)

def predict(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Handle cases where the image cannot be decoded
    if img is None:
        raise ValueError("Could not decode image. The file may be corrupt or in an unsupported format.")

    # Run YOLO prediction
    results = model.predict(img, save=False, stream=False, verbose=False)

    # Process the results
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls_id,
                "confidence": confidence,
                "box": xyxy
            })

    annotated = results[0].plot()
    
     # Encode image as base64 to send via JSON
    _, buffer = cv2.imencode('.jpg', annotated)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return {"detections": detections, "image": encoded_img}
