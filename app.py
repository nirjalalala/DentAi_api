# app.py
"""
This module defines the FastAPI application for the DentAI service.
It provides API endpoints for health checks and for performing
YOLO-based object detection on uploaded images.
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from yolov_model import predict
from pathlib import Path
from ultralytics import YOLO

# Initialize the FastAPI application with a descriptive title for the documentation.
app = FastAPI(title="DentAI YOLOv11 API")

# --- Robust Model Loading ---
# Construct an absolute path to the model file relative to this script's location.
# This ensures the model is found regardless of the current working directory.
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "best.pt"

# Load the model once at startup using the absolute path
model = YOLO(MODEL_PATH)

@app.get("/")
async def root():
    """
    Root endpoint for the API.

    Provides a simple health check message to confirm that the API is running.
    """
    return {"message": "DentAI API is running!"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, performs object detection, and returns the results.

    This endpoint expects a multipart/form-data request with a file part named 'file'.

    Args:
        file (UploadFile): The image file uploaded by the user.

    Raises:
        HTTPException(400): If the uploaded file is not a valid image type
                            (e.g., 'image/jpeg', 'image/png').
        HTTPException(422): If the image file is corrupt or cannot be processed.
        HTTPException(500): For any other unexpected server-side errors.

    Returns:
        JSONResponse: A JSON object containing the detection results, which includes
                      a list of detected objects and a base64 encoded image with
                      bounding boxes drawn on it.
    """
    try:
        # --- 1. Validate Input File ---
        # Ensure the uploaded file has a content type that starts with "image/".
        # This is a basic check to prevent processing of non-image files.
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image with a valid content-type (e.g., image/jpeg).")

        # --- 2. Read Image Data ---
        # Asynchronously read the entire content of the uploaded file into memory.
        image_bytes = await file.read()

        # --- 3. Perform Prediction ---
        # Call the predict function from the model module.
        # This function handles the core logic of decoding the image, running the
        # model, and processing the results.
        try:
            prediction_result = predict(image_bytes)
            # The `predict` function returns a dictionary, which FastAPI automatically
            # converts to a JSON response.
            return prediction_result
        except ValueError as e:
            # This specific exception is raised by our model module for corrupt images.
            raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        # --- 4. Handle Generic Errors ---
        # Catch any other exceptions to prevent the server from crashing.
        # Return a generic 500 Internal Server Error response.
        # In a production environment, you would log the error `e` here.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
