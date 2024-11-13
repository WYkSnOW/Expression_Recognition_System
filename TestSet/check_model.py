import os
import cv2
import joblib  # For loading the model
import numpy as np
from image_processing_pipeline import process_image  # For image processing and face keypoint extraction

# Load random forest model
def load_rf_model(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

# Predict expression using the model
def predict_expression(model, faces):
    if not faces:
        print("No face keypoints detected")
        return None

    # Flatten face keypoints into a 1D array, matching the input format expected by the model
    flat_keypoints = np.array([coord for face in faces for point in face for coord in point]).reshape(1, -1)

    # Predict expression using the model
    prediction = model.predict(flat_keypoints)
    return prediction

# Main function to process the image and recognize expression
def recognize_expression(image_path, model_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to load image: {image_path}")
        return

    # Call the image processing pipeline to extract face keypoints
    _, faces = process_image(image_path)

    # Load random forest model
    model = load_rf_model(model_path)
    if model is None:
        return

    # Predict expression
    expression = predict_expression(model, faces)
    if expression is not None:
        print(f"Predicted expression: {expression[0]}")  # Print prediction result
    else:
        print("Unable to predict expression")

if __name__ == "__main__":
    # Image path and model path
    image_path = "archive/train/fear/fear1/Training_12567.jpg"  # Use absolute path
    model_path = "ml_model/rf_model.pkl"  # Path to trained random forest model

    # Call recognition function
    recognize_expression(image_path, model_path)
