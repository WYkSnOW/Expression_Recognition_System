import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# Define label list
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load the pretrained ResNet18 model and modify the output layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model.load_state_dict(torch.load("ml_model/cnn/cnn_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = LABELS[predicted.item()]
    return label

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to PIL Image for processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label = predict(image)

        # Display the label on the frame
        cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Real-Time Image Classification", frame)

        # Wait for key press and check for 'Escape' (27) to exit
        key = cv2.waitKey(3000) & 0xFF
        if key == 27:  # Escape key to close
            print("Exiting...")
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
