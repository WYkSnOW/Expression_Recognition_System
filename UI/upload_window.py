import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Label, Button
import pickle

# Define label list
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load the pretrained ResNet18 model and modify the output layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)  # Load structure
model.fc = nn.Linear(model.fc.in_features, len(LABELS))  # Modify the fully connected layer to match label count
model.load_state_dict(torch.load("ml_model/cnn/cnn_model.pth", map_location=device))  # Load weights
model = model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")  # Assume color image
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = LABELS[predicted.item()]
    return label

# Create Tkinter window
def open_window():
    root = tk.Tk()
    root.title("Image Classification")

    def upload_and_predict():
        file_path = filedialog.askopenfilename()
        if file_path:
            label = predict(file_path)
            result_label.config(text="Predicted Label: " + label)

    upload_button = Button(root, text="Upload Image", command=upload_and_predict)
    upload_button.pack()

    result_label = Label(root, text="")
    result_label.pack()

    root.mainloop()

# Run the window
if __name__ == "__main__":
    open_window()
