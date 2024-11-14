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

# Initialize timer for emotion detection
last_detection_time = time.time()
detection_interval = 3  # in seconds
current_label = "Detecting..."

# Define button size
button_size = (100, 40)

# Mouse callback function to detect clicks in the exit button area
def click_event(event, x, y, flags, param):
    global running
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is inside the button area
        if button_position[0] <= x <= button_position[0] + button_size[0] and \
           button_position[1] <= y <= button_position[1] + button_size[1]:
            print("Exit button clicked. Exiting...")
            running = False

# Set mouse callback for the window
cv2.namedWindow("Real-Time Image Classification")
cv2.setMouseCallback("Real-Time Image Classification", click_event)

running = True
try:
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Calculate button position dynamically at the top-right corner
        frame_width = frame.shape[1]
        button_position = (frame_width - button_size[0] - 10, 10)  # 10 pixels margin from the edge

        # Every 3 seconds, perform emotion detection
        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            # Convert the frame to PIL Image for processing
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_label = predict(image)
            last_detection_time = current_time  # Reset timer

        # Display the label on the frame
        cv2.putText(frame, f"Predicted: {current_label}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the exit button on the frame at the top-right corner
        cv2.rectangle(frame, button_position, 
                      (button_position[0] + button_size[0], button_position[1] + button_size[1]), 
                      (0, 0, 255), -1)
        cv2.putText(frame, "Exit", (button_position[0] + 10, button_position[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame in real-time
        cv2.imshow("Real-Time Image Classification", frame)

        # Check if the window is closed
        if cv2.getWindowProperty("Real-Time Image Classification", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting...")
            break

        # Wait for key press and check for 'Escape' (27) to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Escape key to close
            print("Exiting...")
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
