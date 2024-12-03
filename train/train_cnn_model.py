import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pickle

# Define data paths and hyperparameters
DATA_DIR = "archive/test"
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 10  # Tolerance for early stopping

# Build tag list
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Define data enhancement and preprocessing
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)

# data loader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define CNN model (using pretrained ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))  # Modify the output size of the last layer
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce learning rate every 5 epochs

# Initialize the dictionary that holds training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_accuracy': [],
    'val_accuracy': []
}
# Initialize optimal validation accuracy and early stopping counters
best_accuracy = 0.0
epochs_no_improve = 0

# training and validation
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # forward propagation
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # record loss
        running_loss += loss.item() * images.size(0)

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train * 100
    history['train_loss'].append(train_loss)        # Record training loss
    history['train_accuracy'].append(train_accuracy) # Record training accuracy

    # verify
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct_val / total_val * 100
    history['val_loss'].append(val_loss)          # Record verification loss
    history['val_accuracy'].append(val_accuracy)  # Record verification accuracy

    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Learning rate scheduler update
    scheduler.step()

    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), "ml_model/cnn_model.pth")
        print("Best model saved")
    else:
        epochs_no_improve += 1

    # Early judgment
    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered due to no improvement in validation accuracy")
        break

# Save training history to file
with open("ml_model/training_history.pkl", "wb") as f:
    pickle.dump(history, f)

print(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
