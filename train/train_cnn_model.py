import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pickle

# 定义数据路径和超参数
DATA_DIR = "archive/train"
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 10  # 早停的容忍度

# 构建标签列表
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# 定义数据增强和预处理
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

# 加载数据集
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义 CNN 模型（使用预训练的 ResNet18）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))  # 修改最后一层的输出大小
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch降低学习率

# 初始化保存训练历史的字典
history = {
    'train_loss': [],
    'val_loss': [],
    'train_accuracy': [],
    'val_accuracy': []
}
# 初始化最佳验证准确率和早停计数器
best_accuracy = 0.0
epochs_no_improve = 0

# 训练和验证
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 记录损失
        running_loss += loss.item() * images.size(0)

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train * 100
    history['train_loss'].append(train_loss)        # 记录训练损失
    history['train_accuracy'].append(train_accuracy) # 记录训练准确率

    # 验证
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
    history['val_loss'].append(val_loss)          # 记录验证损失
    history['val_accuracy'].append(val_accuracy)  # 记录验证准确率

    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # 学习率调度器更新
    scheduler.step()

    # 保存最佳模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), "ml_model/cnn_model_epochs30.pth")
        print("Best model saved")
    else:
        epochs_no_improve += 1

    # 早停判断
    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered due to no improvement in validation accuracy")
        break

# 保存训练历史记录到文件
with open("ml_model/training_history.pkl", "wb") as f:
    pickle.dump(history, f)

print(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
