import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
DATA_DIR = 'dataset/data'
BATCH_SIZE = 128
EPOCHS = 350
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on {DEVICE}")

# --- Data Augmentation ---
train_tf = transforms.Compose([
    transforms.Resize(112),
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

test_tf = transforms.Compose([
    transforms.Resize(112),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Dataset Loading ---
train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tf)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_tf)

# Automatic Class Weight Calculation
class_names = train_dataset.classes
class_counts = [len([item for item in train_dataset.targets if item == i]) for i in range(len(class_names))]
print(f"Classes: {class_names} | Counts: {class_counts}")

weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[train_dataset.targets].tolist()
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Definitions ---

def get_resnet_model(num_classes=3):
    """
    Final optimized model based on ResNet18.
    Achieves 75-80% accuracy.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify for Grayscale (1 channel)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    setattr(model, 'maxpool', nn.Identity())

    # Initial Freeze
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

class EmoModel(nn.Module):
    """
    Legacy Custom CNN Architecture.
    Achieves ~55% accuracy.
    """
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.cat([self.pool_avg(x), self.pool_max(x)], dim=1)
        return self.classifier(self.flatten(x))

# --- Utilities ---

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# --- Training Loop ---

model = get_resnet_model(len(class_names)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

is_frozen = True
best_val_loss = float('inf')
epochs_no_improve = 0
history = {'acc': [], 'prec': []}

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    y_true, y_pred = [], []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    
    train_acc = accuracy_score(y_true, y_pred) * 100
    print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%')

    # --- Validation ---
    model.eval()
    val_loss = 0
    v_true, v_pred = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            v_true.extend(labels.cpu().numpy())
            v_pred.extend(predicted.cpu().numpy())

    avg_v_loss = val_loss / len(test_loader)
    v_acc = accuracy_score(v_true, v_pred) * 100
    v_prec = precision_score(v_true, v_pred, average='weighted') * 100
    
    history['acc'].append(v_acc)
    history['prec'].append(v_prec)
    scheduler.step(avg_v_loss)

    print(f'Val Loss: {avg_v_loss:.4f} | Val Acc: {v_acc:.2f}%')

    # Model Checkpointing
    if avg_v_loss < best_val_loss:
        best_val_loss = avg_v_loss
        torch.save(model.state_dict(), 'best_emotion_model.pth')
        print("Model saved!")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Unfreezing Strategy (Phase 2)
    if epochs_no_improve >= 6 and is_frozen:
        print("Unfreezing weights for fine-tuning...")
        is_frozen = False
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        epochs_no_improve = 0
    elif epochs_no_improve >= 6 and not is_frozen:
        print("Early stopping triggered.")
        break

# Final Plot
plt.figure(figsize=(10, 5))
plt.plot(history['acc'], label='Accuracy')
plt.plot(history['prec'], label='Precision')
plt.title('Training Metrics')
plt.legend()
plt.show()