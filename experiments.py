import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
import os
import json
import time


DATA_DIR = 'data'
RESULTS_DIR = 'tuning_results'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Start pipeline:))))))) {DEVICE}")



class EmoModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.GELU()
        )
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 3)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.cat([self.pool_avg(x), self.pool_max(x)], dim=1)
        return self.classifier(self.flatten(x))

def get_resnet_model(arch='resnet18', num_classes=3):
    
    if arch == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    setattr(model, 'maxpool', nn.Identity())

    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model



def save_training_graphs(history, config_name):
    
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(14, 5))

    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='Train Acc', color='blue')
    plt.plot(epochs, history['val_acc'], label='Val Acc', color='orange')
    plt.title(f'{config_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

   
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='orange')
    plt.title(f'{config_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{config_name}_learning_curves.png"))
    plt.close()

def save_confusion_matrix(y_true, y_pred, classes, config_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{config_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{config_name}_confusion_matrix.png"))
    plt.close()



def train_config(config):
    print(f"\n{'='*50}\nTest accuracy: {config['name']}\n{'='*50}")
    
    
    train_tf = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tf)
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_tf)
    class_names = train_dataset.classes

 
    class_counts = [len([item for item in train_dataset.targets if item == i]) for i in range(len(class_names))]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[train_dataset.targets].tolist()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    if config['model_type'] == 'custom':
        model = EmoModel().to(DEVICE)
        is_frozen = False 
    else:
        model = get_resnet_model(config['model_type'], len(class_names)).to(DEVICE)
        is_frozen = True 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()
        train_loss, y_true, y_pred = 0, [], []
        
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
            
        t_acc = accuracy_score(y_true, y_pred) * 100
        t_loss = train_loss/len(train_loader)

        # Валидация
        model.eval()
        val_loss, v_true, v_pred = 0, [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                v_true.extend(labels.cpu().numpy())
                v_pred.extend(predicted.cpu().numpy())

        v_loss = val_loss/len(test_loader)
        v_acc = accuracy_score(v_true, v_pred) * 100
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        scheduler.step(v_loss)
        print(f"Ep {epoch+1:02d} | T-Loss: {t_loss:.3f} T-Acc: {t_acc:.1f}% | V-Loss: {v_loss:.3f} V-Acc: {v_acc:.1f}%")

       
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{config['name']}_best.pth"))
            best_v_true, best_v_pred = v_true, v_pred 
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        
        if epochs_no_improve >= 5 and is_frozen:
            print("Fine-Tuning...")
            is_frozen = False
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
            epochs_no_improve = 0
        elif epochs_no_improve >= 8 and not is_frozen:
            print("Early Stopping!")
            break

    total_time = (time.time() - start_time) / 60
    print(f"Config: {config['name']} completed in {total_time:.1f} minutes. Best Accuracy: {best_val_acc:.2f}%")
    
    
    save_training_graphs(history, config['name'])
    save_confusion_matrix(best_v_true, best_v_pred, class_names, config['name'])

    return best_val_acc

## configs
configurations = [
    {
        "name": "01_Baseline_CustomCNN",
        "model_type": "custom",
        "img_size": 112,
        "batch_size": 128,
        "lr": 1e-3,
        "optimizer": "adamw",
        "epochs": 50
    },
    {
        "name": "02_ResNet18_Original",
        "model_type": "resnet18",
        "img_size": 112,
        "batch_size": 128,
        "lr": 1e-3,
        "optimizer": "adamw",
        "epochs": 50
    },
    {
        "name": "03_ResNet18_HighRes", 
        "model_type": "resnet18",
        "img_size": 224, 
        "batch_size": 64, 
        "lr": 1e-3,
        "optimizer": "adamw",
        "epochs": 50
    },
    {
        "name": "04_ResNet34_Deep", 
        "model_type": "resnet34",
        "img_size": 112,
        "batch_size": 128,
        "lr": 1e-3,
        "optimizer": "adamw",
        "epochs": 50
    },
    {
        "name": "05_ResNet18_SGD_Momentum", 
        "model_type": "resnet18",
        "img_size": 112,
        "batch_size": 128,
        "lr": 1e-2, 
        "optimizer": "sgd",
        "epochs": 50
    }
]

#pipeline
if __name__ == '__main__':
    results_summary = {}
    
    for config in configurations:
        best_acc = train_config(config)
        results_summary[config['name']] = best_acc

    print("\n" + "="*50)
    print(" Results of pipeline")
    print("="*50)
    
    best_model = None
    max_acc = 0
    
    for name, acc in results_summary.items():
        print(f"Model: {name.ljust(25)} | Accuracy: {acc:.2f}%")
        if acc > max_acc:
            max_acc = acc
            best_model = name
            
    print("-" * 50)
    print(f"Winner: {best_model} with accuracy {max_acc:.2f}%")
    print(f"saving... '{RESULTS_DIR}'.")
    

    with open(os.path.join(RESULTS_DIR, 'final_report.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)