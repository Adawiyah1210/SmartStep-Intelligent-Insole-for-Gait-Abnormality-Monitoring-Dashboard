import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# === 1. Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 2. Load Dataset ===
data_dir = r"C:\Users\hairi\OneDrive\IDP\monai_project\gait_project_backup\dataset"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# === 3. Split Dataset ===
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === 4. Model Definition ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.resnet(x))

model = SimpleCNN()

# === 5. Loss and Optimizer ===
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# === 6. Scheduler ===
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# === 7. Train Function ===
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                labels = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_epoch_loss = val_loss / len(test_loader)
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        scheduler.step(val_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

# === 8. Evaluation Function ===
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            labels = labels.unsqueeze(1).float()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    return accuracy

# === 9. Run Training ===
train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=20)
final_accuracy = evaluate_model(model, test_loader)

# === 10. Save Model ===
torch.save(model.state_dict(), "gait_classifier_model.pth")
print("Model saved as gait_classifier_model.pth")

# === 11. Smoothing Functions ===
def exponential_smoothing(data, alpha=0.9):
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * data[i])
    return smoothed

def moving_average(data, window_size=5):
    return [np.mean(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))]

# === 12. Plotting Only Validation (Smooth) ===
plt.style.use('ggplot')
plt.figure(figsize=(14, 6))

# Smooth Validation Loss
val_loss_smooth = moving_average(exponential_smoothing(val_losses, 0.92), 5)

plt.subplot(1, 2, 1)
plt.plot(val_loss_smooth, 'r-', linewidth=2.5, label='Validation Loss')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Validation Loss', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)

# Smooth Validation Accuracy
val_acc_smooth = moving_average(exponential_smoothing(val_accuracies, 0.9), 4)

plt.subplot(1, 2, 2)
plt.plot(val_acc_smooth, 'm-', linewidth=2.5, label='Validation Accuracy')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Validation Accuracy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)

plt.figtext(0.5, 0.02, 
            f"Final Test Accuracy: {final_accuracy:.2f}% | Model saved as gait_classifier_model.pth", 
            ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout(pad=3.0)
plt.savefig('training_metrics_validation_only.png', dpi=300)
plt.show()