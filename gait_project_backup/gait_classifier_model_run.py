import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# === 1. Setup and Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 2. Data Transformation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 3. Load and Split Dataset ===
data_dir = r"C:\Users\hairi\OneDrive\IDP\monai_project\gait_project_backup\dataset"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

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
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.resnet(x))

model = SimpleCNN().to(device)

# === 5. Training Setup ===
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# === 6. Training Loop ===
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def train_model(num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
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

        # Calculate training metrics
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
                inputs, labels = inputs.to(device), labels.to(device)
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.2f}%")

# === 7. Run Training ===
train_model(num_epochs=20)

# === 8.Plot Results ===
plt.style.use('ggplot')
plt.figure(figsize=(18, 6))

# Plot 1: Actual Training Results
plt.subplot(1, 3, 1)
plt.plot(train_accuracies, 'b-o', label='Training Accuracy')
plt.plot(val_accuracies, 'r-s', label='Validation Accuracy')
plt.title('Your Model Performance', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)

# Plot 2: Ideal Scenario (Graph b)
plt.subplot(1, 3, 2)
ideal_train = np.linspace(85, 95, 20)
ideal_val = np.linspace(83, 93, 20)
plt.plot(ideal_train, 'b-o', label='Training Accuracy')
plt.plot(ideal_val, 'r-s', label='Validation Accuracy')
plt.title('(b) Ideal Performance\n(Good Generalization)', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)

# Plot 3: Overfitting Scenario (Graph a)
plt.subplot(1, 3, 3)
overfit_train = [75, 80, 85, 85, 90] + [88 + np.random.rand()*4 for _ in range(15)]
overfit_val = [70, 75, 80, 82, 85] + [82 + np.random.rand()*3 for _ in range(15)]
plt.plot(overfit_train, 'b-o', label='Training Accuracy')
plt.plot(overfit_val, 'r-s', label='Validation Accuracy')
plt.title('(a) Overfitting Scenario', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_results.png', dpi=300)
plt.show()

# === 9. Save Model ===
torch.save(model.state_dict(), "gait_classifier.pth")
print("Model saved as gait_classifier.pth")