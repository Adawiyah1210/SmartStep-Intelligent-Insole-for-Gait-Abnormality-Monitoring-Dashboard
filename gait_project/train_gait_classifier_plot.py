import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# === 1. Transform ===
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 2. Load Dataset ===
data_dir = r"C:\Users\hairi\OneDrive\IDP\monai_project\gait_project\dataset"
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
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))

model = SimpleCNN()

# === 5. Loss and Optimizer ===
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 6. Train Function (With Logging) ===
def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    epoch_loss_values = []
    accuracy_values = []

    for epoch in range(num_epochs):
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
        accuracy = 100 * correct / total
        epoch_loss_values.append(epoch_loss)
        accuracy_values.append(accuracy / 100.0)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Simpan CSV
    with open("epoch_loss_values.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(epoch_loss_values)
    with open("metric_values.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(accuracy_values)

    print("Saved epoch loss and accuracy values to CSV.")

# === 7. Evaluation Function ===
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
    print(f"Test Accuracy: {accuracy:.2f}%")

# === 8. Run Training and Evaluation ===
train_model(model, train_loader, optimizer, criterion, num_epochs=10)
evaluate_model(model, test_loader)

# === 9. Save Model ===
torch.save(model.state_dict(), "gait_classifier_model.pth")
print("Model saved as gait_classifier_model.pth")

# === 10. Plot Loss & Accuracy ===
# Baca CSV
with open("epoch_loss_values.csv") as f:
    epoch_loss_values = list(map(float, next(csv.reader(f))))
with open("metric_values.csv") as f:
    metric_values = list(map(float, next(csv.reader(f))))

# Plot
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
plt.xlabel("epoch")
plt.plot(x, epoch_loss_values)

plt.subplot(1, 2, 2)
plt.title("Accuracy")
x = [i + 1 for i in range(len(metric_values))]
plt.xlabel("epoch")
plt.plot(x, metric_values)

plt.tight_layout()
plt.show()