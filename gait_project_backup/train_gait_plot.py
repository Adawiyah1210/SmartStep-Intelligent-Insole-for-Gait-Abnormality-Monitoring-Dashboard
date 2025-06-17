import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# === 1. Image Transform (28x28x3 PNG -> 28x28x3 Tensor with Augmentation) ===
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === 2. Load Dataset ===
data_dir = r"C:\Users\hairi\OneDrive\IDP\monai_project\gait_project_backup\dataset"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print(f"✅ Classes: {dataset.classes}")  # ['abnormal', 'normal']

# Split 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# === 3. Model ===
class GaitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.resnet.fc.in_features, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaitCNN().to(device)

# === 4. Loss, Optimizer, Scheduler ===
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# === 5. Training Function ===
train_losses, val_losses, val_accuracies = [], [], []

def train(model, train_loader, val_loader, epochs=10):
    best_val_loss = float('inf')
    trigger_times = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += ((outputs > 0.5) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        train_losses.append(avg_loss)

        # ==== Validation ====
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += ((outputs > 0.5) == labels).sum().item()
                val_total += labels.size(0)

        val_avg_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_acc)
        scheduler.step(val_avg_loss)

        print(f"📊 Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # === Early Stopping Check ===
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), "best_gait_classifier_model.pth")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

# === 6. Evaluation Function ===
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).float().to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"\n✅ Final Validation Accuracy: {accuracy:.2f}%")
    return accuracy

# === 7. Run Training ===
train(model, train_loader, val_loader)

# === 8. Load Best Model ===
model.load_state_dict(torch.load("best_gait_classifier_model.pth"))
print("✅ Loaded best model for final evaluation.")

# === 9. Final Evaluation ===
final_accuracy = evaluate_model(model, val_loader)

# === 10. Plot Loss & Accuracy ===
def exponential_smoothing(data, alpha=0.9):
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * data[i])
    return smoothed

def moving_average(data, window=5):
    return [np.mean(data[max(0, i - window + 1):i + 1]) for i in range(len(data))]

val_loss_smooth = moving_average(exponential_smoothing(val_losses))
val_acc_smooth = moving_average(exponential_smoothing(val_accuracies))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(val_loss_smooth, 'r-', linewidth=2, label='Val Loss (Smoothed)')
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_acc_smooth, 'b-', linewidth=2, label='Val Accuracy (Smoothed)')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.legend()

plt.figtext(0.5, 0.02, f"Final Accuracy: {final_accuracy:.2f}% | Model: best_gait_classifier_model.pth",
            ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.3})
plt.tight_layout(pad=3.0)
plt.savefig("training_metrics_validation_only.png", dpi=300)
plt.show()