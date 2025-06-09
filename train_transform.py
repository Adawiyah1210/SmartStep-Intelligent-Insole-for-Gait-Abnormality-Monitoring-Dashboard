import os
import glob
import numpy as np
import torch
import torch.nn as nn
from monai.transforms import Compose, LoadImage, Resize, ScaleIntensity, ToTensor
from monai.data import ImageDataset, DataLoader
from monai.networks.nets import DenseNet121
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load dataset
normal_dir = "dataset/normal"
abnormal_dir = "dataset/abnormal"

normal_images = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
abnormal_images = sorted(glob.glob(os.path.join(abnormal_dir, "*.png")))

all_images = normal_images + abnormal_images
all_labels = [0] * len(normal_images) + [1] * len(abnormal_images)

# 2. Split dataset
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# 3. Define transforms (no keys since using ImageDataset)
train_transforms = Compose([
    LoadImage(image_only=True),
    Resize((28, 28)),
    ScaleIntensity(),
    ToTensor()
])
val_transforms = Compose([
    LoadImage(image_only=True),
    Resize((28, 28)),
    ScaleIntensity(),
    ToTensor()
])

# 4. Create datasets and dataloaders
train_ds = ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
val_ds = ImageDataset(image_files=val_images, labels=val_labels, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

# 5. Define model
model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2).to(device)

# 6. Define loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 7. Training loop
max_epochs = 10
for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_data in train_loader:
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_inputs)
            loss = loss_function(val_outputs, val_labels)
            val_loss += loss.item()
            _, val_pred = torch.max(val_outputs, 1)
            val_correct += (val_pred == val_labels).sum().item()
            val_total += val_labels.size(0)
    val_acc = val_correct / val_total
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

# 8. Save model
torch.save(model.state_dict(), "gait_classifier.pth")
print("✅ Model saved as gait_classifier.pth")
