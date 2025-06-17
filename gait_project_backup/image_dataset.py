from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity, ToTensor
)
from monai.data import ImageDataset, DataLoader
import os
import glob

# 1. Define transforms
train_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize((28, 28)),
    ScaleIntensity(),
    ToTensor()
])

val_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize((28, 28)),
    ScaleIntensity(),
    ToTensor()
])

# 2. Load images from folders
train_normal = sorted(glob.glob(os.path.join("dataset", "normal", "*.png")))[:16]
train_abnormal = sorted(glob.glob(os.path.join("dataset", "abnormal", "*.png")))[:16]
val_normal = sorted(glob.glob(os.path.join("dataset", "normal", "*.png")))[16:]
val_abnormal = sorted(glob.glob(os.path.join("dataset", "abnormal", "*.png")))[16:]

# 3. Combine and label (0=normal, 1=abnormal)
train_images = train_normal + train_abnormal
train_labels = [0] * len(train_normal) + [1] * len(train_abnormal)

val_images = val_normal + val_abnormal
val_labels = [0] * len(val_normal) + [1] * len(val_abnormal)

# 4. Create ImageDataset and DataLoader
train_ds = ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

val_ds = ImageDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# 5. Debug print
for batch_data in train_loader:
    images, labels = batch_data
    print(type(images), images.shape)  # e.g. torch.Size([8, 1, 28, 28])
    print(labels)
    break  # test 1 batch je