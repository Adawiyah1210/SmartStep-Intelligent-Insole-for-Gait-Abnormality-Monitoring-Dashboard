import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Menyediakan transformasi untuk imej
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Ubah saiz imej ke 28x28
    transforms.ToTensor(),  # Tukar imej ke tensor
    transforms.Normalize(mean=[0.485], std=[0.229])  # Penormalan imej (untuk imej greyscale)
])

# Memuatkan dataset dari folder "normal" dan "abnormal"
dataset = ImageFolder(root=r'C:\Users\hairi\OneDrive\IDP\monai_project\gait_project\dataset', transform=transform)

# Membahagikan data kepada set latihan dan ujian (80% latihan, 20% ujian)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Memuatkan data untuk latihan dan ujian
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Latihan dataset: {len(train_loader.dataset)} imej")
print(f"Ujian dataset: {len(test_loader.dataset)} imej")