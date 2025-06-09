import torch
import torch.nn as nn  # Pastikan anda mengimport nn dari torch
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms

# Model CNN menggunakan ResNet18
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Gunakan model pre-trained ResNet18
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Gantikan lapisan terakhir untuk output binari (normal vs abnormal)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))  # Output binari menggunakan fungsi sigmoid

# Tentukan model, optimizer, dan fungsi kehilangan
model = SimpleCNN()  # Model yang telah anda definisikan
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

# Pastikan model berada di GPU jika tersedia
if torch.cuda.is_available():
    model = model.cuda()

# Fungsi untuk melatih model
def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()  # Ubah model ke mod latihan
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Pindahkan ke GPU jika tersedia

            optimizer.zero_grad()  # Sifar gradien

            # Maju ke hadapan
            outputs = model(inputs)
            labels = labels.unsqueeze(1).float()  # Pastikan label adalah float

            loss = criterion(outputs, labels)
            loss.backward()  # Kira gradien
            optimizer.step()  # Kemaskini parameter model

            running_loss += loss.item()

            predicted = (outputs > 0.5).float()  # Klasifikasikan dengan threshold 0.5
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Fungsi untuk menilai model
def evaluate_model(model, test_loader):
    model.eval()  # Ubah model ke mod evaluasi
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Tidak perlu mengira gradien semasa penilaian
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            # Maju ke hadapan
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # Threshold 0.5 untuk klasifikasi

            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions * 100
    print(f"Ujian Accuracy: {accuracy:.2f}%")

# Latih model
train_model(model, train_loader, optimizer, criterion, num_epochs=10)

# Uji model
evaluate_model(model, test_loader)