import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Load data
df = pd.read_csv('gait_data.csv')

# 2. Pisahkan features dan label (kolum label ialah 'Gait')
X = df.drop('Gait', axis=1).values.astype(float)
y = df['Gait'].values

# 3. Encode label (Normal=0, Abnormal=1)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 4. Standardize feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split train & test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.3, random_state=42)

# 6. Buat dataset PyTorch (Label kena .long() untuk CrossEntropyLoss)
train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
test_ds = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=2)

# 7. Define simple NN model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)  # 10 input features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)   # 2 output classes

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# 8. Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 9. Train loop
epochs = 20
for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 10. Test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_dl:
        pred = model(xb)
        _, predicted = torch.max(pred, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

print(f"Test Accuracy: {correct/total*100:.2f}%")