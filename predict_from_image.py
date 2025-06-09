import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Konfigurasi ===
predict_dir = r"C:\Users\hairi\OneDrive\IDP\monai_project\gait_project\predict_images"
model_path = "gait_classifier_model.pth"
threshold = 0.4  # Tukar jika perlu
show_confusion = True  # Tukar kepada False jika tak nak paparkan

# === 2. Transformasi sama seperti training ===
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 3. Load Model ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))

model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# === 4. Predict semua imej ===
results = []
true_labels = []
predicted_labels = []

print("Predicting images...\n")

with torch.no_grad():
    for filename in os.listdir(predict_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(predict_dir, filename)
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            output = model(input_tensor)
            score = output.item()
            prediction = "Normal" if score >= threshold else "Abnormal"

            # Cubaan baca label sebenar daripada nama fail (jika ada)
            if "normal" in filename.lower():
                true_label = "Normal"
            elif "abnormal" in filename.lower():
                true_label = "Abnormal"
            else:
                true_label = "Unknown"

            results.append({
                "filename": filename,
                "score": round(score, 4),
                "prediction": prediction,
                "true_label": true_label
            })

            print(f"{filename}: Score = {score:.4f} → Prediction: {prediction}")
            if true_label != "Unknown":
                true_labels.append(true_label)
                predicted_labels.append(prediction)

# === 5. Simpan ke predictions.csv ===
df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)
print("\nSaved predictions to predictions.csv")

# === 6. Confusion Matrix (jika perlu) ===
if show_confusion and true_labels:
    cm = confusion_matrix(true_labels, predicted_labels, labels=["Normal", "Abnormal"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Abnormal"],
                yticklabels=["Normal", "Abnormal"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))