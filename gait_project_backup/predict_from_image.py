import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
predict_dir = r"C:\Users\hairi\OneDrive\IDP\monai_project\gait_project_backup\predict_images"
model_path = "best_gait_classifier_model.pth"
threshold = 0.5
show_confusion = True

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Model structure match training (NO Sequential) ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.resnet = models.resnet18(weights=None)  # pretrained=False deprecated
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
           nn.Dropout (0.4),
           nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        x = self.resnet(x)
        return torch.sigmoid(x)

# === Load model ===
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# === Predict ===
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

            # Infer true label from filename
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

# === Save CSV ===
df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)
print("\nSaved predictions to predictions.csv")

# === Confusion matrix (optional) ===
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