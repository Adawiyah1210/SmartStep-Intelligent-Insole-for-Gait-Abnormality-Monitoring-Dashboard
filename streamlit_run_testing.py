import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from scipy.ndimage import zoom
import gdown

# === Streamlit Config ===
st.set_page_config(page_title="🧶 Gait Analysis Dashboard", layout="wide")
st.title("🧶 Gait Classification from IMU + FSR Sensor CSV")

# === GaitCNN Definition (same as training) ===
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

# === Model Loader ===
@st.cache_resource
def load_model(path="best_gait_classifier_model.pth"):
    model_id = "1YAAhzJl8fFnkspx4KnzbQAgycpu8VRM8"  # replace with your actual file ID
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={model_id}", path, quiet=False)
    model = GaitCNN()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# === Heatmap Generator (28x28x3 from IMU + FSR) ===
def generate_heatmap_from_csv(csv_file):
    required_cols = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ', 'fsr1', 'fsr2', 'fsr3']
    df = pd.read_csv(csv_file)
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Missing sensor columns in uploaded CSV")
        return None, None

    # Mean values
    vals = {
        'accelX': df['accelX'].mean(),
        'accelY': df['accelY'].mean(),
        'accelZ': df['accelZ'].mean(),
        'gyroX': df['gyroX'].mean(),
        'gyroY': df['gyroY'].mean(),
        'gyroZ': df['gyroZ'].mean(),
        'fsr1': df['fsr1'].mean() * 5,
        'fsr2': df['fsr2'].mean() * 5,
        'fsr3': df['fsr3'].mean() * 5,
    }

    # Grid 10x7
    grid = np.zeros((10, 7))
    grid[2, 1] = vals['accelX']
    grid[4, 1] = vals['accelY']
    grid[6, 1] = vals['accelZ']

    grid[2, 3] = vals['gyroX']
    grid[4, 3] = vals['gyroY']
    grid[6, 3] = vals['gyroZ']

    grid[2, 5] = vals['fsr1']
    grid[4, 5] = vals['fsr2']
    grid[6, 5] = vals['fsr3']

    # Resize to 28x28
    resized = zoom(grid, (28/10, 28/7), order=1)
    resized = np.clip(resized, -2.0, 2.0)
    normalized = (resized + 2.0) / 4.0

    colored = cm.jet(normalized)[:, :, :3]
    img_array = (colored * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save("generated_heatmap.png")
    return "generated_heatmap.png", df

# === Prediction Function ===
def predict_image(img_path, model, threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        score = output.item()
        prediction = "Normal" if score >= threshold else "Abnormal"
    return score, prediction

# === User Form ===
st.markdown("### Step 1: Upload Image, Info, and CSV")
col1, col2, col3 = st.columns(3)

with col1:
    user_image = st.file_uploader("Upload User Image", type=['jpg', 'png'])
    if user_image:
        st.image(user_image, caption="User Image", width=200)
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0)
    assist = st.selectbox("Assistive Device", ["None", "Cane", "Walker", "Other"])

with col2:
    bmi = round(weight / ((height / 100) ** 2), 2) if height > 0 else 0
    st.markdown(f"**BMI: {bmi}**")
    activity = st.text_input("Activity During Collection")
    medical = st.text_area("Medical History")
    daily = st.selectbox("Daily Activity Level", ["Low", "Moderate", "High"])

with col3:
    foot_type = st.selectbox("Foot Type", ["Neutral", "Flat", "High Arch"])
    shoe_type = st.selectbox("Shoe Type", ["Barefoot", "Sneakers", "Formal", "Other"])

csv_file = st.file_uploader("Upload Sensor CSV File", type=["csv"])

# === Step 2: Heatmap + Graphs ===
if csv_file:
    st.markdown("---")
    st.markdown("### Step 2: View Heatmap + Sensor Graphs")
    heatmap_path, df = generate_heatmap_from_csv(csv_file)

    if heatmap_path:
        st.image(heatmap_path, caption="Generated Heatmap (Accel → Gyro → FSR)")

        st.markdown("#### FSR Sensor Readings")
        st.line_chart(df[['fsr1', 'fsr2', 'fsr3']])

        st.markdown("#### Accelerometer")
        st.line_chart(df[['accelX', 'accelY', 'accelZ']])

        st.markdown("#### Gyroscope")
        st.line_chart(df[['gyroX', 'gyroY', 'gyroZ']])

        # === Step 3: Prediction ===
        st.markdown("---")
        st.markdown("### Step 3: Gait Classification Result")

        model = load_model("best_gait_classifier_model.pth")
        score, prediction = predict_image(heatmap_path, model)

        st.markdown(f"Prediction: **{prediction}**")
        st.markdown(f"Model Score: `{score:.4f}`")

        if prediction == "Abnormal":
            st.error("⚠️ Abnormal Gait Detected")
        else:
            st.success("✅ Normal Gait Detected")
