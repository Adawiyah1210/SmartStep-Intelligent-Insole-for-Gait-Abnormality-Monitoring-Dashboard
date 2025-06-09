import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from PIL import Image, ImageFilter
import torch.nn as nn
import matplotlib.cm as cm
from scipy.ndimage import zoom
import streamlit as st
import gdown
import torch

# Gantikan dengan ID fail Google Drive anda
url = 'https://drive.google.com/uc?export=download&id=1EsvSLsyeNEpNjc1tPuKfK9ezx2MnypAA'
output = 'gait_classifier_model.pth'

# Muat turun fail dari Google Drive
gdown.download(url, output, quiet=False)

# Muatkan model
model = torch.load(output)

# Pastikan model berfungsi
print("Model loaded successfully!")

# === 1. App Layout ===
st.set_page_config(layout="wide", page_title="Gait Analysis Dashboard")
st.title("🦶 Gait Analysis Dashboard")

# === 2. Upload User Image ===
uploaded_user_image = st.file_uploader("📤 Upload Your Profile Image", type=["jpg", "jpeg", "png"])
if uploaded_user_image:
    user_image = Image.open(uploaded_user_image)
    st.image(user_image, caption="Uploaded User Image", width=150)

# === 3. User Information Form ===
with st.form("user_info_form"):
    st.subheader("🧍 User Information")
    col1, col2, col3 = st.columns(3)

    name = col1.text_input("Name")
    age = col2.number_input("Age", min_value=1, max_value=120, value=25)
    gender = col3.selectbox("Gender", ["Male", "Female", "Other"])

    weight = col1.number_input("Weight (kg)", min_value=1.0, value=70.0)
    height = col2.number_input("Height (cm)", min_value=30.0, value=170.0)

    foot_type = col3.selectbox("Foot Type", ["Neutral", "Flat", "High Arch"])
    assist_device = col1.selectbox("Assistive Device", ["None", "Cane", "Walker", "Other"])
    activity = col2.text_input("Activity During Data Collection")
    shoe_type = col3.selectbox("Shoe Type", ["Barefoot", "Sneakers", "Formal", "Other"])

    medical_history = col1.text_area("Medical History")
    daily_activity = col2.selectbox("Daily Activity level", ["Low", "Moderate", "High"])

    bmi = round(weight / ((height / 100) ** 2), 2)
    st.markdown(f"🧮 BMI: {bmi}")

    submit_info = st.form_submit_button("Submit Info")

# === 4. Upload Sensor CSV File ===
uploaded_csv = st.file_uploader("📤 Upload Sensor CSV File", type=["csv"])
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.subheader("📄 Uploaded Sensor Data")
    st.dataframe(df.head())

    # === 5. Plot Sensor Graphs ===
    st.subheader("📈 Sensor Graphs")

    acc_cols = ['accelX', 'accelY', 'accelZ']
    gyro_cols = ['gyroX', 'gyroY', 'gyroZ']
    fsr_cols = ['fsr1', 'fsr2', 'fsr3']

    acc_present = all(col in df.columns for col in acc_cols)
    gyro_present = all(col in df.columns for col in gyro_cols)
    fsr_present = any(col in df.columns for col in fsr_cols)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), dpi=100)
    fig.tight_layout(pad=4.0)

    if acc_present:
        df[acc_cols].plot(ax=axes[0])
        axes[0].set_title("Accelerometer (X, Y, Z)")
        axes[0].legend(acc_cols, loc="upper right")
    else:
        axes[0].set_visible(False)
        st.warning("Accelerometer data not found.")

    if gyro_present:
        df[gyro_cols].plot(ax=axes[1])
        axes[1].set_title("Gyroscope (X, Y, Z)")
        axes[1].legend(gyro_cols, loc="upper right")
    else:
        axes[1].set_visible(False)
        st.warning("Gyroscope data not found.")

    if fsr_present:
        fsr_available = [col for col in fsr_cols if col in df.columns]
        df[fsr_available].plot(ax=axes[2])
        axes[2].set_title("FSR Sensors")
        axes[2].legend(fsr_available, loc="upper right")
    else:
        axes[2].set_visible(False)
        st.warning("FSR sensor data not found.")

    st.pyplot(fig)

    # === 6. Generate FSR Heatmap ===
    st.subheader("👣 FSR Heatmap")

    def create_foot_shaped_heatmap(fsr1, fsr2, fsr3):
        grid = np.zeros((10, 7))
        grid[3, 2] = fsr1  # fsr1 – ball of left foot
        grid[2, 4] = fsr2  # fsr2 – ball of right foot
        grid[8, 3] = fsr3  # fsr3 – heel

        resized = zoom(grid, (28 / 10, 28 / 7), order=1)
        normalized = resized / np.max(resized) if np.max(resized) > 0 else resized
        colored = cm.jet(normalized)[:, :, :3]
        image = Image.fromarray((colored * 255).astype(np.uint8))
        image = image.filter(ImageFilter.SHARPEN)

        return image

    if {"fsr1", "fsr2", "fsr3"}.issubset(df.columns):
        fsr1 = df["fsr1"].mean()
        fsr2 = df["fsr2"].mean()
        fsr3 = df["fsr3"].mean()
        heatmap = create_foot_shaped_heatmap(fsr1, fsr2, fsr3)

        st.image(heatmap, caption="Generated Foot-shaped Heatmap", width=150)

        # === 7. Gait Classification ===
        st.subheader("🧠 Gait Classification")

        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

            def forward(self, x):
                return torch.sigmoid(self.resnet(x))

        model_path = "gait_classifier_model.pth"
        if os.path.exists(model_path):
            model = SimpleCNN()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            input_image = transform(heatmap).unsqueeze(0)

            with torch.no_grad():
                output = model(input_image)
                prediction = (output.item() > 0.5)
                label = "🟢 Normal Gait" if prediction else "🔴 Abnormal Gait"
                st.success(f"**Prediction: {label}")
        else:
            st.error("Model file not found: gait_classifier_model.pth")