import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import gdown

# === 1. Streamlit Page Config ===
st.set_page_config(page_title="Gait Analysis Dashboard", layout="wide")
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
    st.markdown(f"🧮 **BMI: {bmi}**")

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

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), dpi=100)
    fig.tight_layout(pad=4.0)

    if all(col in df.columns for col in acc_cols):
        df[acc_cols].plot(ax=axes[0])
        axes[0].set_title("Accelerometer (X, Y, Z)")
        axes[0].legend(acc_cols)
    else:
        axes[0].set_visible(False)
        st.warning("Accelerometer data not found.")

    if all(col in df.columns for col in gyro_cols):
        df[gyro_cols].plot(ax=axes[1])
        axes[1].set_title("Gyroscope (X, Y, Z)")
        axes[1].legend(gyro_cols)
    else:
        axes[1].set_visible(False)
        st.warning("Gyroscope data not found.")

    if any(col in df.columns for col in fsr_cols):
        available_fsr = [col for col in fsr_cols if col in df.columns]
        df[available_fsr].plot(ax=axes[2])
        axes[2].set_title("FSR Sensors")
        axes[2].legend(available_fsr)
    else:
        axes[2].set_visible(False)
        st.warning("FSR sensor data not found.")

    st.pyplot(fig)

    # === 6. Gait Classification (CNN) ===
    st.subheader("🧠 Gait Classification (CNN-based)")

    def create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ):
        grid = np.zeros((10, 7))
        grid[3, 2] = fsr1
        grid[2, 4] = fsr2
        grid[8, 3] = fsr3
        grid[1, 2] = accelX
        grid[4, 3] = accelY
        grid[6, 4] = accelZ
        grid[2, 5] = gyroX
        grid[5, 2] = gyroY
        grid[7, 3] = gyroZ

        resized = zoom(grid, (28 / 10, 28 / 7), order=1)
        normalized = resized / np.max(resized) if np.max(resized) > 0 else resized
        colored = cm.jet(normalized)[:, :, :3]
        return Image.fromarray((colored * 255).astype(np.uint8))

    required_cols = {"fsr1", "fsr2", "fsr3", "accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"}

    if required_cols.issubset(df.columns):
        fsr1 = df["fsr1"].mean()
        fsr2 = df["fsr2"].mean()
        fsr3 = df["fsr3"].mean()
        accelX = df["accelX"].mean()
        accelY = df["accelY"].mean()
        accelZ = df["accelZ"].mean()
        gyroX = df["gyroX"].mean()
        gyroY = df["gyroY"].mean()
        gyroZ = df["gyroZ"].mean()

        heatmap_img = create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ)
        st.image(heatmap_img, caption="Generated Heatmap", width=200)

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(heatmap_img).unsqueeze(0)

        class GaitCNN(nn.Module):
            def __init__(self):
                super(GaitCNN, self).__init__()
                self.resnet = models.resnet18(weights=None)
                num_ftrs = self.resnet.fc.in_features
                self.resnet.fc = nn.Linear(num_ftrs, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.resnet(x)
                return self.sigmoid(x)

        model = GaitCNN()
        model_path = "gait_classifier_model.pth"

        if not os.path.exists(model_path):
            gdown.download("https://drive.google.com/uc?id=1YAAhzJl8fFnkspx4KnzbQAgycpu8VRM8", model_path, quiet=False)

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        if any(k.startswith("network.") for k in state_dict.keys()):
            state_dict = {k.replace("network.", "network."): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            score = output.item()
            st.write(f"Raw model output (sigmoid): {score:.4f}")
            prediction = score > 0.5
            result = "🟢 Normal Gait" if prediction else "🔴 Abnormal Gait"
            st.success(f"Prediction: {result}")
    else:
        st.error("CSV missing required columns for FSR + IMU heatmap generation.")
