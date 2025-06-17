
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# === 1. Page Configuration ===
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
    st.markdown(f"🧮 BMI: **{bmi}**")

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

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), dpi=100)
    fig.tight_layout(pad=4.0)

    # Accelerometer
    if all(col in df.columns for col in acc_cols):
        df[acc_cols].plot(ax=axes[0])
        axes[0].set_title("Accelerometer (X, Y, Z)")
        axes[0].legend(acc_cols)
    else:
        axes[0].set_visible(False)
        st.warning("Accelerometer data not found.")

    # Gyroscope
    if all(col in df.columns for col in gyro_cols):
        df[gyro_cols].plot(ax=axes[1])
        axes[1].set_title("Gyroscope (X, Y, Z)")
        axes[1].legend(gyro_cols)
    else:
        axes[1].set_visible(False)
        st.warning("Gyroscope data not found.")

    # FSR
    if any(col in df.columns for col in fsr_cols):
        fsr_avail = [col for col in fsr_cols if col in df.columns]
        df[fsr_avail].plot(ax=axes[2])
        axes[2].set_title("FSR Sensors")
        axes[2].legend(fsr_avail)
    else:
        axes[2].set_visible(False)
        st.warning("FSR sensor data not found.")

    st.pyplot(fig)

    # === 6. Correlation Matrix Heatmap ===
    st.subheader("📊 Correlation Matrix Heatmap")

    numeric_data = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))

    # Determine label based on classification (use the output from model)
    label = "abnormal" if "abnormal" in uploaded_csv.name.lower() else "normal"

    # Adjust colormap based on prediction
    cmap = "RdYlBu" if label == "abnormal" else "Greens"

    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, cbar=True, square=True, ax=ax_corr)
    ax_corr.set_title(f"Correlation Heatmap - {label.capitalize()}")

    st.pyplot(fig_corr)

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
        # Convert heatmap to PIL image and apply transforms
        fig_img_path = "temp_corr_heatmap.png"
        fig_corr.savefig(fig_img_path, bbox_inches='tight')
        heatmap_img = Image.open(fig_img_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        input_tensor = transform(heatmap_img).unsqueeze(0)

        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            score = output.item()
            st.write(f"Raw model output (sigmoid): {score:.4f}")
            prediction = score > 0.10
            result = "🟢 Normal Gait" if prediction else "🔴 Abnormal Gait"
            st.success(f"Prediction: {result}")
    else:
        st.error("Model file not found: gait_classifier_model.pth")