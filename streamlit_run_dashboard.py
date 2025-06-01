import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

# ==============================
# Streamlit Config & Title
# ==============================
st.set_page_config(layout="wide")
st.title("🦶 Gait Analysis Dashboard")

# ==============================
# Sidebar - User Input Form
# ==============================
st.sidebar.header("User Information")

# Upload gambar
user_image = st.sidebar.file_uploader("Upload Your Image (Foot/Face)", type=["png", "jpg", "jpeg"])
if user_image is not None:
    st.sidebar.image(user_image, caption="Uploaded Image", use_column_width=True)

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0)
height = st.sidebar.number_input("Height (cm)", min_value=30.0)
bmi = weight / ((height / 100) ** 2) if height > 0 else 0

medical_history = st.sidebar.text_area("Medical History")
daily_activity = st.sidebar.selectbox("Daily Activity Level", ["Low", "Moderate", "High"])
assistive_device = st.sidebar.selectbox("Assistive Device", ["None", "Cane", "Walker", "Crutches"])
foot_type = st.sidebar.selectbox("Foot Type", ["Neutral", "Flat", "High Arch"])
activity_during_data = st.sidebar.text_input("Activity During Data Collection")
shoe_type = st.sidebar.selectbox("Shoe Type", ["Barefoot", "Sneakers", "Formal", "Sandals"])

# ==============================
# Main - User Info Summary
# ==============================
st.markdown(f"### 👤 User: {name}")
st.markdown(f"- Age: {age} | Gender: {gender}")
st.markdown(f"- Weight: {weight} kg | Height: {height} cm | **BMI: `{bmi:.2f}`**")
st.markdown(f"- Daily Activity: {daily_activity} | Assistive Device: {assistive_device}")
st.markdown(f"- Foot Type: {foot_type} | Activity: {activity_during_data} | Shoe: {shoe_type}")
st.markdown(f"- Medical History: {medical_history}")

# ==============================
# Load CSV & FSR Heatmap
# ==============================
try:
    df = pd.read_csv("new_gait_window.csv")
    fsr_cols = ['fsr1', 'fsr2', 'fsr3', 'fsr4']
    df[fsr_cols] = df[fsr_cols].clip(0, 1023)

    st.markdown("## 🔥 Foot Pressure Heatmap (2x2)")
    stframe = st.empty()

    for i in range(len(df)):
        row = df.iloc[i]
        pressure_grid = np.array([
            [row['fsr2'], row['fsr3']],
            [row['fsr1'], row['fsr4']]
        ])

        fig, ax = plt.subplots()
        heatmap = ax.imshow(pressure_grid, cmap='jet', interpolation='nearest', vmin=0, vmax=1023)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Mid-Left', 'Mid-Right'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Heel', 'Toe'])
        ax.set_title(f"Frame {i+1}/{len(df)}")
        plt.colorbar(heatmap, ax=ax)
        ax.invert_yaxis()
        stframe.pyplot(fig)

except Exception as e:
    st.error(f"❌ Error loading FSR data: {e}")

# ==============================
# 📈 Graphs - Sensor Time Series
# ==============================
if 'df' in locals():
    st.markdown("## 📊 FSR Sensor Readings Over Time")
    st.line_chart(df[fsr_cols])

    if all(col in df.columns for col in ['accelX', 'accelY', 'accelZ']):
        st.markdown("## 🌀 Accelerometer (X/Y/Z)")
        st.line_chart(df[['accelX', 'accelY', 'accelZ']])

    if all(col in df.columns for col in ['gyroX', 'gyroY', 'gyroZ']):
        st.markdown("## 🔄 Gyroscope (X/Y/Z)")
        st.line_chart(df[['gyroX', 'gyroY', 'gyroZ']])

# ==============================
# 🧠 Gait Classification
# ==============================
st.markdown("## 🧠 Gait Classification")

try:
    model = joblib.load("gait_classifier_rf.pkl")
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    # Feature extraction function (adjust based on training)
    def extract_features(data):
        return {
            'mean_accelX': data['accelX'].mean(),
            'mean_accelY': data['accelY'].mean(),
            'mean_accelZ': data['accelZ'].mean(),
'mean_gyroX': data['gyroX'].mean(),
            'mean_gyroY': data['gyroY'].mean(),
            'mean_gyroZ': data['gyroZ'].mean(),
        }

    features = extract_features(df)
    df_input = pd.DataFrame([features])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(df_input)[0]
    st.success(f"🚶 Gait Status: **{pred.upper()}**")

except Exception as e:
    st.error(f"❌ Classification Error: {e}")