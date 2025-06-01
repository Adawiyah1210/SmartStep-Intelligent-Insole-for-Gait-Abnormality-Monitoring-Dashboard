import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

# Sidebar - User Input Form
st.sidebar.header("User Information")

user_image = st.sidebar.file_uploader("Upload Your Image (Foot/Face)", type=["png", "jpg", "jpeg"])
if user_image is not None:
    st.sidebar.image(user_image, caption="Uploaded Image", use_column_width=True)

name = st.sidebar.text_input("Name", "John Doe")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
height = st.sidebar.number_input("Height (cm)", min_value=30.0, max_value=250.0, value=170.0)
bmi = weight / ((height / 100) ** 2) if height > 0 else 0
medical_history = st.sidebar.text_area("Medical History", "")
daily_activity = st.sidebar.selectbox("Daily Activity Level", ["Low", "Moderate", "High"])
assistive_device = st.sidebar.selectbox("Assistive Device", ["None", "Cane", "Walker", "Crutches"])
foot_type = st.sidebar.selectbox("Foot Type", ["Neutral", "Flat", "High Arch"])
activity_during_data = st.sidebar.text_input("Activity During Data Collection", "Walking")
shoe_type = st.sidebar.selectbox("Shoe Type", ["Barefoot", "Sneakers", "Formal", "Sandals"])

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("new_gait_window.csv")
    fsr_cols = ['fsr1', 'fsr2', 'fsr3', 'fsr4']
    df[fsr_cols] = df[fsr_cols].clip(0, 1023)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

def pressure_grade(pressure_values):
    avg_pressure = np.mean(pressure_values)
    if avg_pressure < 300:
        return "Good ✅ Normal Pressure"
    elif avg_pressure < 700:
        return "Moderate ⚠️ Moderate Pressure"
    else:
        return "High ⚠️ High Pressure! Attention Needed"

# Load gait classification model
try:
    model = joblib.load("gait_classifier_rf.pkl")
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
except Exception as e:
    st.warning("⚠️ Model or feature columns not loaded. Gait classification not available.")
    model = None

def extract_features(data):
    return {
        'mean_accelX': data['accelX'].mean(),
        'mean_accelY': data['accelY'].mean(),
        'mean_accelZ': data['accelZ'].mean(),
        'mean_gyroX': data['gyroX'].mean(),
        'mean_gyroY': data['gyroY'].mean(),
        'mean_gyroZ': data['gyroZ'].mean(),
    }

st.title("SmartStep Intelligent Insole for Gait Abnormality Monitoring Dashboard")
st.markdown("""
Welcome! This dashboard provides detailed insights from your smart insole device, tracking foot pressure and movement patterns.  
It helps identify and monitor irregularities in your gait such as uneven steps or pressure points, enabling you to better manage and improve your overall foot health and mobility.
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Foot Pressure Heatmap (Right Foot Only)")
    frame_num = st.slider("Select Frame", 1, len(df), 1)
    row = df.iloc[frame_num - 1]
    pressure_grid = np.array([
        [row['fsr2'], row['fsr3']],
        [row['fsr1'], row['fsr4']]
    ])

    min_val = pressure_grid.min()
    max_val = pressure_grid.max()
    if min_val == max_val:
        max_val = min_val + 0.1  # prevent flat color scale

    fig, ax = plt.subplots(figsize=(4,4))
    heatmap = ax.imshow(pressure_grid, cmap='jet', interpolation='nearest', vmin=min_val, vmax=max_val)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Mid-Left', 'Mid-Right'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Heel', 'Toe'])
    ax.set_title(f"Frame {frame_num} / {len(df)}")
    plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    ax.invert_yaxis()
    st.pyplot(fig)

    grade = pressure_grade(pressure_grid.flatten())
    st.markdown(f"**Pressure Status:** {grade}")

    st.markdown("""
    Heatmap Color Legend:
    - 🔴 Red: High pressure — area experiencing the most weight/load  
    - 🟠 Orange: Moderate-high pressure  
    - 🟡 Yellow: Moderate pressure  
    - 🟢 Green: Low pressure — less weight/load on this area  
    - 🔵 Blue: Minimal or no pressure detected
    """)

    st.markdown("## 📈 Sensor Data Readings Over Time")

    fsr_cols = ['fsr1', 'fsr2', 'fsr3', 'fsr4']
    acc_cols = ['accelX', 'accelY', 'accelZ']
    gyro_cols = ['gyroX', 'gyroY', 'gyroZ']

    st.subheader("🔥 FSR Pressure Sensors")
    st.line_chart(df[fsr_cols])
    st.markdown("**Comment:** Pressure values from 4 foot sensors. Higher values indicate more weight bearing on that area.")

    if all(col in df.columns for col in acc_cols):
        st.subheader("🌀 Accelerometer (X, Y, Z)")
        st.line_chart(df[acc_cols])
        st.markdown("**Comment:** Measures foot movement along 3 axes. Peaks represent foot impacts and steps.")

    if all(col in df.columns for col in gyro_cols):
        st.subheader("🔄 Gyroscope (X, Y, Z)")
        st.line_chart(df[gyro_cols])
        st.markdown("**Comment:** Measures foot rotation and orientation. Helps detect gait imbalances or irregularities.")

with col2:
    st.markdown("### 👤 User Information Summary")
    if user_image is not None:
        st.image(user_image, caption="User Uploaded Image", use_column_width=True)
    st.markdown(f"- Name: {name}")
    st.markdown(f"- Age: {age}")
    st.markdown(f"- Gender: {gender}")
    st.markdown(f"- Weight: {weight} kg")
    st.markdown(f"- Height: {height} cm")
    st.markdown(f"- BMI: {bmi:.2f}")
    st.markdown(f"- Daily Activity: {daily_activity}")
    st.markdown(f"- Assistive Device: {assistive_device}")
    st.markdown(f"- Foot Type: {foot_type}")
    st.markdown(f"- Activity During Data Collection: {activity_during_data}")
    st.markdown(f"- Shoe Type: {shoe_type}")
    st.markdown(f"- Medical History: {medical_history}")

if model is not None:
    features = extract_features(df)
    df_input = pd.DataFrame([features])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(df_input)[0]
    st.markdown(f"## 🧠 Gait Classification Result: **{pred.upper()}**")

    if pred.lower() == "normal":
        st.success("👍 Your gait appears NORMAL. Foot pressure and movement are stable.")
        st.info("Keep up an active lifestyle and wear comfortable shoes to maintain foot health.")
    else:
        st.error("⚠️ Your gait is classified as ABNORMAL. There are imbalances or unusual pressure patterns.")
        st.warning("Please consult a physiotherapist or orthopedic specialist for further evaluation and treatment.")
else:
    st.info("Gait classification model is not available; analysis based on sensor data only.")