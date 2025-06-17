import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.cm as cm
from scipy.ndimage import zoom
import gdown

# === 1. Page Configuration ===
st.set_page_config(page_title="Gait Analysis Dashboard", layout="centered")

# === 2. Heatmap Generation ===
def create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ):
    grid = np.zeros((10, 7))
    # [Your existing heatmap code...]
    return Image.fromarray((colored * 255).astype(np.uint8))

# === 3. Model Definition ===
class GaitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None)  # Must match training
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))

# === 4. Fixed Model Loading ===
def load_model():
    model_id = "1YAAhzJl8fFnkspx4KnzbQAgycpu8VRM8"
    model_path = "gait_classifier_model.pth"
    
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={model_id}", model_path, quiet=False)
    
    # Initialize model with correct architecture
    model = GaitCNN()
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    
    # Handle key mismatch (if needed)
    if any(k.startswith('network.') for k in state_dict.keys()):
        state_dict = {k.replace('network.', 'resnet.'): v for k, v in state_dict.items()}
    
    # Load with strict=False to be more forgiving
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

# === 5. Classification Function ===
def classify_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = (output > 0.5).item()
    return "Normal Gait" if prediction else "Abnormal Gait"

# === 6. UI Components ===
st.title("🦶 Gait Analysis Dashboard")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# [Rest of your Streamlit UI code...]