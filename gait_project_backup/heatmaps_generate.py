import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import matplotlib.pyplot as plt  # Use this for stable colormap
import matplotlib.cm as cm

# === Folder setup ===
input_folder = "csv_files"
output_folder = "dataset"
normal_folder = os.path.join(output_folder, "normal")
abnormal_folder = os.path.join(output_folder, "abnormal")
os.makedirs(normal_folder, exist_ok=True)
os.makedirs(abnormal_folder, exist_ok=True)

# === Process all CSV files ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # Check label by filename
        label = "normal" if "normal" in filename.lower() else "abnormal"
        save_folder = normal_folder if label == "normal" else abnormal_folder

        file_path = os.path.join(input_folder, filename)
        data = pd.read_csv(file_path)

        # Sensor columns to use
        sensor_cols = ['accelX', 'accelY', 'accelZ',
                       'gyroX', 'gyroY', 'gyroZ',
                       'fsr1', 'fsr2', 'fsr3']
        if not all(col in data.columns for col in sensor_cols):
            print(f"Skipping {filename}: missing required columns.")
            continue

        # Normalize data
        sensor_data = data[sensor_cols]
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(sensor_data)

        # Convert normalized data to grayscale image (0-255)
        gray_img = Image.fromarray((normalized * 255).astype(np.uint8))

        # Resize to 28x28
        gray_img = gray_img.resize((224, 224), Image.Resampling.NEAREST)

        # Apply colormap to create RGB heatmap
        cmap = plt.get_cmap('coolwarm')  # Compatible with all versions
        heatmap_rgb = cmap(np.array(gray_img) / 255.0)[..., :3]  # Drop alpha
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_rgb)

        # Save the image
        output_path = os.path.join(save_folder, filename.replace('.csv', '.png'))
        heatmap_img.save(output_path)
        print(f"Saved: {output_path}")