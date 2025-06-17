import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# === Folder setup ===
input_folder = "csv_files"
output_folder = "dataset"
normal_folder = os.path.join(output_folder, "normal")
abnormal_folder = os.path.join(output_folder, "abnormal")
os.makedirs(normal_folder, exist_ok=True)
os.makedirs(abnormal_folder, exist_ok=True)

# === Sensor columns to extract ===
sensor_cols = ['accelX', 'accelY', 'accelZ',
               'gyroX', 'gyroY', 'gyroZ',
               'fsr1', 'fsr2', 'fsr3']

# === Process each CSV ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        label = "normal" if "normal" in filename.lower() else "abnormal"
        save_folder = normal_folder if label == "normal" else abnormal_folder

        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)

        if not all(col in df.columns for col in sensor_cols):
            print(f"Skipping {filename}: missing columns.")
            continue

        # Extract and normalize
        data = df[sensor_cols].values
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data)

        # Flatten and reshape to 28x28
        flat = normalized.flatten()
        if len(flat) < 784:
            flat = np.pad(flat, (0, 784 - len(flat)), mode='constant')
        else:
            flat = flat[:784]
        heatmap_array = flat.reshape(28, 28)

        # Plot heatmap with clear grid
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            heatmap_array,
            cmap='coolwarm',
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            square=True,
            linewidths=0.5,
            linecolor='white'
        )
        plt.axis('off')

        # Save image
        output_name = filename.replace(".csv", ".png")
        output_path = os.path.join(save_folder, output_name)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved: {output_path}")