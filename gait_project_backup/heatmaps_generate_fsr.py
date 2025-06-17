import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import zoom

# === Folder setup ===
input_folder = "csv_files"  # Folder CSV asal
output_folder = "dataset"
os.makedirs(os.path.join(output_folder, "normal"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "abnormal"), exist_ok=True)

# === Function to create heatmap ===
def create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ):
    grid = np.zeros((10, 7))

    # FSR sensors
    grid[3, 2] = fsr1
    grid[2, 4] = fsr2
    grid[8, 3] = fsr3

    # Accelerometer
    grid[1, 2] = accelX
    grid[4, 3] = accelY
    grid[6, 4] = accelZ

    # Gyroscope
    grid[2, 5] = gyroX
    grid[5, 2] = gyroY
    grid[7, 3] = gyroZ

    # Resize to 28x28
    resized = zoom(grid, (28 / 10, 28 / 7), order=1)

    # Min-max normalization
    min_val = np.min(resized)
    max_val = np.max(resized)
    normalized = (resized - min_val) / (max_val - min_val + 1e-5)

    # Apply better colormap: viridis (smooth + perceptually uniform)
    colored = cm.viridis(normalized)[:, :, :3]  # Drop alpha

    return Image.fromarray((colored * 255).astype(np.uint8))

# === Process CSV files ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)

        required_cols = {"fsr1", "fsr2", "fsr3", "accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"}
        if required_cols.issubset(df.columns):
            # Average all sensor values
            fsr1 = df["fsr1"].mean()
            fsr2 = df["fsr2"].mean()
            fsr3 = df["fsr3"].mean()
            accelX = df["accelX"].mean()
            accelY = df["accelY"].mean()
            accelZ = df["accelZ"].mean()
            gyroX = df["gyroX"].mean()
            gyroY = df["gyroY"].mean()
            gyroZ = df["gyroZ"].mean()

            # Create and save heatmap image
            heatmap = create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ)
            label = "abnormal" if "abnormal" in filename.lower() else "normal"
            safe_name = filename.replace(".csv", "").replace(" ", "_").replace("-", "_")
            save_path = os.path.join(output_folder, label, safe_name + ".png")
            heatmap.save(save_path)
            print(f"Saved: {save_path}")
        else:
            print(f"Skipped {filename}: Missing required columns")

# === Summary ===
print("\nTotal Normal Images:", len(os.listdir(os.path.join(output_folder, "normal"))))
print("Total Abnormal Images:", len(os.listdir(os.path.join(output_folder, "abnormal"))))
