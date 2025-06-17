import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import zoom

# === Folder setup ===
input_folder = "csv_files"  # Folder with input CSVs
output_folder = "dataset"
os.makedirs(os.path.join(output_folder, "normal"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "abnormal"), exist_ok=True)

# === Function to create a combined IMU + FSR heatmap with fixed normalization range ===
def create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ):
    grid = np.zeros((10, 7))

    # --- Sensor placement: Left (Accel) → Center (Gyro) → Right (FSR) ---
    # Accelerometer (left)
    grid[2, 1] = accelX
    grid[4, 1] = accelY
    grid[6, 1] = accelZ

    # Gyroscope (middle)
    grid[2, 3] = gyroX
    grid[4, 3] = gyroY
    grid[6, 3] = gyroZ

    # FSR sensors (right)
    grid[2, 5] = fsr1
    grid[4, 5] = fsr2
    grid[6, 5] = fsr3

    # Resize to 28x28
    resized = zoom(grid, (28 / 10, 28 / 7), order=1)

    # === Fixed normalization range for consistency across all heatmaps ===
    # IMU data typically ranges from -2 to 2, FSR typically from 0 to 1
    # Use a safe global range: [-2, 2]
    resized = np.clip(resized, -2.0, 2.0)  # Clip extreme values
    normalized = (resized + 2.0) / 4.0      # Scale from [-2, 2] → [0, 1]

    # Generate RGB colormap
    colored = cm.jet(normalized)[:, :, :3]
    return Image.fromarray((colored * 255).astype(np.uint8))

# === Process all CSV files ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)

        required_cols = {"fsr1", "fsr2", "fsr3", "accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"}
        if required_cols.issubset(df.columns):
            # Average values from each column
            fsr1 = df["fsr1"].mean() * 5
            fsr2 = df["fsr2"].mean() * 5
            fsr3 = df["fsr3"].mean() * 5
            accelX = df["accelX"].mean()
            accelY = df["accelY"].mean()
            accelZ = df["accelZ"].mean()
            gyroX = df["gyroX"].mean()
            gyroY = df["gyroY"].mean()
            gyroZ = df["gyroZ"].mean()

            # Generate heatmap image
            heatmap = create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ)

            # Save under "normal" or "abnormal" folder
            label = "abnormal" if "abnormal" in filename.lower() else "normal"
            save_path = os.path.join(output_folder, label, filename.replace(".csv", ".png"))
            heatmap.save(save_path)
            print(f"Saved: {save_path}")
        else:
            print(f"Skipped {filename}: Missing required columns")

# === Summary ===
print("\nJumlah gambar Normal:", len(os.listdir(os.path.join(output_folder, "normal"))))
print("Jumlah gambar Abnormal:", len(os.listdir(os.path.join(output_folder, "abnormal"))))
