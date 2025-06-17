import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as cm
from scipy.ndimage import zoom

# === Folder setup ===
input_folder = "csv_files"  # Folder CSV input
output_folder = "dataset"
os.makedirs(os.path.join(output_folder, "normal"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "abnormal"), exist_ok=True)

# === Fungsi untuk hasilkan spatial heatmap (28x28x3) ===
def create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ):
    grid = np.zeros((10, 7))

    # FSR mappings
    grid[3, 2] = fsr1     # ball (left)
    grid[2, 4] = fsr2     # ball (right)
    grid[8, 3] = fsr3     # heel

    # Accelerometer mappings
    grid[1, 2] = accelX   # toe
    grid[4, 3] = accelY   # arch
    grid[6, 4] = accelZ   # vertical heel accel

    # Gyroscope mappings
    grid[2, 5] = gyroX    # toe rotation
    grid[5, 2] = gyroY    # arch rotation
    grid[7, 3] = gyroZ    # heel rotation

    # Resize to 28x28
    resized = zoom(grid, (28 / 10, 28 / 7), order=1)

    # Normalize & convert to RGB image
    normalized = resized / np.max(resized) if np.max(resized) > 0 else resized
    colored = cm.jet(normalized)[:, :, :3]  # ambil RGB shj
    image = Image.fromarray((colored * 255).astype(np.uint8))
    image = image.filter(ImageFilter.SHARPEN)
    return image

# === Proses setiap CSV ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)

        # Pastikan kolum wajib ada
        required_cols = {"fsr1", "fsr2", "fsr3", "accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"}
        if required_cols.issubset(df.columns):
            # Kira purata setiap sensor
            fsr1 = df["fsr1"].mean()
            fsr2 = df["fsr2"].mean()
            fsr3 = df["fsr3"].mean()
            accelX = df["accelX"].mean()
            accelY = df["accelY"].mean()
            accelZ = df["accelZ"].mean()
            gyroX = df["gyroX"].mean()
            gyroY = df["gyroY"].mean()
            gyroZ = df["gyroZ"].mean()

            # Hasilkan heatmap image
            image = create_combined_heatmap(fsr1, fsr2, fsr3, accelX, accelY, accelZ, gyroX, gyroY, gyroZ)

            # Tentukan label ikut nama fail
            label = "abnormal" if "abnormal" in filename.lower() else "normal"

            # Simpan imej
            save_path = os.path.join(output_folder, label, f"{filename.replace('.csv', '.png')}")
            image.save(save_path)

            print(f"Saved: {save_path}")
        else:
            print(f"Skipped {filename}: Missing required columns")

# === Ringkasan ===
print("\nTotal normal images:", len(os.listdir(os.path.join(output_folder, "normal"))))
print("Total abnormal images:", len(os.listdir(os.path.join(output_folder, "abnormal"))))