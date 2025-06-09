
import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import zoom

# === Tetapan folder ===
input_folder = "csv_files"  # Folder CSV asal
output_folder = "dataset"
os.makedirs(os.path.join(output_folder, "normal"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "abnormal"), exist_ok=True)

# === Fungsi untuk buat heatmap dalam bentuk kaki ===
def create_foot_shaped_heatmap(fsr1, fsr2, fsr3):
    grid = np.zeros((10, 7))

    # Lokasi FSR:
    grid[3, 2] = fsr1   # fsr1 – bola kiri kaki
    grid[2, 4] = fsr2   # fsr2 – bola kanan kaki
    grid[8, 3] = fsr3   # fsr3 – tumit

    resized = zoom(grid, (28 / 10, 28 / 7), order=1)
    normalized = resized / np.max(resized) if np.max(resized) > 0 else resized
    colored = cm.jet(normalized)[:, :, :3]  # buang alpha channel
    return Image.fromarray((colored * 255).astype(np.uint8))

# === Proses semua fail CSV ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)

        # Pastikan ada semua kolum FSR
        if {"fsr1", "fsr2", "fsr3"}.issubset(df.columns):
            fsr1 = df["fsr1"].mean()
            fsr2 = df["fsr2"].mean()
            fsr3 = df["fsr3"].mean()

            heatmap = create_foot_shaped_heatmap(fsr1, fsr2, fsr3)

            # === Label: utamakan 'abnormal' ===
            label = "abnormal" if "abnormal" in filename.lower() else "normal"
            save_path = os.path.join(output_folder, label, filename.replace(".csv", ".png"))
            heatmap.save(save_path)
            print(f"Saved: {save_path}")
        else:
            print(f"Skipped {filename}: Missing FSR columns")

# === Papar jumlah imej ===
print("\nJumlah gambar Normal:", len(os.listdir(os.path.join(output_folder, "normal"))))
print("Jumlah gambar Abnormal:", len(os.listdir(os.path.join(output_folder, "abnormal"))))