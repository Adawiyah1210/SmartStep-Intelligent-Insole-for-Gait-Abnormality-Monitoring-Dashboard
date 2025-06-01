import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

# === Fail input ===
csv_file = "new_gait_window.csv"
foot_image = "foot_right.png"

# === Baca CSV dan gambar tapak kaki ===
df = pd.read_csv(csv_file)
foot_img = mpimg.imread(foot_image)

# === Lokasi FSR atas imej (x, y) ===
fsr_positions = {
    'fsr1': (150, 620),  # tumit bawah
    'fsr2': (150, 120),  # ibu jari
    'fsr3': (190, 140),  # jari tengah
    'fsr4': (230, 300),  # sisi luar kaki
}
sensors = list(fsr_positions.keys())

# === Sediakan figura ===
fig, ax = plt.subplots(figsize=(6, 9))
ax.imshow(foot_img)
ax.axis('off')

# === Plot kosong awal ===
scatter = ax.scatter([], [], s=[], c=[], cmap='hot', vmin=0, vmax=100)

# === Fungsi update untuk setiap frame ===
def update(frame):
    row = df.iloc[frame]
    pressures = [row[s] for s in sensors]
    positions = np.array([fsr_positions[s] for s in sensors])
    scatter.set_offsets(positions)
    scatter.set_sizes([p * 2 for p in pressures])
    scatter.set_array(np.array(pressures))
    return scatter,

# === Animasi ===
ani = FuncAnimation(fig, update, frames=len(df), interval=200, blit=True)

# === Simpan sebagai GIF ===
ani.save("gait_heatmap_animated.gif", writer=PillowWriter(fps=5))
print("✅ GIF telah disimpan sebagai gait_heatmap_animated.gif")