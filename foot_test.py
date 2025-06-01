

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
import cv2

# Load CSV data sensor
df = pd.read_csv("new_gait_window.csv")

# Load gambar tapak kaki
foot_img = mpimg.imread("foot_right.png")

# Convert gambar ke grayscale dan buat mask (thresholding)
foot_gray = cv2.cvtColor((foot_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(foot_gray, 250, 255, cv2.THRESH_BINARY_INV)  # Putih jadi hitam, kaki putih

# Koordinat sensor FSR ikut gambar
sensor_coords = np.array([
    [50, 250],   # fsr1 heel
    [40, 150],   # fsr2 mid-left
    [80, 150],   # fsr3 mid-right
    [65, 50]     # fsr4 toe
])

# Grid untuk interpolation (tinggi x lebar)
grid_x, grid_y = np.mgrid[0:100:400j, 0:300:800j]

# Cari nilai max tekanan sensor dalam data untuk vmax heatmap
max_pressure = np.max(df[['fsr1','fsr2','fsr3','fsr4']].values)

# Setup figure dan axis
fig, ax = plt.subplots(figsize=(4, 8))
ax.imshow(foot_img, extent=[0, 100, 300, 0])
heatmap = ax.imshow(np.zeros_like(grid_x.T), extent=[0, 100, 300, 0], origin='lower',
                    cmap='jet', alpha=0.7, vmin=0, vmax=max_pressure)
plt.axis('off')
plt.title("Masked & Smoothed FSR Pressure Heatmap Over Foot")

def update(frame):
    pressures = df.iloc[frame][['fsr1', 'fsr2', 'fsr3', 'fsr4']].values.astype(float)
    print(f"Frame {frame}, pressures: {pressures}")  # Debug print

    grid_z = griddata(sensor_coords, pressures, (grid_x, grid_y), method='cubic', fill_value=0)
    smooth_grid = gaussian_filter(grid_z, sigma=10)

    mask_resized = cv2.resize(mask, (smooth_grid.shape[1], smooth_grid.shape[0]))
    smooth_grid[mask_resized == 0] = 0

    heatmap.set_data(smooth_grid.T)
    return [heatmap]

ani = FuncAnimation(fig, update, frames=len(df), interval=500, blit=True)
plt.show()