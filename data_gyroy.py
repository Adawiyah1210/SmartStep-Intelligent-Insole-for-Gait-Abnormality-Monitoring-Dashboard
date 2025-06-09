import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === 1. Load data ===
df = pd.read_csv("data11.csv")
gyro_y = df["gyroY"].to_numpy()
time = np.arange(len(gyro_y)) * 0.5  # assuming 500ms interval

# === 2. Mid-Swing (X mark): Peaks in gyroY ===
peaks, _ = find_peaks(gyro_y, height=10, distance=5)

# === 3. Toe-Off (Green): local minima before peak ===
toe_offs = []
heel_strikes = []
for p in peaks:
    # Cari minimum sebelum dan selepas peak (dalam julat tertentu)
    search_before = gyro_y[max(0, p - 10):p]
    search_after = gyro_y[p:min(len(gyro_y), p + 10)]

    if len(search_before) > 0:
        toe_off_idx = np.argmin(search_before) + max(0, p - 10)
        toe_offs.append(toe_off_idx)

    if len(search_after) > 0:
        heel_strike_idx = np.argmin(search_after) + p
        heel_strikes.append(heel_strike_idx)

# === 4. Plot ===
plt.figure(figsize=(15, 6))
plt.plot(time, gyro_y, label="Gyro Y", color='blue')

# Plot Mid-Swing (X)
plt.plot(time[peaks], gyro_y[peaks], 'kx', label='Mid-Swing')

# Plot Toe Off (Green)
plt.plot(time[toe_offs], gyro_y[toe_offs], 'go', label='Toe Off')

# Plot Heel Strike (Red)
plt.plot(time[heel_strikes], gyro_y[heel_strikes], 'ro', label='Heel Strike')

# Optionally mark stride/swing/stance times
for i in range(len(peaks) - 1):
    t_start = time[heel_strikes[i]]
    t_toe = time[toe_offs[i]]
    t_mid = time[peaks[i]]
    t_next_hs = time[heel_strikes[i + 1]]

    # Stride time
    plt.axvline(t_start, linestyle="--", color="gray", alpha=0.5)
    plt.axvline(t_next_hs, linestyle="--", color="gray", alpha=0.5)
    plt.text((t_start + t_next_hs) / 2, -150, "Stride", ha='center', fontsize=8)

    # Swing time
    plt.axvline(t_toe, linestyle=":", color="green", alpha=0.6)
    plt.axvline(t_next_hs, linestyle=":", color="green", alpha=0.6)
    plt.text((t_toe + t_next_hs) / 2, -120, "Swing", ha='center', fontsize=8)

    # Stance time
    plt.text((t_start + t_toe) / 2, -90, "Stance", ha='center', fontsize=8)

plt.title("Gait Event Detection using Gyro Y")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity [deg/s]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()