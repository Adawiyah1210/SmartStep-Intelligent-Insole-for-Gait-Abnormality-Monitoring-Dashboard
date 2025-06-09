import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Load data ===
df = pd.read_csv("data 1.csv")

# === 2. Ambil semua paksi gyro ===
gyro_x = df["gyroX"].to_numpy()
gyro_y = df["gyroY"].to_numpy()
gyro_z = df["gyroZ"].to_numpy()

# === 3. Waktu (500ms sampling) ===
time = np.arange(len(df)) * 0.5

# === 4. Kira magnitud gabungan ===
gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

# === 5. Threshold untuk swing phase ===
swing_threshold = 15
swing_mask = gyro_mag > swing_threshold

# === 6. Cari peak angular velocity ===
peak_indices = np.where((np.roll(gyro_mag, 1) < gyro_mag) & (np.roll(gyro_mag, -1) < gyro_mag))[0]

# === 7. Cari stride time ===
stride_indices = np.where(np.diff((gyro_mag > 10).astype(int)) != 0)[0]
stride_times = time[stride_indices]

# === 8. Plot ===
plt.figure(figsize=(15, 6))
plt.plot(time, gyro_mag, label="Gyro Magnitude", color="purple")

# === 9. Tanda peak ===
for i in range(min(3, len(peak_indices))):
    pt = time[peak_indices[i]]
    pv = gyro_mag[peak_indices[i]]
    plt.plot(pt, pv, 'r^', label="Peak Angular Velocity" if i == 0 else "")
    plt.text(pt, pv + 2, f"{pv:.1f}", color='red')

# === 10. Highlight swing phase ===
plt.fill_between(time, 0, gyro_mag, where=swing_mask, color="lightgreen", alpha=0.4, label="Swing Phase")

# === 11. Garis stride ===
for s_time in stride_times:
    plt.axvline(s_time, color="orange", linestyle="--", alpha=0.6)

# === 12. Final Display ===
plt.xlabel("Time (s)")
plt.ylabel("Gyro Magnitude (deg/s)")
plt.title("IMU-Based Gait Analysis (All Axes Combined)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()