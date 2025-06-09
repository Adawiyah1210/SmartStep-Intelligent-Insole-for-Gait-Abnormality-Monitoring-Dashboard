import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Load data ===
df = pd.read_csv("data 1.csv")

# === 2. Waktu (andaian setiap 0.5 saat) ===
time = np.arange(len(df)) * 0.5  # seconds

# === 3. Ambil data angular velocity dari gyroY ===
gyro_y = df["gyroY"].to_numpy()

# === 4. Threshold untuk swing phase ===
swing_threshold = 10
swing_mask = gyro_y > swing_threshold

# === 5. Cari peak angular velocity ===
peak_indices = np.where((np.roll(gyro_y, 1) < gyro_y) & (np.roll(gyro_y, -1) < gyro_y))[0]

# === 6. Masa untuk stride time (titik perubahan signifikan) ===
stride_indices = np.where(np.diff((gyro_y > 5).astype(int)) != 0)[0]
stride_times = time[stride_indices]

# === 7. Plot ===
plt.figure(figsize=(15, 6))
plt.plot(time, gyro_y, label="Shank Angular Velocity (gyroY)", color="blue")

# === 8. Tanda Peak Angular Velocity ===
if len(peak_indices) > 0:
    for i in range(min(3, len(peak_indices))):  # tunjuk max 3 peak
        peak_time = time[peak_indices[i]]
        peak_val = gyro_y[peak_indices[i]]
        plt.plot(peak_time, peak_val, 'r^', label="Peak Angular Velocity" if i == 0 else "")
        plt.text(peak_time, peak_val + 5, f"{peak_val:.1f}", color='red')

# === 9. Highlight Swing Phase ===
plt.fill_between(time, 0, gyro_y, where=swing_mask, color="lightgreen", alpha=0.4, label="Swing Phase")

# === 10. Garis Stride Time ===
for stride_time in stride_times:
    plt.axvline(stride_time, color="orange", linestyle="--", alpha=0.6)

# === 11. Label dan paparan ===
plt.xlabel("Time (s)")
plt.ylabel("Shank Angular Velocity (deg/s)")
plt.title("IMU-Based Gait Analysis (gyroY)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()