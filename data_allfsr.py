import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data2.csv")

# Sensor Mapping
fsr_heel = df["fsr4"].to_numpy()
fsr_fore1 = df["fsr1"].to_numpy()
fsr_fore2 = df["fsr2"].to_numpy()
fsr_mid = df["fsr3"].to_numpy()
time = np.arange(len(df)) * 0.5  # 500ms interval

# --- DETECTION CONFIGURATION ---
heel_threshold = 0.2
fore_threshold = 0.2
delta_threshold = 0.1  # minimum change to detect event

# --- DETECTION LOGIC ---

# Heel strike = bila heel meningkat secara mendadak
delta_heel = np.diff(fsr_heel)
heel_strikes = np.where(delta_heel > delta_threshold)[0] + 1

# Toe off = bila kedua forefoot turun secara mendadak
delta_fore1 = np.diff(fsr_fore1)
delta_fore2 = np.diff(fsr_fore2)
toe_offs = np.where((delta_fore1 < -delta_threshold) & (delta_fore2 < -delta_threshold))[0] + 1

print(f"Heel Strikes detected: {len(heel_strikes)}")
print(f"Toe Offs detected: {len(toe_offs)}")

# --- PLOTTING ---
plt.figure(figsize=(15, 6))
plt.plot(time, fsr_heel, label="Heel (FSR4)", color="blue")
plt.plot(time, fsr_fore1, label="Forefoot 1 (FSR1)", color="green")
plt.plot(time, fsr_fore2, label="Forefoot 2 (FSR2)", color="lime")
plt.plot(time, fsr_mid, label="Midfoot (FSR3)", color="orange")

# Plot events
plt.plot(time[heel_strikes], fsr_heel[heel_strikes], 'ro', label="Heel Strike", markersize=8)
plt.plot(time[toe_offs], fsr_fore1[toe_offs], 'mo', label="Toe Off", markersize=8)

plt.title("Gait Event Detection (Low-Range FSR Data)")
plt.xlabel("Time (s)")
plt.ylabel("FSR Sensor Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()