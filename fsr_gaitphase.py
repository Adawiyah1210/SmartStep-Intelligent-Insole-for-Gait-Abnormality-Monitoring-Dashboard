import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Load CSV ==========
file_path = "Augmented_scaled_data.csv"  # Tukar ke path file anda jika perlu
df = pd.read_csv(file_path)

# ========== 2. Extract Sensor Data ==========
fsr_heel = df["fsr4"].to_numpy()    # Heel
fsr_mid = df["fsr3"].to_numpy()     # Midfoot
fsr_fore1 = df["fsr1"].to_numpy()   # Forefoot 1
fsr_fore2 = df["fsr2"].to_numpy()   # Forefoot 2
time = np.arange(len(df)) * 0.5     # Assuming 500ms interval

# ========== 3. Tetapkan Threshold ==========
heel_thresh = 0.2
mid_thresh = 0.2
fore_thresh = 0.2
delta = 0.05  # Kenaikan/penurunan cepat

# ========== 4. Fasa Gait Detection ==========

# Initial Contact (IC): heel pressure naik tiba-tiba
delta_heel = np.diff(fsr_heel)
ic = np.where(delta_heel > delta)[0] + 1

# Foot Flat (FF): heel + midfoot > threshold
ff = np.where((fsr_mid > mid_thresh) & (fsr_heel > heel_thresh))[0]

# Midstance (MS): mid tinggi, heel kurang sikit
ms = np.where((fsr_mid > mid_thresh) & (fsr_heel < heel_thresh + 0.1))[0]

# Heel Lift (HL): heel drop tiba-tiba
hl = np.where(delta_heel < -delta)[0] + 1

# Toe Off (TO): forefoot drop tiba-tiba
delta_fore1 = np.diff(fsr_fore1)
delta_fore2 = np.diff(fsr_fore2)
to = np.where((delta_fore1 < -delta) & (delta_fore2 < -delta))[0] + 1

# ========== 5. Plot ==========

plt.figure(figsize=(15, 6))
plt.plot(time, fsr_heel, label="Heel (FSR4)", color="blue")
plt.plot(time, fsr_mid, label="Midfoot (FSR3)", color="orange")
plt.plot(time, fsr_fore1, label="Forefoot 1 (FSR1)", color="green")
plt.plot(time, fsr_fore2, label="Forefoot 2 (FSR2)", color="lime")

# Tanda setiap fasa
plt.plot(time[ic], fsr_heel[ic], 'ro', label="Initial Contact (IC)")
plt.plot(time[ff], fsr_mid[ff], 'mo', label="Foot Flat (FF)", markersize=4)
plt.plot(time[ms], fsr_mid[ms], 'co', label="Midstance (MS)", markersize=4)
plt.plot(time[hl], fsr_heel[hl], 'yo', label="Heel Lift (HL)", markersize=6)
plt.plot(time[to], fsr_fore1[to], 'ko', label="Toe Off (TO)", markersize=6)

plt.title("FSR Gait Phase Detection (Left Leg)")
plt.xlabel("Time (s)")
plt.ylabel("FSR Value")
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()