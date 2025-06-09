import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("data 1.csv")

# Time vector
time = np.arange(len(df)) * 0.5  # andaian: data dihantar setiap 500ms

# Jumlah tekanan (proxy untuk contact phase)
df["fsr_total"] = df[["fsr1", "fsr2", "fsr3", "fsr4"]].sum(axis=1)

# Plot tekanan sepanjang masa
plt.figure(figsize=(14, 6))
plt.plot(time, df["fsr_total"], color='darkgreen', label='Total FSR Force')
plt.xlabel("Time (s)")
plt.ylabel("Total Force (a.u.)")
plt.title("Force over Time (Based on FSR)")
plt.grid(True)

# Cari fase ayunan (swing = force ~ 0)
threshold = 0.1
swing_mask = df["fsr_total"] < threshold
contact_mask = df["fsr_total"] >= threshold

# Tandakan fasa ayunan
plt.fill_between(time, 0, df["fsr_total"], where=swing_mask, color='lightblue', alpha=0.4, label='Swing Phase')

# Tunjukkan peralihan sebagai stride approximation
stride_indices = np.where(np.diff(contact_mask.astype(int)) != 0)[0]
for idx in stride_indices:
    plt.axvline(time[idx], color='orange', linestyle='--', alpha=0.5)

plt.legend()
plt.tight_layout()
plt.show()