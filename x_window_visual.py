import numpy as np
import matplotlib.pyplot as plt

# Label feature dalam English
feature_names = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'FSR1', 'FSR2', 'FSR3', 'FSR4']

# Load data X
X = np.load("X.npy")

# Pilih window pertama
window = X[0]  # bentuk (30, 10)

# Plot semua 10 features dengan label feature names
plt.figure(figsize=(12, 6))
for i in range(window.shape[1]):
    plt.plot(window[:, i], label=feature_names[i])
plt.title("Visualize Window First (X[0])")
plt.xlabel("Time Step (0–29)")
plt.ylabel("Sensor Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Simpan gambar
plt.savefig("X0_window_visual.png")

# Papar gambar
plt.show()