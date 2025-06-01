import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

# =========================
# 1. Baca Data
df = pd.read_csv("data_sensor.csv")
df = df.dropna(subset=["Gait"])  # Buang label kosong

# =========================
# 2. Encode Label
label_encoder = LabelEncoder()
df["Gait"] = label_encoder.fit_transform(df["Gait"])  # Normal = 0, Abnormal = 1

# =========================
# 3. Standardize Sensor Data
features = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'FSR1', 'FSR2', 'FSR3', 'FSR4']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# =========================
# 4. Sliding Window
window_size = 30
X, y = [], []

for i in range(len(df) - window_size + 1):
    window = df.iloc[i:i+window_size]
    label_counts = Counter(window['Gait'])
    dominant_label = label_counts.most_common(1)[0][0]  # Ambil label majoriti
    X.append(window[features].values)
    y.append(dominant_label)

X = np.array(X)  # shape: (n_samples, 30, 10)
y = np.array(y)  # shape: (n_samples,)

print("✅ Siap sliding window!")
print("Bentuk data X:", X.shape)
print("Bentuk label y:", y.shape)

# Simpan jika nak digunakan untuk model
np.save("X.npy", X)
np.save("y.npy", y)