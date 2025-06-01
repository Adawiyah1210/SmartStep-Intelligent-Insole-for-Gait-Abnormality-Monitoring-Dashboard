import pandas as pd
import numpy as np

# Load data dari gait_data.csv
df = pd.read_csv('gait_data.csv')

window_size = 30  # contoh 30 frame/window
step_size = 15    # 50% overlap

features_list = []

for start in range(0, len(df) - window_size + 1, step_size):
    window = df.iloc[start:start + window_size]

    features = {}
    sensors = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ', 'fsr1', 'fsr2', 'fsr3', 'fsr4']

    for sensor in sensors:
        data = window[sensor].values
        features[f'mean_{sensor}'] = np.mean(data)
        features[f'std_{sensor}'] = np.std(data)
        features[f'min_{sensor}'] = np.min(data)
        features[f'max_{sensor}'] = np.max(data)
        features[f'range_{sensor}'] = np.max(data) - np.min(data)
        features[f'rms_{sensor}'] = np.sqrt(np.mean(data**2))
        zero_crossings = ((data[:-1] * data[1:]) < 0).sum()
        features[f'zero_cross_{sensor}'] = zero_crossings

    features_list.append(features)

features_df = pd.DataFrame(features_list)
features_df.to_csv('gait_features_extracted.csv', index=False)

print("Selesai extract feature dari gait_data.csv. Output saved to gait_features_extracted.csv")