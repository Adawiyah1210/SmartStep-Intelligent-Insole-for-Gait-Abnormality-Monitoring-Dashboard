import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def extract_stat_features(window):
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
        features[f'zero_cross_{sensor}'] = ((data[:-1] * data[1:]) < 0).sum()
        features[f'energy_{sensor}'] = np.sum(data**2)
        features[f'slope_{sensor}'] = np.mean(np.abs(np.diff(data)))
    return features

def extract_dynamic_features(window, fsr_threshold=0.2):
    features = {}
    fsr4 = window['fsr4'].values
    peaks, _ = find_peaks(fsr4, height=fsr_threshold, distance=10)
    features['step_count_fsr4'] = len(peaks)

    if len(peaks) >= 2:
        stride_times = np.diff(peaks) / 30.0  # Asumsi 30 fps
        features['mean_stride_time'] = np.mean(stride_times)
        features['std_stride_time'] = np.std(stride_times)
    else:
        features['mean_stride_time'] = 0
        features['std_stride_time'] = 0

    stance_time = np.sum(fsr4 > fsr_threshold) / 30.0
    total_time = len(fsr4) / 30.0
    features['stance_time_ratio'] = stance_time / total_time if total_time > 0 else 0
    features['swing_time_ratio'] = 1 - features['stance_time_ratio']

    toe_peak = np.argmax(window['fsr1'].values)
    midfoot_peak = np.argmax(window[['fsr2','fsr3']].values, axis=0).mean()
    heel_peak = np.argmax(fsr4)
    features['load_shift_order'] = int(toe_peak < midfoot_peak < heel_peak)

    return features

def extract_combination_features(window):
    features = {}
    gyroX = window['gyroX'].values
    fsr4 = window['fsr4'].values
    fsr4_peaks, _ = find_peaks(fsr4, height=0.2, distance=10)
    gyroX_peaks, _ = find_peaks(gyroX, height=np.percentile(gyroX, 90), distance=10)

    overlap_peaks = 0
    for p in fsr4_peaks:
        if any(abs(p - g) < 3 for g in gyroX_peaks):
            overlap_peaks += 1
    features['heel_strike_events'] = overlap_peaks

    fsr1 = window['fsr1'].values
    gyroZ = window['gyroZ'].values
    toe_off_events = np.sum((fsr1 > 0.2) & (gyroZ < np.percentile(gyroZ, 10)))
    features['toe_off_events'] = toe_off_events

    features['gyroY_peak'] = np.max(window['gyroY'].values)
    features['accelY_peak'] = np.max(window['accelY'].values)

    return features

def full_feature_extractor(window):
    features = {}
    features.update(extract_stat_features(window))
    features.update(extract_dynamic_features(window))
    features.update(extract_combination_features(window))
    return features

if __name__ == "__main__":
    df = pd.read_csv('gait_data_new.csv')

    window_size = 30
    step_size = 15
    features_list = []

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start+window_size]
        feat = full_feature_extractor(window)

        # Dapatkan label paling kerap dalam window (pastikan ada kolom 'label' di csv)
        window_label = window['label'].mode()[0]

        feat['label'] = window_label
        features_list.append(feat)

    features_df = pd.DataFrame(features_list)
    features_df.to_csv('gait_features_new.csv', index=False)
    print("Feature extraction lengkap siap! Output disimpan ke gait_features_new.csv")