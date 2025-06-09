import os
import pandas as pd
import numpy as np

def augment_data(df, num_aug=3, noise_level=0.05):
    augmented_dfs = []
    sensor_cols = [col for col in df.columns if col.lower() != 'timestamp']  # Semua kolum sensor kecuali timestamp
    for i in range(num_aug):
        df_aug = df.copy()
        for col in sensor_cols:
            std = df[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(df))
            df_aug[col] = df[col] + noise

        # Optional clamp (contoh, FSR dan accelerometer tak boleh negatif)
        for col in sensor_cols:
            # Kalau kolum FSR atau acc, pastikan >= 0
            if 'fsr' in col.lower() or 'acc' in col.lower():
                df_aug[col] = df_aug[col].clip(lower=0)

        augmented_dfs.append(df_aug)
    return augmented_dfs

def augment_folder(csv_dir, output_dir, target_class="abnormal", target_num=20):
    os.makedirs(output_dir, exist_ok=True)

    abnormal_files = [f for f in os.listdir(csv_dir) if target_class in f.lower() and f.endswith('.csv')]
    print(f"Abnormal files found: {len(abnormal_files)}")

    all_abnormal_dfs = []
    for f in abnormal_files:
        df = pd.read_csv(os.path.join(csv_dir, f))
        all_abnormal_dfs.append((f, df))

    current_num = len(abnormal_files)
    if current_num >= target_num:
        print("Data sudah mencukupi, tiada augmentasi diperlukan.")
        return

    augment_per_file = (target_num - current_num) // current_num + 1
    print(f"Augmentasi setiap fail: {augment_per_file}")

    # Salin asal ke folder baru
    for f, df in all_abnormal_dfs:
        df.to_csv(os.path.join(output_dir, f), index=False)

    count = current_num
    for f, df in all_abnormal_dfs:
        augmented = augment_data(df, num_aug=augment_per_file)
        base_name = f.replace('.csv', '')
        for i, df_aug in enumerate(augmented):
            if count >= target_num:
                break
            new_name = f"{base_name}_aug{i+1}.csv"
            df_aug.to_csv(os.path.join(output_dir, new_name), index=False)
            count += 1
            print(f"[AUG] {f} → {new_name}")

    print(f"Total abnormal data after augmentation: {count}")

# Run example
augment_folder("csv_files", "augmented_csv_files", target_class="abnormal", target_num=20)
