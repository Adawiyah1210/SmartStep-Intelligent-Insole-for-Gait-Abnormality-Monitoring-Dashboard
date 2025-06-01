import socket
import csv
from datetime import datetime
import pandas as pd
import joblib
from full_feature_extractor import full_feature_extractor

HOST = '0.0.0.0'
PORT = 3000

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"gait_data_{timestamp}.csv"

headers = [
    "accelX", "accelY", "accelZ",
    "gyroX", "gyroY", "gyroZ",
    "fsr1", "fsr2", "fsr3", "fsr4",
    "label"
]

# Load model sekali saja sebelum loop
model = joblib.load('gait_classifier_rf.pkl')
model_features = model.feature_names_in_

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[🔌] Server running, listening on port {PORT}")

        # Tulis header CSV sekali saja
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

        while True:
            print("[🔌] Waiting for ESP32 connection...")
            conn, addr = s.accept()
            print(f"[✅] Connected by {addr}")

            buffer = []  # kumpul data baris untuk window
            window_size = 30  # contoh window size (ubah ikut keperluan)
            step_size = 15    # sliding window step
            
            with conn, open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                try:
                    while True:
                        data = conn.recv(1024).decode().strip()
                        if not data:
                            print("[ℹ️] ESP32 disconnected.")
                            break

                        print(f"[📥] {data}")
                        row = data.split(',')

                        if len(row) == len(headers):
                            # Simpan ke csv
                            writer.writerow(row)
                            file.flush()

                            # Tambah ke buffer untuk buat sliding window
                            buffer.append(row)

                            # Jika buffer cukup panjang, buat sliding window dan klasifikasi
                            if len(buffer) >= window_size:
                                # Ambil window data sebagai DataFrame
                                window_df = pd.DataFrame(buffer[:window_size], columns=headers)

                                # Tukar data ke jenis sesuai, kecuali label
                                for col in headers[:-1]:
                                    window_df[col] = pd.to_numeric(window_df[col], errors='coerce')

                                # Extract fitur
                                features = full_feature_extractor(window_df)

                                # Susun DataFrame features ikut feature names model
                                df_feat = pd.DataFrame([features])
                                df_feat = df_feat[model_features]

                                # Predict
                                pred_label = model.predict(df_feat)[0]

                                print(f"[🦶] Gait classification: {pred_label}")

                                # Buat sliding window (step size)
                                buffer = buffer[step_size:]
                        else:
                            print("[⚠️] Incomplete data format:", row)

                except Exception as e:
                    print(f"[❌] Error: {e}")

if __name__ == "__main__":
    start_server()