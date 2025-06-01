import socket
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from collections import deque
from datetime import datetime

# ======= Konfigurasi =======
HOST = '0.0.0.0'
PORT = 3000
MODEL_PATH = 'model_cnn_lstm.h5'
USERNAME = 'Nur'
SEQ_LEN = 30  # Pastikan sama dengan yang digunakan dalam training
EXCEL_FILE = 'log_gait_nur.xlsx'

# Posisi FSR dalam grid tapak kaki kanan (x, y)
fsr_positions = {
    'FSR1': (2, 4),  # Toe
    'FSR2': (2, 3),  # Midfoot
    'FSR3': (2, 2),  # Heel outer
    'FSR4': (3, 2)   # Heel inner
}

buffer = deque(maxlen=SEQ_LEN)
data_log = []

# ======= Load Model Deep Learning =======
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ======= Fungsi Heatmap =======
def show_heatmap(fsr_values, label):
    grid = np.zeros((6, 6))
    for i, (name, (x, y)) in enumerate(fsr_positions.items()):
        grid[y, x] = fsr_values[i]
    plt.clf()
    plt.imshow(grid, cmap='jet', interpolation='nearest', vmin=0, vmax=25)
    plt.title(f"{USERNAME} | Gait: {label}")
    plt.axis('off')
    plt.pause(0.01)

# ======= Fungsi Simpan Excel =======
def save_to_excel():
    columns = ['Timestamp', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'FSR1', 'FSR2', 'FSR3', 'FSR4', 'PredictedLabel']
    df = pd.DataFrame(data_log, columns=columns)
    df.to_excel(EXCEL_FILE, index=False)
    print(f"💾 Data saved to {EXCEL_FILE}")

# ======= Fungsi Urus Sambungan ESP32 =======
def handle_client(conn, addr):
    global buffer, data_log
    print(f"📡 ESP32 connected from {addr}")

    while True:
        try:
            raw = conn.recv(1024).decode().strip()
            if not raw:
                break

            parts = raw.split(',')
            if len(parts) != 11:
                print("⚠️ Format salah:", raw)
                continue

            ax, ay, az = map(float, parts[0:3])
            gx, gy, gz = map(float, parts[3:6])
            f1, f2, f3, f4 = map(float, parts[6:10])
            _label = parts[10]

            buffer.append([ax, ay, az, gx, gy, gz])

            if len(buffer) == SEQ_LEN:
                seq_input = np.array(buffer).reshape(1, SEQ_LEN, 6)
                prediction = model.predict(seq_input, verbose=0)
                predicted_label = 'Normal' if np.argmax(prediction) == 0 else 'Abnormal'
            else:
                predicted_label = f"Loading ({len(buffer)}/{SEQ_LEN})"

            fsr_values = [f1, f2, f3, f4]
            show_heatmap(fsr_values, predicted_label)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_log.append([timestamp, ax, ay, az, gx, gy, gz, f1, f2, f3, f4, predicted_label])

            print(f"{timestamp}  {ax:.2f} {ay:.2f} {az:.2f}  {gx:.2f} {gy:.2f} {gz:.2f}  {f1:.2f} {f2:.2f} {f3:.2f} {f4:.2f}  {predicted_label}")

        except Exception as e:
            print("❌ Error:", e)
            break

    conn.close()
    print("🔌 ESP32 disconnected")
    save_to_excel()

# ======= TCP Server =======
def start_server():
    plt.ion()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"🚀 Server ready at {HOST}:{PORT}")

    while True:
        print("🔄 Waiting for ESP32 connection...")
        conn, addr = s.accept()
        handle_client(conn, addr)

# ======= Main Run =======
if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n🛑 Server manually stopped.")
    finally:
        plt.ioff()
        plt.show()