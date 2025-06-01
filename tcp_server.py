import socket
import csv
from datetime import datetime

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

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[🔌] Server running, listening on port {PORT}")

        # Write CSV header once at start
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

        while True:
            print("[🔌] Waiting for ESP32 connection...")
            conn, addr = s.accept()
            print(f"[✅] Connected by {addr}")

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
                            writer.writerow(row)
                            file.flush()
                        else:
                            print("[⚠️] Incomplete data format:", row)

                except Exception as e:
                    print(f"[❌] Error: {e}")

if __name__ == "__main__":
    start_server()