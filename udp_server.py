import socket
import csv
from datetime import datetime

# UDP setup
HOST = '0.0.0.0'  # Listen from any IP
PORT = 3000       # Same as ESP32 target UDP port

# CSV setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"gait_data_{timestamp}.csv"

headers = [
    "accelX", "accelY", "accelZ",
    "gyroX", "gyroY", "gyroZ",
    "fsr1", "fsr2", "fsr3"
]

def start_udp_server():
    print(f"[🟢] Starting UDP server on port {PORT}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))

    # Write CSV header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    # Append incoming data
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        while True:
            data, addr = sock.recvfrom(1024)
            data_str = data.decode().strip()
            print(f"[📨] Received from {addr}: {data_str}")

            row = data_str.split(',')
            if len(row) == len(headers):
                writer.writerow(row)
                file.flush()
            else:
                print("[⚠️] Incomplete data:", row)

if __name__ == "__main__":
    start_udp_server()