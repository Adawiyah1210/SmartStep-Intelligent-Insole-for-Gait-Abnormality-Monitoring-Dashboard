import socket
import pandas as pd
from datetime import datetime

UDP_IP = "0.0.0.0"
UDP_PORT = 3000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"[🟢] UDP server running on port {UDP_PORT}...")

data_list = []

while True:
    data, addr = sock.recvfrom(1024)
    decoded = data.decode().strip()
    print(f"[📨] Received from {addr}: {decoded}")

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    values = decoded.split(',')

    if len(values) == 9:
        data_list.append([timestamp] + values)

    # Simpan setiap 30 baris
    if len(data_list) >= 30:
        df = pd.DataFrame(data_list, columns=['Time', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'FSR1', 'FSR2', 'FSR3'])
        df.to_excel("udp_sensor_data.xlsx", index=False)
        print("✅ Data saved to udp_sensor_data.xlsx")
        data_list = []