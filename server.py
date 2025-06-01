import socket

# Tetapan server
HOST = '172.20.10.10'   # Terima sambungan dari mana-mana IP
PORT = 3000        # Sama dengan port dalam Arduino

# Cipta socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"[INFO] Menunggu sambungan di {HOST}:{PORT}...")
client_socket, client_address = server_socket.accept()
print(f"[INFO] Sambungan dari {client_address} diterima!")

try:
    while True:
        data = client_socket.recv(1024).decode().strip()
        if not data:
            break
        print(f"[DATA] {data}")
        # Di sini nanti kita boleh sambung ke model deep learning untuk klasifikasi real-time
except KeyboardInterrupt:
    print("\n[INFO] Dihentikan secara manual.")
finally:
    client_socket.close()
    server_socket.close()
    print("[INFO] Sambungan ditutup.")