import socket

UDP_IP = "172.20.10.10"  # IP PC
UDP_PORT = 3000
MESSAGE = b"Test UDP packet from PC"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
print("✅ Test packet sent")