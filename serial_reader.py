import csv
import serial
import time

def main():
    try:
        ser = serial.Serial('COM3', 9600, timeout=1)
        print("Membaca data dari Arduino di COM3...")

        with open('data_sensor.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Tulis header ikut format data Arduino kamu
            writer.writerow(['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'FSR1', 'FSR2', 'FSR3', 'FSR4', 'Gait'])
            
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(">>", line)
                    data = line.split(',')
                    if len(data) == 11:  # Pastikan data lengkap 11 elemen
                        writer.writerow(data)

    except serial.SerialException:
        print("Tidak dapat sambung ke COM3. Pastikan port betul dan tidak digunakan program lain.")
    except KeyboardInterrupt:
        print("\nBerhenti membaca data serial.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()