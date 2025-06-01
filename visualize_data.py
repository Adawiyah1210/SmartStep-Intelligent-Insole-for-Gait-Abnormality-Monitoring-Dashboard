import pandas as pd
import matplotlib.pyplot as plt

# Baca data CSV
df = pd.read_csv('data_sensor.csv')

# Pastikan data dah betul-betul loaded
print(df.head())  # Print 5 baris pertama

# Plot acceleration data (accX, accY, accZ)
plt.figure(figsize=(12, 6))
plt.plot(df['accX'], label='accX')
plt.plot(df['accY'], label='accY')
plt.plot(df['accZ'], label='accZ')
plt.title('Acceleration Sensor Data')
plt.xlabel('Sample Number')
plt.ylabel('Acceleration')
plt.legend()
plt.show()

# Plot gyro data (gyroX, gyroY, gyroZ)
plt.figure(figsize=(12, 6))
plt.plot(df['gyroX'], label='gyroX')
plt.plot(df['gyroY'], label='gyroY')
plt.plot(df['gyroZ'], label='gyroZ')
plt.title('Gyroscope Sensor Data')
plt.xlabel('Sample Number')
plt.ylabel('Angular Velocity')
plt.legend()
plt.show()

# Plot FSR forces (FSR1, FSR2, FSR3, FSR4)
plt.figure(figsize=(12, 6))
plt.plot(df['FSR1'], label='FSR1')
plt.plot(df['FSR2'], label='FSR2')
plt.plot(df['FSR3'], label='FSR3')
plt.plot(df['FSR4'], label='FSR4')
plt.title('FSR Sensor Forces')
plt.xlabel('Sample Number')
plt.ylabel('Force (N)')
plt.legend()
plt.show()