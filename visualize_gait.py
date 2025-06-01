import pandas as pd
import matplotlib.pyplot as plt

# Baca CSV
df = pd.read_csv('data_sensor.csv')

# Pastikan column sesuai
print(df.columns)

# ===============================
# 1. Plot Acceleration
plt.figure(figsize=(12, 6))
plt.plot(df['accX'], label='accX')
plt.plot(df['accY'], label='accY')
plt.plot(df['accZ'], label='accZ')
plt.title('Acceleration Sensor Data')
plt.xlabel('Sample Number')
plt.ylabel('Acceleration')
plt.legend()
plt.savefig('acceleration_plot.png')
plt.close()

# ===============================
# 2. Plot Gyroscope
plt.figure(figsize=(12, 6))
plt.plot(df['gyroX'], label='gyroX')
plt.plot(df['gyroY'], label='gyroY')
plt.plot(df['gyroZ'], label='gyroZ')
plt.title('Gyroscope Sensor Data')
plt.xlabel('Sample Number')
plt.ylabel('Angular Velocity')
plt.legend()
plt.savefig('gyroscope_plot.png')
plt.close()

# ===============================
# 3. Plot FSR
plt.figure(figsize=(12, 6))
plt.plot(df['FSR1'], label='FSR1')
plt.plot(df['FSR2'], label='FSR2')
plt.plot(df['FSR3'], label='FSR3')
plt.plot(df['FSR4'], label='FSR4')
plt.title('FSR Sensor Forces')
plt.xlabel('Sample Number')
plt.ylabel('Force (N)')
plt.legend()
plt.savefig('fsr_plot.png')
plt.close()

# ===============================
# 4. Plot Label Gait (warna ikut kategori)
plt.figure(figsize=(12, 6))
color_map = {'Normal': 'green', 'Limping': 'red'}
colors = df['Gait'].map(color_map)

plt.scatter(df.index, [1]*len(df), c=colors, label='Gait Label', marker='|', s=100)
plt.title('Gait Classification Visualization')
plt.yticks([])
plt.xlabel('Sample Number')
plt.savefig('gait_label_plot.png')
plt.close()

# ===============================
# 5. Kiraan Bilangan Gait
gait_counts = df['Gait'].value_counts()
print("\n🔍 Bilangan Gait Classes:")
print(gait_counts)

# Optional: Simpan ke fail text
with open('gait_summary.txt', 'w') as f:
    f.write("Gait Class Count:\n")
    f.write(gait_counts.to_string())