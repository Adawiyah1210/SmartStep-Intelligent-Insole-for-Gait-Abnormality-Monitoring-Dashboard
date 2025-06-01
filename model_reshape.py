import numpy as np
from tensorflow.keras.models import load_model

# Contoh buffer data: 30 baris × 6 ciri (accX, accY, accZ, gyroX, gyroY, gyroZ)
# Gantikan ini dengan data sebenar daripada sensor Nur nanti
buffer = [
    [0.12, 0.24, 9.65, 0.01, -0.02, 0.005],
    [0.10, 0.23, 9.62, 0.02, -0.01, 0.004],
    # ... (28 baris lagi)
] * 30  # kalau hanya ada 1 baris, ulang untuk cukupkan 30 baris sebagai demo

# Tukar buffer ke NumPy array dan reshape ikut model
data_seq = np.array(buffer).reshape(1, 30, 6)

# Load model
model = load_model('model_cnn_lstm.h5')

# Buat ramalan
prediction = model.predict(data_seq)

# Label output (ikut model Nur: 0 = Normal, 1 = Abnormal)
labels = ['Normal', 'Abnormal']
predicted_class = np.argmax(prediction, axis=1)[0]
print("Gait Classification:", labels[predicted_class])