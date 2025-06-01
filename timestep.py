from tensorflow.keras.models import load_model

# Load model
model = load_model("model_cnn_lstm.h5")

# Tunjuk ringkasan struktur model
model.summary()