import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# ====== 1. LOAD DATA ======
X = np.load('C:/Users/hairi/OneDrive/IDP/monai_project/X.npy')
y = np.load('C:/Users/hairi/OneDrive/IDP/monai_project/y.npy')

# One-hot encoding
y_cat = to_categorical(y)

# Split data (train, val, test)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

# ====== 2. BINA MODEL CNN-LSTM ======
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=X.shape[1:]),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')  # output class
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ====== 3. LATIH MODEL ======
history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_val, y_val)
)

# ====== 4. SIMPAN MODEL & DATA TEST ======
model.save('C:/Users/hairi/OneDrive/IDP/monai_project/model_cnn_lstm.h5')
np.save('C:/Users/hairi/OneDrive/IDP/monai_project/X_test.npy', X_test)
np.save('C:/Users/hairi/OneDrive/IDP/monai_project/y_test.npy', y_test)

print("✅ Model & test data disimpan!")

# ====== 5. UJI MODEL ======
model = load_model('C:/Users/hairi/OneDrive/IDP/monai_project/model_cnn_lstm.h5')
X_test = np.load('C:/Users/hairi/OneDrive/IDP/monai_project/X_test.npy')
y_test = np.load('C:/Users/hairi/OneDrive/IDP/monai_project/y_test.npy')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Report
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Normal", "Abnormal"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()