import json
import joblib
import pandas as pd
import blynklib
import time

# Blynk Auth Token kau
BLYNK_AUTH = 's6twEQqusbfwEcrQYdID2avZ00GTA_r8'
blynk = None

# Fungsi load feature columns
def load_feature_columns(filepath='feature_columns.json'):
    with open(filepath, 'r') as f:
        feature_columns = json.load(f)
    return feature_columns

# Feature extractor contoh (copy dari anda / sesuaikan)
def full_feature_extractor(window):
    features = {}
    # Versi kedua: contoh extract mean accel (ikut code asal kau)
    features['mean_accelX'] = window['accelX'].mean()
    features['mean_accelY'] = window['accelY'].mean()
    features['mean_accelZ'] = window['accelZ'].mean()
    # Kalau ada code lain, tambah kat sini
    return features

def send_classification_status(status):
    global blynk
    print(f"Sending status to Blynk: {status}")
    if blynk:
        try:
            blynk.virtual_write(13, status)
        except Exception as e:
            print(f"Error sending data: {e}")

def connect_blynk():
    global blynk
    while True:
        try:
            print("Trying to connect to Blynk...")
            blynk = blynklib.Blynk(BLYNK_AUTH)
            blynk.run()  # test run sekali connect
            print("Connected to Blynk!")
            break
        except Exception as e:
            print(f"Connection failed: {e}")
            time.sleep(5)

def main():
    # Load feature columns (dari file JSON)
    feature_columns = load_feature_columns()

    # Load model ML
    model = joblib.load('gait_classifier_rf.pkl')

    # Load data baru dari CSV
    window = pd.read_csv('new_gait_window.csv')

    # Extract features guna fungsi kau
    features_new = full_feature_extractor(window)

    # Siapkan dataframe ikut column training
    df_new = pd.DataFrame([features_new])
    df_new = df_new.reindex(columns=feature_columns, fill_value=0)

    # Predict class label
    pred_label = model.predict(df_new)[0]

    print("Predicted gait class:", pred_label)

    # Connect ke Blynk dan hantar status
    connect_blynk()
    send_classification_status(pred_label)

    # Maintain connection dengan auto reconnect bila lost
    while True:
        try:
            blynk.run()
        except Exception as e:
            print(f"Blynk disconnected: {e}")
            connect_blynk()
            send_classification_status(pred_label)
        time.sleep(0.1)

# Betulkan statement main supaya Python faham
if __name__ == "main":
    main()