import json
import joblib
import pandas as pd

# Fungsi load feature columns
def load_feature_columns(filepath='feature_columns.json'):
    with open(filepath, 'r') as f:
        feature_columns = json.load(f)
    return feature_columns

# Feature extractor contoh (copy dari anda / sesuaikan)
def full_feature_extractor(window):
    features = {}
    # ... (copy code feature extraction anda di sini) ...
    # pastikan return dict feature nama sama dengan yang training
    return features

if __name__ == "__main__":
    # 1. Load feature columns
    feature_columns = load_feature_columns()

    # 2. Load model
    model = joblib.load('gait_classifier_rf.pkl')

    # 3. Ambil data baru sebagai pandas DataFrame window (contoh load csv untuk test)
    # Gantikan dengan data realtime anda nanti
    window = pd.read_csv('new_gait_window.csv')

    # 4. Extract features dari window
    features_new = full_feature_extractor(window)

    # 5. Buat dataframe untuk model predict
    df_new = pd.DataFrame([features_new])
    df_new = df_new.reindex(columns=feature_columns, fill_value=0)  # pastikan kolum sama

    # 6. Predict label
    pred_label = model.predict(df_new)[0]

    print("Predicted gait class:", pred_label)