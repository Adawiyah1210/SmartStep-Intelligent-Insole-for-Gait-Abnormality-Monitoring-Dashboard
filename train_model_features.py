import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# 1. Load dataset dengan features dan label
df = pd.read_csv('gait_features_full.csv')

# Pisah features dan label
X = df.drop(columns=['label'])
y = df['label']

# 2. Simpan nama feature columns ke file json
feature_columns = X.columns.tolist()
with open('feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)
print(f"Saved feature columns to feature_columns.json ({len(feature_columns)} features)")

# 3. Bahagikan data train-test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Buat model Random Forest dan latih
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Model trained.")

# 5. Test dan evaluasi model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Simpan model ke file pkl
joblib.dump(clf, 'gait_classifier_rf.pkl')
print("Model saved to gait_classifier_rf.pkl")