from utility import load_dataset
import joblib
import os

X, y = load_dataset()

os.makedirs("model", exist_ok=True)
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("Feature names saved successfully.")