"""
app.py — Fraud Detection API
"""
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "fraud_model.ubj")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ─── Load model & scaler at startup ───────────────────────────────────────────
try:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler: {e}")

# Features the model was trained on (in exact order)
FEATURE_COLUMNS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud"
]

# Optimal decision threshold (tuned via precision-recall curve)
# Default 0.5 gives precision=0.91, recall=0.95
# 0.6835 gives precision=0.96, recall=0.94 — better balance for production
FRAUD_THRESHOLD = 0.6835

# Numerical features that need scaling (must match train.py)
NUM_FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest", "step"]

# ─── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"status": "Fraud Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Normalize to list of records
    records = [data] if isinstance(data, dict) else data

    # Validate required fields
    missing = [f for f in FEATURE_COLUMNS if f not in records[0]]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}",
                        "required_fields": FEATURE_COLUMNS}), 400

    try:
        df = pd.DataFrame(records)[FEATURE_COLUMNS]

        # Apply the same scaling used during training
        df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])

        probability = model.predict_proba(df)[:, 1]
        prediction  = (probability >= FRAUD_THRESHOLD).astype(int)

        return jsonify({
            "prediction": int(prediction[0]),
            "fraud_probability": round(float(probability[0]), 4),
            "label": "FRAUD" if prediction[0] == 1 else "LEGITIMATE",
            "threshold_used": FRAUD_THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)
