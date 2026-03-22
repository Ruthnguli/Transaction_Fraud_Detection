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
from dotenv import load_dotenv
from functools import wraps



load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not set. Add it to your .env file.")

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

# Numerical features that need scaling (must match train.py)
NUM_FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest", "step"]

# ─── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key or key != API_KEY:
            return jsonify({"error": "Unauthorized. Provide a valid API key in the X-API-Key header."}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/")
def home():
    return jsonify({"status": "Fraud Detection API is running"})

@app.route("/predict", methods=["POST"])
@require_api_key
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

        prediction  = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        return jsonify({
            "prediction": int(prediction[0]),
            "fraud_probability": round(float(probability[0]), 4),
            "label": "FRAUD" if prediction[0] == 1 else "LEGITIMATE"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)