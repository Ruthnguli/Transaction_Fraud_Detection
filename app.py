"""
app.py — Fraud Detection API
"""
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from functools import wraps

# ─── Environment ──────────────────────────────────────────────────────────────
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

# ─── Config ───────────────────────────────────────────────────────────────────
# Fields the API caller must provide (engineered features computed server-side)
RAW_FIELDS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud"
]

# All features the model expects (including engineered ones)
FEATURE_COLUMNS = RAW_FIELDS + ["balance_error_orig", "balance_error_dest"]

# Numerical features to scale (must match train_v2.py)
NUM_FEATURES = [
    "amount", "oldbalanceOrg", "oldbalanceDest", "step",
    "balance_error_orig", "balance_error_dest"
]

FRAUD_THRESHOLD = 0.6835

# ─── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─── Auth decorator ───────────────────────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key or key != API_KEY:
            return jsonify({
                "error": "Unauthorized. Provide a valid API key in the X-API-Key header."
            }), 401
        return f(*args, **kwargs)
    return decorated

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"status": "Fraud Detection API is running"})

@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Normalize to list
    records = [data] if isinstance(data, dict) else data

    # Validate raw input fields only — engineered features are computed here
    missing = [f for f in RAW_FIELDS if f not in records[0]]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {missing}",
            "required_fields": RAW_FIELDS
        }), 400

    try:
        df = pd.DataFrame(records)[RAW_FIELDS]

        # Engineer features server-side — same logic as train_v2.py
        df["balance_error_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
        df["balance_error_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

        # Scale numerical features
        df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])

        # Predict using tuned threshold
        probability = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
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