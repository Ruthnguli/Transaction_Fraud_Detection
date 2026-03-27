"""
app.py — Fraud Detection API
Supports numeric transaction types and Kenyan mobile money transaction names.
"""
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from functools import wraps
import time
import africastalking


# ─── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not set. Add it to your .env file.")

# ─── Africa's Talking Setup ───────────────────────────────────────────────────
AT_USERNAME  = os.getenv("AT_USERNAME")
AT_API_KEY   = os.getenv("AT_API_KEY")
ALERT_PHONE  = os.getenv("ALERT_PHONE")

africastalking.initialize(AT_USERNAME, AT_API_KEY)
sms = africastalking.SMS

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "fraud_model.ubj")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
START_TIME = time.time()
MODEL_VERSION = "3.0"

# ─── Load model & scaler ──────────────────────────────────────────────────────
try:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler: {e}")

# ─── Transaction type mapper ───────────────────────────────────────────────────
# Maps Kenyan mobile money transaction names → model type codes
# Model codes: 0=CASH_IN, 1=CASH_OUT, 2=DEBIT, 3=PAYMENT, 4=TRANSFER
TRANSACTION_TYPE_MAP = {
    # ── M-Pesa ────────────────────────────────────────────────────────────────
    "send_money":               4,  # TRANSFER  — person to person
    "lipa_na_mpesa_till":       3,  # PAYMENT   — buy goods (till number)
    "lipa_na_mpesa_paybill":    3,  # PAYMENT   — pay bill (utility, rent)
    "pochi_la_biashara":        3,  # PAYMENT   — small business wallet
    "withdraw_agent":           1,  # CASH_OUT  — agent withdrawal
    "deposit_agent":            0,  # CASH_IN   — agent deposit
    "mpesa_global":             4,  # TRANSFER  — international send
    "reverse_transaction":      0,  # CASH_IN   — reversal (money back)
    "fuliza":                   2,  # DEBIT     — overdraft/loan product

    # ── Airtel Money ──────────────────────────────────────────────────────────
    "airtel_send":              4,  # TRANSFER
    "airtel_withdraw":          1,  # CASH_OUT
    "airtel_deposit":           0,  # CASH_IN
    "airtel_pay":               3,  # PAYMENT

    # ── T-Kash (Telkom) ───────────────────────────────────────────────────────
    "tkash_send":               4,  # TRANSFER
    "tkash_withdraw":           1,  # CASH_OUT
    "tkash_pay":                3,  # PAYMENT

    # ── Equitel / Equity ──────────────────────────────────────────────────────
    "equitel_send":             4,  # TRANSFER
    "equitel_withdraw":         1,  # CASH_OUT
    "eazzy_pay":                3,  # PAYMENT

    # ── Generic fallbacks ─────────────────────────────────────────────────────
    "transfer":                 4,
    "payment":                  3,
    "cash_out":                 1,
    "cash_in":                  0,
    "debit":                    2,

    # ── Numeric strings (in case sent as "0", "1", etc.) ─────────────────────
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
}

VALID_TYPE_CODES = {0, 1, 2, 3, 4}

def resolve_type(raw_type):
    """
    Accepts:
      - int/float: 0-4 (model native codes)
      - str: Kenyan mobile money transaction name or numeric string
    Returns:
      - int type code (0-4)
    Raises:
      - ValueError if unrecognised
    """
    if isinstance(raw_type, (int, float)):
        code = int(raw_type)
        if code in VALID_TYPE_CODES:
            return code
        raise ValueError(f"Numeric type must be 0-4, got {raw_type}")

    if isinstance(raw_type, str):
        key = raw_type.strip().lower()
        if key in TRANSACTION_TYPE_MAP:
            return TRANSACTION_TYPE_MAP[key]
        raise ValueError(
            f"Unknown transaction type '{raw_type}'. "
            f"Accepted names: {sorted(TRANSACTION_TYPE_MAP.keys())}"
        )

    raise ValueError(f"type must be a number (0-4) or a transaction name string, got {type(raw_type)}")

# ─── Config ───────────────────────────────────────────────────────────────────
RAW_FIELDS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud"
]

FEATURE_COLUMNS = RAW_FIELDS + ["balance_error_orig", "balance_error_dest", "amount_to_balance_ratio"]

NUM_FEATURES = [
    "amount", "oldbalanceOrg", "oldbalanceDest", "step",
    "balance_error_orig", "balance_error_dest", "amount_to_balance_ratio"
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

def send_fraud_alert(amount, tx_type, probability):
    """Send SMS alert to account holder when fraud is detected."""
    try:
        message = (
            f"[FraudShield Alert] SUSPICIOUS TRANSACTION DETECTED\n"
            f"Type    : {tx_type.upper()}\n"
            f"Amount  : KES {amount:,.2f}\n"
            f"Risk    : {probability*100:.1f}%\n"
            f"Action  : Contact your bank immediately if you did not initiate this."
        )
        response = sms.send(message, [ALERT_PHONE])
        print(f"  SMS alert sent: {response}")
        return True
    except Exception as e:
        print(f"  SMS alert failed: {e}")
        return False

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"status": "Fraud Detection API is running"})

@app.route("/health", methods=["GET"])
def health():
    uptime_seconds = int(time.time() - START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return jsonify({
        "status"         : "healthy",
        "model_version"  : MODEL_VERSION,
        "threshold"      : FRAUD_THRESHOLD,
        "features"       : FEATURE_COLUMNS,
        "uptime"         : f"{hours}h {minutes}m {seconds}s",
        "supported_types": list(TRANSACTION_TYPE_MAP.keys())
    })

@app.route("/transaction-types", methods=["GET"])
def transaction_types():
    """Returns all supported Kenyan mobile money transaction type names."""
    grouped = {
        "mpesa":   ["send_money", "lipa_na_mpesa_till", "lipa_na_mpesa_paybill",
                    "pochi_la_biashara", "withdraw_agent", "deposit_agent",
                    "mpesa_global", "reverse_transaction", "fuliza"],
        "airtel":  ["airtel_send", "airtel_withdraw", "airtel_deposit", "airtel_pay"],
        "tkash":   ["tkash_send", "tkash_withdraw", "tkash_pay"],
        "equitel": ["equitel_send", "equitel_withdraw", "eazzy_pay"],
        "generic": ["transfer", "payment", "cash_out", "cash_in", "debit"],
        "numeric": [0, 1, 2, 3, 4]
    }
    return jsonify({
        "supported_transaction_types": grouped,
        "model_codes": {
            "0": "CASH_IN",
            "1": "CASH_OUT",
            "2": "DEBIT",
            "3": "PAYMENT",
            "4": "TRANSFER"
        }
    })


@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    records = [data] if isinstance(data, dict) else data

    # Validate required fields
    missing = [f for f in RAW_FIELDS if f not in records[0]]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {missing}",
            "required_fields": RAW_FIELDS
        }), 400

    try:
        df = pd.DataFrame(records)[RAW_FIELDS].copy()

        # Resolve transaction type — accepts name or numeric code
        resolved_types = []
        for _, row in df.iterrows():
            try:
                resolved_types.append(resolve_type(row["type"]))
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        df["type"] = resolved_types

        # Engineer features server-side
        df["balance_error_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
        df["balance_error_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
        df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

        # Scale
        df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])

        # Predict
        probability = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        prediction  = (probability >= FRAUD_THRESHOLD).astype(int)
        
        # Resolve original type name for response
        raw_type = records[0]["type"]
        type_name = raw_type if isinstance(raw_type, str) else {
            0:"CASH_IN", 1:"CASH_OUT", 2:"DEBIT", 3:"PAYMENT", 4:"TRANSFER"
        }.get(int(raw_type), str(raw_type))

        probability = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        prediction  = (probability >= FRAUD_THRESHOLD).astype(int)

        # Send SMS alert if fraud detected
        sms_sent = False
        if prediction[0] == 1 and ALERT_PHONE:
            sms_sent = send_fraud_alert(
                amount    = records[0]["amount"],
                tx_type   = records[0]["type"] if isinstance(records[0]["type"], str) else str(records[0]["type"]),
                probability = float(probability[0])
            )

        raw_type = records[0]["type"]
        type_name = raw_type if isinstance(raw_type, str) else {
            0:"CASH_IN",1:"CASH_OUT",2:"DEBIT",3:"PAYMENT",4:"TRANSFER"
        }.get(int(raw_type), str(raw_type))

        return jsonify({
            "prediction"        : int(prediction[0]),
            "fraud_probability" : round(float(probability[0]), 4),
            "label"             : "FRAUD" if prediction[0] == 1 else "LEGITIMATE",
            "threshold_used"    : FRAUD_THRESHOLD,
            "transaction_type"  : type_name,
            "type_code"         : resolved_types[0],
            "sms_alert_sent"    : sms_sent
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)