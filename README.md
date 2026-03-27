# 🛡️ FraudShield — Kenya Mobile Money Fraud Detection

A production-grade AI fraud detection system built for the **Africa's Talking Open Hackathon — Cybersecurity Solutions**. FraudShield detects fraudulent mobile money transactions in real time and instantly alerts account holders via SMS using the Africa's Talking API.

Trained on 6.3M transactions, the V3 model achieves **ROC-AUC 0.9998** with **98.4% fraud precision** and sends SMS alerts within milliseconds of detection.

---

## 🎯 Hackathon Context

**Event:** Africa's Talking Open Hackathon — CyberSecurity Solutions, Nairobi  
**Challenge:** Financial Security and Fraud Prevention  
**Solution:** AI-driven anomaly detection + real-time SMS alerts via Africa's Talking APIs

### What we built
- Real-time fraud detection API supporting M-Pesa, Airtel Money, T-Kash and Equitel transactions
- Instant SMS alert to account holder when fraud is detected (Africa's Talking SMS API)
- Live monitoring dashboard for fraud analysts
- Dockerized, production-ready deployment

---

## 📁 Project Structure

```
fraud_detect/
├── app.py                          ← Flask REST API with AT SMS integration
├── dashboard.html                  ← FraudShield monitoring dashboard
├── Dockerfile                      ← Production container definition
├── docker-compose.yml              ← Local container orchestration
├── requirements.txt                ← Python dependencies
├── README.md                       ← Project documentation
├── STARTUP.md                      ← Local startup guide
├── TRANSACTION_GUIDE.md            ← Manual input reference
├── .env                            ← Secrets (gitignored)
├── src/
│   ├── train.py                    ← V1 training script (baseline)
│   ├── train_v2.py                 ← V2 training script (balance error features)
│   └── train_v3.py                 ← V3 training script (production)
├── models/
│   ├── fraud_model.ubj             ← Trained XGBoost V3 model
│   └── scaler.pkl                  ← Fitted StandardScaler
├── notebooks/
│   ├── Fraud_Detection.ipynb       ← Main notebook (EDA → model → evaluation)
│   ├── testnote.ipynb              ← V3 feature exploration notebook
│   └── model_improvement.ipynb     ← V1 vs V2 vs V3 comparison
├── plots/
│   ├── balance_error_distribution.png
│   ├── model_comparison_v1_v2.png
│   └── model_comparison_v2_v3.png
└── data/                           ← gitignored — place CSV here
    └── Synthetic_Financial_datasets_log.csv
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fraud_detect.git
cd fraud_detect
```

### 2. Create and activate environment
```bash
conda activate datascience
# or
python -m venv venv && source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```dotenv
API_KEY=your_secret_api_key
AT_USERNAME=sandbox
AT_API_KEY=your_africastalking_api_key
ALERT_PHONE=+2547XXXXXXXXX
```

### 5. Add the dataset
Download from Kaggle and place at:
```
data/Synthetic_Financial_datasets_log.csv
```

### 6. Train the model
```bash
python src/train_v3.py
```

---

## 🚀 Running the API

### Option A — Python (development)
```bash
python app.py
```

### Option B — Docker (production)
```bash
docker-compose up -d
```

API available at `http://localhost:5000`

---

## 📡 API Reference

### `GET /`
Health check.
```json
{ "status": "Fraud Detection API is running" }
```

### `GET /health`
Model status and version info.
```json
{
  "status": "healthy",
  "model_version": "3.0",
  "threshold": 0.6835,
  "uptime": "0h 5m 32s",
  "features": ["step", "type", "amount", "..."],
  "supported_types": ["send_money", "lipa_na_mpesa_till", "..."]
}
```

### `GET /transaction-types`
Returns all supported Kenyan mobile money transaction names grouped by provider.

### `POST /predict`
Analyze a transaction for fraud. Requires `X-API-Key` header.

**Request:**
```json
{
  "step": 1,
  "type": "send_money",
  "amount": 450000.0,
  "oldbalanceOrg": 450000.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0,
  "isFlaggedFraud": 0
}
```

**Supported `type` values:**

| Provider | Transaction names |
|---|---|
| M-Pesa | `send_money`, `lipa_na_mpesa_till`, `lipa_na_mpesa_paybill`, `pochi_la_biashara`, `withdraw_agent`, `deposit_agent`, `mpesa_global`, `fuliza` |
| Airtel Money | `airtel_send`, `airtel_pay`, `airtel_withdraw`, `airtel_deposit` |
| T-Kash | `tkash_send`, `tkash_pay`, `tkash_withdraw` |
| Equitel | `equitel_send`, `eazzy_pay`, `equitel_withdraw` |
| Numeric | `0` (CASH_IN), `1` (CASH_OUT), `2` (DEBIT), `3` (PAYMENT), `4` (TRANSFER) |

**Response:**
```json
{
  "prediction": 1,
  "fraud_probability": 1.0,
  "label": "FRAUD",
  "threshold_used": 0.6835,
  "transaction_type": "send_money",
  "type_code": 4,
  "sms_alert_sent": true
}
```

When `prediction == 1`, an SMS alert is automatically sent to the registered `ALERT_PHONE` via Africa's Talking SMS API.

---

## 📱 SMS Alert

When fraud is detected, the account holder receives an instant SMS:

```
[FraudShield Alert] SUSPICIOUS TRANSACTION DETECTED
Type    : SEND_MONEY
Amount  : KES 450,000.00
Risk    : 100.0%
Action  : Contact your bank immediately if you did not initiate this.
```

This uses the **Africa's Talking SMS API**. Switch from sandbox to live by updating `AT_USERNAME` and `AT_API_KEY` in `.env`.

---

## 🧠 Model Details

### Version History

| Version | Key Change | Fraud F1 | Precision | Missed Cases |
|---|---|---|---|---|
| V1 | Baseline XGBoost | 75% | 61% | — |
| V2 | + balance error features | 93% | 87% | 20 |
| V3 | + amount-to-balance ratio | **99%** | **98%** | **10** |

### V3 Features

| Feature | Description |
|---|---|
| `step` | Hour of transaction (1–744) |
| `type` | Transaction type code (0–4) |
| `amount` | Transaction amount in KES |
| `oldbalanceOrg` | Sender balance before |
| `newbalanceOrig` | Sender balance after |
| `oldbalanceDest` | Receiver balance before |
| `newbalanceDest` | Receiver balance after |
| `isFlaggedFraud` | System fraud flag |
| `balance_error_orig` | `oldbalanceOrg - newbalanceOrig - amount` — detects sender anomalies |
| `balance_error_dest` | `newbalanceDest - oldbalanceDest - amount` — detects layering |
| `amount_to_balance_ratio` | `amount / (oldbalanceOrg + 1)` — detects full account drains |

### Fraud Signatures Detected

**Type A — Money laundering (layering):**
```
newbalanceOrig = 0    ← sender fully drained
newbalanceDest = 0    ← money vanished (moved again immediately)
```

**Type B — Straight theft:**
```
newbalanceOrig = 0           ← sender fully drained
amount ≈ oldbalanceOrg       ← entire balance sent
newbalanceDest > 0           ← money arrived at destination
```

### Training Pipeline
1. Drop missing values and identifier columns
2. Encode transaction type as category codes
3. Engineer 3 derived features
4. Scale numerical features with `StandardScaler`
5. SMOTE oversampling (sampling_strategy=0.2) for class imbalance
6. XGBoost with optimized hyperparameters
7. Threshold tuning via precision-recall curve (optimal: 0.6835)

### Final Metrics (V3)
| Metric | Value |
|---|---|
| ROC-AUC | 0.9998 |
| Fraud Precision | 98.4% |
| Fraud Recall | 99.4% |
| Fraud F1 | 98.9% |
| False Negatives | 10 / 1,643 fraud cases in test set |

---

## 🐳 Docker

```bash
# Build
docker build -t fraudshield-api .

# Run
docker-compose up -d

# Check logs
docker-compose logs -f
```

---

## 🌍 Africa's Talking Integration

Current integration: **SMS fraud alerts**

Planned integrations:
- **USSD** — transaction verification via `*384#` (no smartphone needed)
- **Voice** — automated call for high-risk fraud (probability > 95%)
- **SMS OTP** — confirm high-value transactions before processing

---

## 📦 Dependencies

```
flask
flask-cors
pandas
scikit-learn
xgboost
joblib
imbalanced-learn
python-dotenv
gunicorn
africastalking
```

---

## 📄 License

MIT