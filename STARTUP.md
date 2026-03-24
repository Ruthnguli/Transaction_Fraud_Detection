# FraudShield — Local Startup Guide

## Every time you start your PC

### Option A — Run with Docker (recommended)
Use this when you want the full production setup.

```bash
# 1. Navigate to project
cd /mnt/d/hakath/fraud_detect

# 2. Start the API (builds if needed, runs in background)
docker-compose up -d

# 3. Confirm it's running
curl http://127.0.0.1:5000/
# Expected: {"status": "Fraud Detection API is running"}

# 4. Open the dashboard
#    Double-click dashboard.html in your file explorer
#    Or open in browser: file:///mnt/d/hakath/fraud_detect/dashboard.html
```

**To stop:**
```bash
docker-compose down
```

---

### Option B — Run directly with Python (faster for development)
Use this when actively editing code.

```bash
# 1. Navigate to project
cd /mnt/d/hakath/fraud_detect

# 2. Activate conda environment
conda activate datascience

# 3. Start the API
python app.py

# 4. Open the dashboard
#    Double-click dashboard.html in your file explorer
```

**To stop:** `Ctrl+C` in the terminal

---

## If you need to retrain the model

```bash
cd /mnt/d/hakath/fraud_detect
conda activate datascience
python src/train_v2.py
```

---

## If you need to test the API manually

```bash
# Health check
curl http://127.0.0.1:5000/

# Legitimate transaction
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: acbfd54923b78bc76f23d57de58814ce7d2bcb33d6fda09d643958e4904682c7" \
  -d '{"step":1,"type":3,"amount":4878.0,"oldbalanceOrg":170136.0,"newbalanceOrig":165258.0,"oldbalanceDest":0.0,"newbalanceDest":4878.0,"isFlaggedFraud":0}'

# Fraud transaction (should return FRAUD at ~100%)
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: acbfd54923b78bc76f23d57de58814ce7d2bcb33d6fda09d643958e4904682c7" \
  -d '{"step":1,"type":4,"amount":450000.0,"oldbalanceOrg":450000.0,"newbalanceOrig":0.0,"oldbalanceDest":0.0,"newbalanceDest":0.0,"isFlaggedFraud":0}'
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `API_KEY not set` | Make sure `.env` exists in project root |
| `Port 5000 already in use` | `docker-compose down` then try again |
| `Model not found` | Run `python src/train_v2.py` to regenerate |
| Dashboard shows no data | Make sure API is running on port 5000 and CORS is enabled |
| Docker build fails | `docker system prune` then `docker-compose up --build` |

---

## Project structure reminder

```
fraud_detect/
├── app.py                          ← Flask REST API
├── dashboard.html                  ← FraudShield monitoring dashboard
├── Dockerfile                      ← Production container definition
├── docker-compose.yml              ← Local container orchestration
├── requirements.txt                ← Python dependencies
├── README.md                       ← Project documentation
├── STARTUP.md                      ← Local startup guide
├── TRANSACTION_GUIDE.md            ← Manual input reference
├── .env                            ← API key (gitignored)
├── .dockerignore                   
├── .gitignore                      
├── src/
│   ├── train.py                    ← V1 training script (baseline)
│   └── train_v2.py                 ← V2 training script (production)
├── models/
│   ├── fraud_model.ubj             ← Trained XGBoost model (xgboost native format)
│   └── scaler.pkl                  ← Fitted StandardScaler
├── notebooks/
│   ├── Fraud_Detection.ipynb       ← Main notebook (EDA → model → evaluation)
│   ├── Africatalking_Fraud_Detection.ipynb  ← Original exploration notebook
│   ├── model_improvement.ipynb     ← V1 vs V2 comparison work
│   └── testnote.ipynb              ← API testing notebook
├── plots/
│   ├── balance_error_distribution.png   ← Feature engineering analysis
│   └── model_comparison_v1_v2.png       ← V1 vs V2 comparison charts
└── data/                           ← gitignored — place CSV here
    └── Synthetic_Financial_datasets_log.csv
```
