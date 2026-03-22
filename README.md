# 🔍 Fraud Detection API

A machine learning API for detecting fraudulent financial transactions in real time, built with XGBoost and Flask.

Trained on the [Synthetic Financial Datasets](https://www.kaggle.com/datasets/ealaxi/paysim1) (6.3M transactions), the model achieves a **ROC-AUC of 0.9995** with a **98% fraud recall**.

---

## 📁 Project Structure

```
fraud_detect/
├── app.py                  # Flask REST API
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── src/
│   └── train.py            # Model training script
├── models/
│   ├── fraud_model.ubj     # Trained XGBoost model (xgboost native format)
│   └── scaler.pkl          # Fitted StandardScaler
├── notebooks/
│   ├── Africatalking_Fraud_Detection.ipynb   # EDA & model exploration
│   └── testnote.ipynb      # API testing notebook
└── data/                   # (gitignored) place your CSV here
    └── Synthetic_Financial_datasets_log.csv
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fraud_detect.git
cd fraud_detect
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the dataset from Kaggle and place it at:
```
data/Synthetic_Financial_datasets_log.csv
```

### 5. Train the model
```bash
python src/train.py
```
This will create `models/fraud_model.ubj` and `models/scaler.pkl`.

---

## 🚀 Running the API

```bash
python app.py
```

The API will be available at `http://localhost:5000`.

To enable debug mode:
```bash
FLASK_DEBUG=true python app.py
```

---

## 📡 API Reference

### `GET /`
Health check.

**Response:**
```json
{ "status": "Fraud Detection API is running" }
```

---

### `POST /predict`
Predict whether a transaction is fraudulent.

**Request body:**
```json
{
  "step": 1,
  "type": 4,
  "amount": 500000.0,
  "oldbalanceOrg": 500000.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 500000.0,
  "isFlaggedFraud": 0
}
```

| Field | Type | Description |
|---|---|---|
| `step` | int | Time step (1 step = 1 hour) |
| `type` | int | Transaction type encoded: CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4 |
| `amount` | float | Transaction amount |
| `oldbalanceOrg` | float | Sender's balance before transaction |
| `newbalanceOrig` | float | Sender's balance after transaction |
| `oldbalanceDest` | float | Receiver's balance before transaction |
| `newbalanceDest` | float | Receiver's balance after transaction |
| `isFlaggedFraud` | int | System flag for large transfers (0 or 1) |

**Response:**
```json
{
  "prediction": 1,
  "fraud_probability": 0.9243,
  "label": "FRAUD"
}
```

| Field | Description |
|---|---|
| `prediction` | `0` = Legitimate, `1` = Fraud |
| `fraud_probability` | Model confidence score (0.0 – 1.0) |
| `label` | Human-readable result: `LEGITIMATE` or `FRAUD` |

---

## 🐳 Running with Docker

```bash
# Build the image
docker build -t fraud-detection-api .

# Run the container
docker run -p 5000:5000 fraud-detection-api
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Imbalance handling | SMOTE (sampling_strategy=0.2) |
| Hyperparameter tuning | RandomizedSearchCV (20 iterations, 5-fold CV) |
| ROC-AUC Score | 0.9995 |
| Fraud Recall | 98% |
| False Negatives | 31 / 1,643 fraud cases in test set |
| Training data | 6.3M transactions |
| Model format | XGBoost native `.ubj` (version-safe) |

### Key Features Used
- Transaction type, amount, step
- Origin and destination account balances (before and after)
- System fraud flag

### Training Pipeline
1. Drop missing values and identifier columns (`nameOrig`, `nameDest`)
2. Encode `type` as category codes
3. Scale numerical features with `StandardScaler`
4. SMOTE oversampling on training set to handle class imbalance (0.13% fraud rate)
5. Train XGBClassifier with optimized hyperparameters
6. Evaluate on held-out 20% test set

---

## 📦 Dependencies

```
flask
pandas
scikit-learn
xgboost
joblib
imbalanced-learn
```

---

## 📄 License

MIT
