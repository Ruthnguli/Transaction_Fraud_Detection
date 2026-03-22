"""
src/train.py
Fraud Detection Model Training Script
--------------------------------------
Trains an XGBClassifier on the Synthetic Financial dataset,
saves the model (.ubj) and scaler (.pkl) to the models/ directory.

Usage:
    python src/train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Synthetic_Financial_datasets_log.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.ubj")      # xgboost native format
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")          # scaler must be saved too

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ─── 2. Clean ─────────────────────────────────────────────────────────────────
df = df.dropna()
df = df.drop(columns=["nameOrig", "nameDest"])          # high-cardinality IDs, no signal
df["type"] = df["type"].astype("category").cat.codes    # encode transaction type

# ─── 3. Scale numerical features ──────────────────────────────────────────────
NUM_FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest", "step"]
scaler = StandardScaler()
df[NUM_FEATURES] = scaler.fit_transform(df[NUM_FEATURES])

# Save scaler — CRITICAL: needed at inference time
joblib.dump(scaler, SCALER_PATH)
print(f"  Scaler saved → {SCALER_PATH}")

# ─── 4. Split ─────────────────────────────────────────────────────────────────
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

print(f"\nFeatures used for training: {list(X.columns)}")
print(f"Class distribution:\n{y.value_counts(normalize=True).round(4)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── 5. Handle Imbalance with SMOTE ───────────────────────────────────────────
print("\nApplying SMOTE...")
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: {y_train_sm.value_counts().to_dict()}")

# ─── 6. Cross-Validate ────────────────────────────────────────────────────────
print("\nRunning cross-validation...")
model = XGBClassifier(
    subsample=1.0,
    reg_lambda=1,
    reg_alpha=0.1,
    n_estimators=300,
    max_depth=7,
    learning_rate=0.2,
    gamma=0.5,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"          # moved here from constructor in xgb 3.x
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring="roc_auc")
print(f"  CV AUC:  mean={np.mean(cv_scores):.4f}  std={np.std(cv_scores):.4f}")

# ─── 7. Final Fit on full training set ────────────────────────────────────────
# NOTE: This is the step that was missing in the original notebook.
# cross_val_score fits temporary clones — it does NOT fit the model object itself.
print("\nFitting final model on full training data...")
model.fit(X_train_sm, y_train_sm)
print("  Model fitted ✓")

# ─── 8. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── Evaluation on Test Set ──")
print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 9. Save Model (xgboost native format — version-safe) ─────────────────────
model.save_model(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# ─── 10. Verify the saved model loads and predicts correctly ──────────────────
print("\nVerifying saved model...")
verify_model = XGBClassifier()
verify_model.load_model(MODEL_PATH)

sample = X_test.iloc[:1]
pred   = verify_model.predict(sample)
prob   = verify_model.predict_proba(sample)[:, 1]
print(f"  Sample prediction : {pred[0]}  (fraud probability: {prob[0]:.4f})")
print("  Verification passed ✓")

print("\n✅ Training complete.")
print(f"   Model  → {MODEL_PATH}")
print(f"   Scaler → {SCALER_PATH}")
print(f"   Features → {list(X.columns)}")
