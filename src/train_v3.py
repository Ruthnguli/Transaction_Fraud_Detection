"""
src/train_v3.py
Fraud Detection — V3 Model Training
-------------------------------------
Adds amount_to_balance_ratio feature on top of V2.
This single feature cuts missed fraud cases by 50%.

New feature:
    amount_to_balance_ratio = amount / (oldbalanceOrg + 1)

Signal: 98% of fraud has ratio >= 0.99 (sender drains entire balance).
        Legitimate transactions have a median ratio of 6.51.

Improvements over V2:
    Fraud Precision : 87% → 98%
    Fraud F1        : 93% → 99%
    Missed cases    : 20  → 10
    ROC-AUC         : 0.9996 → 0.9998

Usage:
    python src/train_v3.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "Synthetic_Financial_datasets_log.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
OLD_MODEL   = os.path.join(MODEL_DIR, "fraud_model.ubj")
OLD_SCALER  = os.path.join(MODEL_DIR, "scaler.pkl")
NEW_MODEL   = os.path.join(MODEL_DIR, "fraud_model_v3.ubj")
NEW_SCALER  = os.path.join(MODEL_DIR, "scaler_v3.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use("dark_background")
COLORS = {"fraud":"#ff4757","legit":"#00ff88","neutral":"#6b7280","accent":"#3b82f6"}

# ─── 1. Load & Clean ──────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — Loading data")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
df = df.dropna().drop(columns=["nameOrig","nameDest"])
df["type"] = df["type"].astype("category").cat.codes
print(f"  Rows       : {len(df):,}")
print(f"  Fraud cases: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.3f}%)")

# ─── 2. Engineer Features ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 — Feature engineering")
print("=" * 60)

# V2 features
df["balance_error_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
df["balance_error_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

# V3 new feature
df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

fraud = df[df["isFraud"]==1]
legit = df[df["isFraud"]==0]

print(f"  amount_to_balance_ratio — fraud median : {fraud['amount_to_balance_ratio'].median():.4f}")
print(f"  amount_to_balance_ratio — legit median : {legit['amount_to_balance_ratio'].median():.4f}")
print(f"  Fraud with ratio >= 0.99 : {(fraud['amount_to_balance_ratio']>=0.99).mean()*100:.1f}%")
print(f"  Legit with ratio >= 0.99 : {(legit['amount_to_balance_ratio']>=0.99).mean()*100:.1f}%")

# ─── 3. Split & Scale ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3 — Split and scale")
print("=" * 60)

FEATURES_V2 = ["step","type","amount","oldbalanceOrg","newbalanceOrig",
               "oldbalanceDest","newbalanceDest","isFlaggedFraud",
               "balance_error_orig","balance_error_dest"]
FEATURES_V3 = FEATURES_V2 + ["amount_to_balance_ratio"]
NUM_COLS_V3 = ["amount","oldbalanceOrg","oldbalanceDest","step",
               "balance_error_orig","balance_error_dest","amount_to_balance_ratio"]

X = df[FEATURES_V3]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler_v3 = StandardScaler()
X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()
X_train_sc[NUM_COLS_V3] = scaler_v3.fit_transform(X_train[NUM_COLS_V3])
X_test_sc[NUM_COLS_V3]  = scaler_v3.transform(X_test[NUM_COLS_V3])

joblib.dump(scaler_v3, NEW_SCALER)
print(f"  Scaler v3 saved → {NEW_SCALER}")

smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc[FEATURES_V3], y_train)
print(f"  After SMOTE: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# ─── 4. Train V3 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4 — Training V3 model")
print("=" * 60)

model_v3 = XGBClassifier(
    subsample=1.0, reg_lambda=1, reg_alpha=0.1,
    n_estimators=300, max_depth=7, learning_rate=0.2,
    gamma=0.5, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_v3, X_train_sm, y_train_sm, cv=cv, scoring="roc_auc")
print(f"  CV AUC: mean={np.mean(cv_scores):.4f}  std={np.std(cv_scores):.4f}")

model_v3.fit(X_train_sm, y_train_sm)
print("  Model fitted ✓")

model_v3.save_model(NEW_MODEL)
print(f"  Model v3 saved → {NEW_MODEL}")

# ─── 5. Compare V2 vs V3 ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5 — V2 vs V3 comparison")
print("=" * 60)

old_model  = XGBClassifier()
old_model.load_model(OLD_MODEL)
old_scaler = joblib.load(OLD_SCALER)

NUM_COLS_V2 = ["amount","oldbalanceOrg","oldbalanceDest","step",
               "balance_error_orig","balance_error_dest"]
X_test_v2 = X_test[FEATURES_V2].copy()
X_test_v2[NUM_COLS_V2] = old_scaler.transform(X_test_v2[NUM_COLS_V2])

y_prob_v2 = old_model.predict_proba(X_test_v2)[:,1]
y_prob_v3 = model_v3.predict_proba(X_test_sc[FEATURES_V3])[:,1]
y_pred_v2 = (y_prob_v2 >= 0.6835).astype(int)
y_pred_v3 = (y_prob_v3 >= 0.6835).astype(int)

auc_v2 = roc_auc_score(y_test, y_prob_v2)
auc_v3 = roc_auc_score(y_test, y_prob_v3)
rep_v2 = classification_report(y_test, y_pred_v2, output_dict=True)
rep_v3 = classification_report(y_test, y_pred_v3, output_dict=True)

print(f"\n  {'Metric':<25} {'V2 (current)':>14} {'V3 (new)':>10} {'Change':>10}")
print(f"  {'-'*62}")
print(f"  {'ROC-AUC':<25} {auc_v2:>14.4f} {auc_v3:>10.4f} {auc_v3-auc_v2:>+10.4f}")
for metric in ["precision","recall","f1-score"]:
    v2 = rep_v2["1"][metric]
    v3 = rep_v3["1"][metric]
    print(f"  {'Fraud '+metric:<25} {v2:>14.4f} {v3:>10.4f} {v3-v2:>+10.4f}")

# Subtype breakdown
meta = X_test.copy()
meta["isFraud"]   = y_test.values
meta["pred_v2"]   = y_pred_v2
meta["pred_v3"]   = y_pred_v3
fraud_test = meta[meta["isFraud"]==1]
type_a = fraud_test[fraud_test["newbalanceDest"]==0]
type_b = fraud_test[fraud_test["newbalanceDest"]!=0]

print(f"\n  Type A (layering) — V2: {type_a['pred_v2'].sum()}/{len(type_a)}  V3: {type_a['pred_v3'].sum()}/{len(type_a)}")
print(f"  Type B (theft)    — V2: {type_b['pred_v2'].sum()}/{len(type_b)}  V3: {type_b['pred_v3'].sum()}/{len(type_b)}")
print(f"  Total missed      — V2: {(fraud_test['pred_v2']==0).sum()}  V3: {(fraud_test['pred_v3']==0).sum()}")

# ─── 6. Comparison plots ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Model Comparison — V2 vs V3", fontsize=14, color="white", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
RocCurveDisplay.from_predictions(y_test, y_prob_v2, ax=ax1,
    name=f"V2 (AUC={auc_v2:.4f})", color=COLORS["neutral"])
RocCurveDisplay.from_predictions(y_test, y_prob_v3, ax=ax1,
    name=f"V3 (AUC={auc_v3:.4f})", color=COLORS["accent"])
ax1.set_title("ROC Curve", color="white")

ax2 = fig.add_subplot(gs[0,1])
PrecisionRecallDisplay.from_predictions(y_test, y_prob_v2, ax=ax2,
    name="V2", color=COLORS["neutral"])
PrecisionRecallDisplay.from_predictions(y_test, y_prob_v3, ax=ax2,
    name="V3", color=COLORS["accent"])
ax2.set_title("Precision-Recall Curve", color="white")

ax3 = fig.add_subplot(gs[0,2])
metrics = ["Precision","Recall","F1"]
v2_vals = [rep_v2["1"]["precision"],rep_v2["1"]["recall"],rep_v2["1"]["f1-score"]]
v3_vals = [rep_v3["1"]["precision"],rep_v3["1"]["recall"],rep_v3["1"]["f1-score"]]
x = np.arange(len(metrics))
ax3.bar(x-0.2, v2_vals, 0.35, label="V2", color=COLORS["neutral"], alpha=0.8)
ax3.bar(x+0.2, v3_vals, 0.35, label="V3", color=COLORS["accent"], alpha=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(metrics)
ax3.set_ylim(0,1.1); ax3.set_title("Fraud class metrics", color="white")
ax3.legend()

for ax, y_pred, title in zip(
    [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])],
    [y_pred_v2, y_pred_v3],
    ["Confusion Matrix — V2","Confusion Matrix — V3"]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=",", cmap="RdYlGn", ax=ax,
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    ax.set_title(title, color="white")

ax5 = fig.add_subplot(gs[1,2])
importance = pd.Series(model_v3.feature_importances_, index=FEATURES_V3).sort_values()
colors = [COLORS["fraud"] if f=="amount_to_balance_ratio"
          else COLORS["accent"] if "error" in f
          else COLORS["neutral"] for f in importance.index]
importance.plot(kind="barh", ax=ax5, color=colors)
ax5.set_title("Feature importance V3\n(red=new, blue=V2 engineered)", color="white")

plot_path = os.path.join(PLOTS_DIR, "model_comparison_v2_v3.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Comparison plot saved → {plot_path}")

# ─── 7. Promote V3 to production ─────────────────────────────────────────────
import shutil
print("\n" + "=" * 60)
print("PHASE 6 — Promote V3 to production")
print("=" * 60)

shutil.copy(NEW_MODEL,  os.path.join(MODEL_DIR, "fraud_model.ubj"))
shutil.copy(NEW_SCALER, os.path.join(MODEL_DIR, "scaler.pkl"))
print("  fraud_model.ubj → updated to V3")
print("  scaler.pkl      → updated to V3")

print(f"\n  Update app.py FEATURES and NUM_FEATURES to include:")
print(f"  + 'amount_to_balance_ratio'")
print(f"\n  And add this line in the predict route after building df:")
print(f"  df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)")

print(f"\n✅ Training v3 complete.")
print(f"   Fraud Precision : {rep_v3['1']['precision']:.4f}  (was {rep_v2['1']['precision']:.4f})")
print(f"   Fraud Recall    : {rep_v3['1']['recall']:.4f}  (was {rep_v2['1']['recall']:.4f})")
print(f"   Fraud F1        : {rep_v3['1']['f1-score']:.4f}  (was {rep_v2['1']['f1-score']:.4f})")
print(f"   ROC-AUC         : {auc_v3:.4f}  (was {auc_v2:.4f})")
print(f"   Missed cases    : {(fraud_test['pred_v3']==0).sum()}  (was {(fraud_test['pred_v2']==0).sum()})")
