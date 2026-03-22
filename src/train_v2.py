"""
src/train_v2.py
Fraud Detection — Improved Model Training
------------------------------------------
Adds engineered features (balance_error_orig, balance_error_dest),
compares old vs new model, saves improved model and scaler.

Usage:
    python src/train_v2.py
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
NEW_MODEL   = os.path.join(MODEL_DIR, "fraud_model_v2.ubj")
NEW_SCALER  = os.path.join(MODEL_DIR, "scaler_v2.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
COLORS = {"fraud": "#ff4757", "legit": "#00ff88", "neutral": "#6b7280", "accent": "#3b82f6"}

# ─── 1. Load & Clean ──────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — Loading data")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
df = df.dropna().drop(columns=["nameOrig", "nameDest"])
df["type"] = df["type"].astype("category").cat.codes
print(f"  Rows       : {len(df):,}")
print(f"  Fraud cases: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.3f}%)")

# ─── 2. Engineer Features ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 — Feature engineering")
print("=" * 60)

df["balance_error_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
df["balance_error_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

print("  New features added:")
print("  balance_error_orig = oldbalanceOrg - newbalanceOrig - amount")
print("  balance_error_dest = newbalanceDest - oldbalanceDest - amount")

fraud    = df[df["isFraud"] == 1]
legit    = df[df["isFraud"] == 0].sample(5000, random_state=42)

print(f"\n  balance_error_orig (fraud mean): {fraud['balance_error_orig'].mean():,.2f}")
print(f"  balance_error_orig (legit mean): {legit['balance_error_orig'].mean():,.2f}")
print(f"  balance_error_dest (fraud mean): {fraud['balance_error_dest'].mean():,.2f}")
print(f"  balance_error_dest (legit mean): {legit['balance_error_dest'].mean():,.2f}")

# Plot: balance error distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Balance Error Distribution — Fraud vs Legitimate", fontsize=13, color="white")

for ax, col, title in zip(axes,
    ["balance_error_orig", "balance_error_dest"],
    ["Origin account error", "Destination account error"]):
    ax.hist(np.clip(legit[col], -1e5, 1e5), bins=60, alpha=0.6, color=COLORS["legit"], label="Legit", density=True)
    ax.hist(np.clip(fraud[col], -1e5, 1e5), bins=60, alpha=0.6, color=COLORS["fraud"], label="Fraud", density=True)
    ax.set_title(title, color="white")
    ax.set_xlabel("Error value (clipped)", color=COLORS["neutral"])
    ax.legend()
    ax.tick_params(colors=COLORS["neutral"])

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "balance_error_distribution.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved → {plot_path}")

# ─── 3. Split & Scale ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3 — Split and scale")
print("=" * 60)

FEATURES_V1 = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]
FEATURES_V2 = FEATURES_V1 + ["balance_error_orig", "balance_error_dest"]
NUM_COLS_V2 = ["amount", "oldbalanceOrg", "oldbalanceDest", "step",
               "balance_error_orig", "balance_error_dest"]

X = df[FEATURES_V2]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler_v2 = StandardScaler()
X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()
X_train_sc[NUM_COLS_V2] = scaler_v2.fit_transform(X_train[NUM_COLS_V2])
X_test_sc[NUM_COLS_V2]  = scaler_v2.transform(X_test[NUM_COLS_V2])

joblib.dump(scaler_v2, NEW_SCALER)
print(f"  Scaler v2 saved → {NEW_SCALER}")

smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc[FEATURES_V2], y_train)
print(f"  After SMOTE: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# ─── 4. Train New Model ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4 — Training improved model")
print("=" * 60)

model_v2 = XGBClassifier(
    subsample=1.0, reg_lambda=1, reg_alpha=0.1,
    n_estimators=300, max_depth=7, learning_rate=0.2,
    gamma=0.5, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_v2, X_train_sm, y_train_sm, cv=cv, scoring="roc_auc")
print(f"  CV AUC: mean={np.mean(cv_scores):.4f}  std={np.std(cv_scores):.4f}")

model_v2.fit(X_train_sm, y_train_sm)
print("  Model fitted ✓")

model_v2.save_model(NEW_MODEL)
print(f"  Model v2 saved → {NEW_MODEL}")

# ─── 5. Compare Old vs New ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5 — Old vs New comparison")
print("=" * 60)

# Load old model
old_model  = XGBClassifier()
old_model.load_model(OLD_MODEL)
old_scaler = joblib.load(OLD_SCALER)

NUM_COLS_V1 = ["amount", "oldbalanceOrg", "oldbalanceDest", "step"]
X_test_v1   = X_test[FEATURES_V1].copy()
X_test_v1[NUM_COLS_V1] = old_scaler.transform(X_test_v1[NUM_COLS_V1])

y_prob_v1 = old_model.predict_proba(X_test_v1)[:, 1]
y_prob_v2 = model_v2.predict_proba(X_test_sc[FEATURES_V2])[:, 1]
y_pred_v1 = (y_prob_v1 >= 0.6835).astype(int)
y_pred_v2 = (y_prob_v2 >= 0.6835).astype(int)

auc_v1 = roc_auc_score(y_test, y_prob_v1)
auc_v2 = roc_auc_score(y_test, y_prob_v2)

print(f"\n  {'Metric':<25} {'V1 (old)':>12} {'V2 (new)':>12} {'Change':>10}")
print(f"  {'-'*60}")
print(f"  {'ROC-AUC':<25} {auc_v1:>12.4f} {auc_v2:>12.4f} {auc_v2-auc_v1:>+10.4f}")

rep_v1 = classification_report(y_test, y_pred_v1, output_dict=True)
rep_v2 = classification_report(y_test, y_pred_v2, output_dict=True)

for metric in ["precision", "recall", "f1-score"]:
    v1 = rep_v1["1"][metric]
    v2 = rep_v2["1"][metric]
    label = f"Fraud {metric}"
    print(f"  {label:<25} {v1:>12.4f} {v2:>12.4f} {v2-v1:>+10.4f}")

# ─── 6. Comparison plots ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Model Comparison — V1 vs V2", fontsize=14, color="white", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ROC curves
ax1 = fig.add_subplot(gs[0, 0])
RocCurveDisplay.from_predictions(y_test, y_prob_v1, ax=ax1, name=f"V1 (AUC={auc_v1:.4f})", color=COLORS["neutral"])
RocCurveDisplay.from_predictions(y_test, y_prob_v2, ax=ax1, name=f"V2 (AUC={auc_v2:.4f})", color=COLORS["accent"])
ax1.set_title("ROC Curve", color="white")
ax1.tick_params(colors=COLORS["neutral"])

# PR curves
ax2 = fig.add_subplot(gs[0, 1])
PrecisionRecallDisplay.from_predictions(y_test, y_prob_v1, ax=ax2, name="V1", color=COLORS["neutral"])
PrecisionRecallDisplay.from_predictions(y_test, y_prob_v2, ax=ax2, name="V2", color=COLORS["accent"])
ax2.set_title("Precision-Recall Curve", color="white")
ax2.tick_params(colors=COLORS["neutral"])

# Bar comparison
ax3 = fig.add_subplot(gs[0, 2])
metrics = ["Precision", "Recall", "F1"]
v1_vals = [rep_v1["1"]["precision"], rep_v1["1"]["recall"], rep_v1["1"]["f1-score"]]
v2_vals = [rep_v2["1"]["precision"], rep_v2["1"]["recall"], rep_v2["1"]["f1-score"]]
x = np.arange(len(metrics))
ax3.bar(x - 0.2, v1_vals, 0.35, label="V1", color=COLORS["neutral"], alpha=0.8)
ax3.bar(x + 0.2, v2_vals, 0.35, label="V2", color=COLORS["accent"], alpha=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(metrics)
ax3.set_ylim(0, 1.1); ax3.set_title("Fraud class metrics", color="white")
ax3.legend(); ax3.tick_params(colors=COLORS["neutral"])

# Confusion matrices
for ax, y_pred, title in zip(
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    [y_pred_v1, y_pred_v2],
    ["Confusion Matrix V1", "Confusion Matrix V2"]
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=",", cmap="RdYlGn", ax=ax,
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    ax.set_title(title, color="white")
    ax.tick_params(colors=COLORS["neutral"])

# Feature importance
ax5 = fig.add_subplot(gs[1, 2])
importance = pd.Series(model_v2.feature_importances_, index=FEATURES_V2).sort_values()
colors = [COLORS["accent"] if "error" in f else COLORS["neutral"] for f in importance.index]
importance.plot(kind="barh", ax=ax5, color=colors)
ax5.set_title("Feature importance (V2)", color="white")
ax5.tick_params(colors=COLORS["neutral"])

plot_path2 = os.path.join(PLOTS_DIR, "model_comparison_v1_v2.png")
plt.savefig(plot_path2, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Comparison plot saved → {plot_path2}")

# ─── 7. Promote V2 to production ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6 — Promote V2 to production")
print("=" * 60)

import shutil
shutil.copy(NEW_MODEL,  os.path.join(MODEL_DIR, "fraud_model.ubj"))
shutil.copy(NEW_SCALER, os.path.join(MODEL_DIR, "scaler.pkl"))
print("  fraud_model.ubj → updated to V2")
print("  scaler.pkl      → updated to V2")
print(f"\n  New FEATURE_COLUMNS for app.py:")
print(f"  {FEATURES_V2}")

print("\n✅ Training v2 complete.")
print(f"   Fraud Precision : {rep_v2['1']['precision']:.4f}  (was {rep_v1['1']['precision']:.4f})")
print(f"   Fraud Recall    : {rep_v2['1']['recall']:.4f}  (was {rep_v1['1']['recall']:.4f})")
print(f"   Fraud F1        : {rep_v2['1']['f1-score']:.4f}  (was {rep_v1['1']['f1-score']:.4f})")
print(f"   ROC-AUC         : {auc_v2:.4f}  (was {auc_v1:.4f})")
