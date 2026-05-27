import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.utils import resample
from xgboost import XGBClassifier

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ── Chargement ────────────────────────────────────────────────────────────────
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
test       = pd.read_csv(save_path + "split_test.csv",
                         parse_dates=["order_purchase_timestamp"])
hp12       = json.load(open(save_path + "hyperparams_m1_m2.json"))

# Même 1500 obs que M3
train_m2_1500, _ = train_test_split(
    train_pool, train_size=1500,
    stratify=train_pool["churn"], random_state=42
)
train_m2_1500 = train_m2_1500.sort_values(
    "order_purchase_timestamp").reset_index(drop=True)

print(f"Train M2 1500 : {len(train_m2_1500)} obs | churn : {train_m2_1500['churn'].mean():.3f}")

# ── Features structurées ──────────────────────────────────────────────────────
struct_features = [
    "recency_days", "frequency", "monetary",
    "n_items", "n_sellers", "freight_ratio",
    "delivery_delay_days", "is_late", "processing_time_days",
    "n_installments", "payment_value",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    "product_weight_g"
]

state_means  = train_m2_1500.groupby("customer_state")["churn"].mean()
cat_means    = train_m2_1500.groupby("product_category_name_english")["churn"].mean()
train_median = train_m2_1500[struct_features].median()
churn_mean   = train_m2_1500["churn"].mean()

def apply_encoding(df, sm, cm):
    df = df.copy()
    df["state_encoded"]    = df["customer_state"].map(sm).fillna(churn_mean)
    df["category_encoded"] = df["product_category_name_english"].map(cm).fillna(churn_mean)
    return df

def impute(df):
    df = df.copy()
    for col in struct_features:
        df[col] = df[col].fillna(train_median[col])
    return df

feats1 = struct_features + ["state_encoded", "category_encoded"]

train_enc = impute(apply_encoding(train_m2_1500, state_means, cat_means))
test_enc  = impute(apply_encoding(test,          state_means, cat_means))

X_train_s = train_enc[feats1].astype(float).values
X_test_s  = test_enc[feats1].astype(float).values
y_train   = train_m2_1500["churn"].values
y_test    = test["churn"].values

# ── Embeddings ────────────────────────────────────────────────────────────────
X_train_emb_full = np.load(save_path + "final_X_train_emb.npy")
X_test_emb       = np.load(save_path + "final_X_test_emb.npy")

train_pool_ids = pd.read_csv(save_path + "split_train_pool.csv")
idx_1500 = train_pool_ids[
    train_pool_ids["order_id"].isin(train_m2_1500["order_id"])
].index.tolist()

X_train_emb_1500 = X_train_emb_full[idx_1500]
print(f"Embeddings train 1500 : {X_train_emb_1500.shape}")

# ── Normalisation ─────────────────────────────────────────────────────────────
sc_s = StandardScaler()
sc_e = StandardScaler()
X_tr = np.hstack([sc_s.fit_transform(X_train_s), sc_e.fit_transform(X_train_emb_1500)])
X_te = np.hstack([sc_s.transform(X_test_s),      sc_e.transform(X_test_emb)])

spw = (y_train==0).sum() / (y_train==1).sum()

# ── Fonction évaluation ───────────────────────────────────────────────────────
def evaluate(y_true, y_proba, label):
    pr_auc  = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    thresholds = json.load(open(save_path + "optimal_thresholds.json"))
    seuil = thresholds.get("M2_LR", {}).get("threshold", 0.5)
    y_pred = (y_proba >= seuil).astype(int)
    f1 = f1_score(y_true, y_pred)

    scores = []
    for _ in range(1000):
        idx = resample(range(len(y_true)))
        try:
            scores.append(average_precision_score(
                np.array(y_true)[idx], np.array(y_proba)[idx]))
        except:
            pass
    ci = np.std(scores) * 1.96

    print(f"\n=== {label} ===")
    print(f"PR-AUC  : {pr_auc:.4f} ± {ci:.3f}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"F1      : {f1:.4f} (seuil : {seuil:.3f})")
    return {"label": label, "pr_auc": pr_auc, "roc_auc": roc_auc,
            "f1": f1, "ci": ci, "y_proba": y_proba}

results = []

# ── M2 LR sur 1500 ────────────────────────────────────────────────────────────
p = hp12["M2_LR"]
lr_1500 = LogisticRegression(C=p["C"], class_weight="balanced",
                              max_iter=1000, random_state=42)
lr_1500.fit(X_tr, y_train)
probas_lr_1500 = lr_1500.predict_proba(X_te)[:,1]
res = evaluate(y_test, probas_lr_1500, "M2 LR — 1500 obs")
results.append(res)

# ── M2 XGB sur 1500 ───────────────────────────────────────────────────────────
p = hp12["M2_XGB"]
xgb_1500 = XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
    subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
    reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
    scale_pos_weight=spw,
    eval_metric="aucpr", random_state=42, verbosity=0, nthread=1
)
xgb_1500.fit(X_tr, y_train)
probas_xgb_1500 = xgb_1500.predict_proba(X_te)[:,1]
res = evaluate(y_test, probas_xgb_1500, "M2 XGB — 1500 obs")
results.append(res)

# ── Sauvegarde des probabilités ───────────────────────────────────────────────
pd.DataFrame({
    "churn":        y_test,
    "M2_LR_1500":  probas_lr_1500,
    "M2_XGB_1500": probas_xgb_1500
}).to_csv(save_path + "test_probas_m2_1500.csv", index=False)
print("✅ Probabilités M2 1500 sauvegardées")

# ── Test de permutation M2 XGB 1500 vs M3 XGB ────────────────────────────────
print("\n" + "="*60)
print("TEST DE PERMUTATION — M2 XGB 1500 vs M3 XGB 1500")
print("="*60)

probas_m3 = pd.read_csv(save_path + "test_probas_m3.csv")
y_m3      = probas_m3["churn"].values
p_m3      = probas_m3["M3_XGB"].values

pr_m2 = average_precision_score(y_test, probas_xgb_1500)
pr_m3 = average_precision_score(y_m3,   p_m3)
delta = pr_m3 - pr_m2

count = 0
n     = min(len(y_test), len(y_m3))
for _ in range(1000):
    perm = np.random.permutation(n)
    pr_a = average_precision_score(y_test[:n], probas_xgb_1500[perm])
    pr_b = average_precision_score(y_m3[:n],   p_m3[perm])
    if (pr_b - pr_a) >= delta:
        count += 1

p_val = count / 1000
print(f"M2 XGB 1500 PR-AUC : {pr_m2:.4f}")
print(f"M3 XGB      PR-AUC : {pr_m3:.4f}")
print(f"Δ M3-M2 = {delta:+.4f} | p = {p_val:.3f}")

# ── Tableau comparatif ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TABLEAU COMPARATIF — Taille fixe 1500 observations")
print("="*60)
print(f"{'Méthode':<25} {'PR-AUC':>8} {'IC 95%':>8} {'ROC-AUC':>8} {'F1':>8}")
print("-" * 60)
for r in results:
    ci_str = f"±{r['ci']:.3f}"
    print(f"{r['label']:<25} {r['pr_auc']:>8.4f} {ci_str:>8} "
          f"{r['roc_auc']:>8.4f} {r['f1']:>8.4f}")

print(f"{'M3 XGB — 1500 obs':<25} {pr_m3:>8.4f} {'±0.024':>8} {'0.9552':>8} {'0.8080':>8}")

# ── Sauvegarde métriques ──────────────────────────────────────────────────────
pd.DataFrame([{k:v for k,v in r.items() if k != "y_proba"}
              for r in results]).to_csv(
    save_path + "m2_1500_results.csv", index=False)
print("\n✅ Résultats sauvegardés : m2_1500_results.csv")