import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon
from xgboost import XGBRegressor

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ── Chargement ────────────────────────────────────────────────────────────────
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
test       = pd.read_csv(save_path + "split_test.csv",
                         parse_dates=["order_purchase_timestamp"])
test_rag   = pd.read_csv(save_path + "m3_test_rag.csv")
zs_b       = pd.read_csv(save_path + "m4_test_zeroshot_optionB.csv")
hp12       = json.load(open(save_path + "hyperparams_m1_m2.json"))
hp3        = json.load(open(save_path + "hyperparams_m3.json"))

train_m3, _ = train_test_split(
    train_pool, train_size=1500,
    stratify=train_pool["churn"], random_state=42
)
train_m3 = train_m3.sort_values("order_purchase_timestamp").reset_index(drop=True)

# ── Features ──────────────────────────────────────────────────────────────────
struct_features = [
    "recency_days", "frequency", "monetary",
    "n_items", "n_sellers", "freight_ratio",
    "delivery_delay_days", "is_late", "processing_time_days",
    "n_installments", "payment_value",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    "product_weight_g"
]
rag_features       = ["livraison","qualite_produit","service_client","prix"]
mentioned_features = [f'{d}_mentioned' for d in rag_features]

# Target encoding M1/M2 sur review_score
state_means_12  = train_pool.groupby("customer_state")["review_score"].mean()
cat_means_12    = train_pool.groupby("product_category_name_english")["review_score"].mean()
train_median_12 = train_pool[struct_features].median()
score_mean_12   = train_pool["review_score"].mean()

# Target encoding M3 sur review_score
state_means_3   = train_m3.groupby("customer_state")["review_score"].mean()
cat_means_3     = train_m3.groupby("product_category_name_english")["review_score"].mean()
train_median_3  = train_m3[struct_features].median()
score_mean_3    = train_m3["review_score"].mean()

def apply_encoding(df, state_means, cat_means):
    df = df.copy()
    df["state_encoded"]    = df["customer_state"].map(state_means)
    df["category_encoded"] = df["product_category_name_english"].map(cat_means)
    return df

def impute(df, median, fill_mean):
    df = df.copy()
    for col in struct_features:
        df[col] = df[col].fillna(median[col])
    df["state_encoded"]    = df["state_encoded"].fillna(fill_mean)
    df["category_encoded"] = df["category_encoded"].fillna(fill_mean)
    return df

feats1 = struct_features + ["state_encoded", "category_encoded"]

train_12 = impute(apply_encoding(train_pool, state_means_12, cat_means_12),
                  train_median_12, score_mean_12)
test_12  = impute(apply_encoding(test,       state_means_12, cat_means_12),
                  train_median_12, score_mean_12)

X_train_12 = train_12[feats1].astype(float).values
X_test_12  = test_12[feats1].astype(float).values
y_train_12 = train_12["review_score"].values
y_test     = test["review_score"].values

# ── M1 XGB ────────────────────────────────────────────────────────────────────
p1 = hp12["M1_XGB"]
xgb1 = XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=p1["max_depth"], min_child_weight=p1["min_child_weight"],
    subsample=p1["subsample"], colsample_bytree=p1["colsample_bytree"],
    reg_alpha=p1["reg_alpha"], reg_lambda=p1["reg_lambda"],
    random_state=42, verbosity=0, nthread=1
)
xgb1.fit(X_train_12, y_train_12)
pred_m1 = xgb1.predict(X_test_12)
print(f"M1 XGB — MAE : {np.mean(np.abs(y_test - pred_m1)):.4f}")

# ── M2 Ridge ──────────────────────────────────────────────────────────────────
X_train_emb = np.load(save_path + "final_X_train_emb.npy")
X_test_emb  = np.load(save_path + "final_X_test_emb.npy")

sc2s = StandardScaler()
sc2e = StandardScaler()
X_tr2 = np.hstack([sc2s.fit_transform(X_train_12), sc2e.fit_transform(X_train_emb)])
X_te2 = np.hstack([sc2s.transform(X_test_12),      sc2e.transform(X_test_emb)])

ridge2 = Ridge(alpha=10.0)
ridge2.fit(X_tr2, y_train_12)
pred_m2 = ridge2.predict(X_te2)
print(f"M2 Ridge — MAE : {np.mean(np.abs(y_test - pred_m2)):.4f}")

# ── M3 XGB ────────────────────────────────────────────────────────────────────
train3 = impute(apply_encoding(train_m3, state_means_3, cat_means_3),
                train_median_3, score_mean_3)
test3  = impute(apply_encoding(test,     state_means_3, cat_means_3),
                train_median_3, score_mean_3)

train3 = train3.merge(
    pd.read_csv(save_path + "m3_train_rag_stratified.csv")[["order_id"] + rag_features],
    on="order_id", how="inner")
test3  = test3.merge(test_rag[["order_id"] + rag_features],
                     on="order_id", how="inner")

for dim in rag_features:
    train3[f'{dim}_mentioned'] = (train3[dim] != 0.5).astype(int)
    test3[f'{dim}_mentioned']  = (test3[dim]  != 0.5).astype(int)
for col in rag_features + mentioned_features:
    train3[col] = train3[col].fillna(0.5)
    test3[col]  = test3[col].fillna(0.5)

all_feats3  = feats1 + rag_features + mentioned_features
X_train3    = train3[all_feats3].astype(float).values
X_test3     = test3[all_feats3].astype(float).values
y_train3    = train3["review_score"].values
y_test3     = test3["review_score"].values

sc3         = StandardScaler()
X_train3_sc = sc3.fit_transform(X_train3)
X_test3_sc  = sc3.transform(X_test3)

p3 = hp3["M3_XGB"]
xgb3 = XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=p3["max_depth"], min_child_weight=p3["min_child_weight"],
    subsample=p3["subsample"], colsample_bytree=p3["colsample_bytree"],
    reg_alpha=p3["reg_alpha"], reg_lambda=p3["reg_lambda"],
    random_state=42, verbosity=0, nthread=1
)
xgb3.fit(X_train3_sc, y_train3)
pred_m3 = xgb3.predict(X_test3_sc)
print(f"M3 XGB — MAE : {np.mean(np.abs(y_test3 - pred_m3)):.4f}")

# ── M4 Zero-shot ──────────────────────────────────────────────────────────────
pred_m4  = zs_b["pred_score"].values
y_test_4 = zs_b["review_score"].values
print(f"M4 ZS  — MAE : {np.mean(np.abs(y_test_4 - pred_m4)):.4f}")

# ── Tests de Wilcoxon ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TESTS DE WILCOXON — Significativité des différences (Option B)")
print("="*60)

err_m1 = np.abs(y_test  - pred_m1)
err_m2 = np.abs(y_test  - pred_m2)
err_m3 = np.abs(y_test3 - pred_m3)
err_m4 = np.abs(y_test_4 - pred_m4)


def wilcoxon_test(err1, err2, label1, label2):
    n       = min(len(err1), len(err2))
    stat, p = wilcoxon(err1[:n], err2[:n])
    better  = label2 if err2[:n].mean() < err1[:n].mean() else label1
    print(f"  {label1} vs {label2}")
    print(f"    stat={stat:.1f} | p={p:.4f} | {better} meilleur")

wilcoxon_test(err_m1, err_m2, "M1 XGB", "M2 Ridge")
wilcoxon_test(err_m1, err_m3, "M1 XGB", "M3 XGB")
wilcoxon_test(err_m2, err_m3, "M2 Ridge", "M3 XGB")
wilcoxon_test(err_m1, err_m4, "M1 XGB", "M4 Zero-shot")
wilcoxon_test(err_m2, err_m4, "M2 Ridge", "M4 Zero-shot")