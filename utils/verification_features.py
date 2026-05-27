import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ── M1 XGB ────────────────────────────────────────────────────────────────────
train_pool = pd.read_csv(save_path + "split_train_pool.csv")
hp12       = json.load(open(save_path + "hyperparams_m1_m2.json"))

struct_features = [
    "recency_days", "frequency", "monetary",
    "n_items", "n_sellers", "freight_ratio",
    "delivery_delay_days", "is_late", "processing_time_days",
    "n_installments", "payment_value",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    "product_weight_g"
]

state_means = train_pool.groupby("customer_state")["churn"].mean()
cat_means   = train_pool.groupby("product_category_name_english")["churn"].mean()
train_pool["state_encoded"]    = train_pool["customer_state"].map(state_means)
train_pool["category_encoded"] = train_pool["product_category_name_english"].map(cat_means)

feats1   = struct_features + ["state_encoded", "category_encoded"]
X_train  = train_pool[feats1].fillna(train_pool[feats1].median()).astype(float).values
y_train  = train_pool["churn"].values
spw      = (y_train==0).sum() / (y_train==1).sum()

p = hp12["M1_XGB"]
xgb1 = XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
    subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
    reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
    scale_pos_weight=spw,
    eval_metric="aucpr", random_state=42, verbosity=0, nthread=1
)
xgb1.fit(X_train, y_train)

print("=== Feature Importance M1 XGB (top 5) ===")
fi = sorted(zip(feats1, xgb1.feature_importances_), key=lambda x: -x[1])
for feat, imp in fi[:5]:
    print(f"  {feat:<30} : {imp:.4f}")

# ── M3 XGB ────────────────────────────────────────────────────────────────────
hp3        = json.load(open(save_path + "hyperparams_m3.json"))
train_rag  = pd.read_csv(save_path + "m3_train_rag_stratified.csv")

train_m3, _ = train_test_split(
    train_pool, train_size=1500,
    stratify=train_pool["churn"], random_state=42
)
train_m3 = train_m3.sort_values("order_purchase_timestamp").reset_index(drop=True)

state_means_3 = train_m3.groupby("customer_state")["churn"].mean()
cat_means_3   = train_m3.groupby("product_category_name_english")["churn"].mean()
train_m3["state_encoded"]    = train_m3["customer_state"].map(state_means_3)
train_m3["category_encoded"] = train_m3["product_category_name_english"].map(cat_means_3)

rag_features       = ["livraison","qualite_produit","service_client","prix"]
mentioned_features = [f'{d}_mentioned' for d in rag_features]
all_feats3         = feats1 + rag_features + mentioned_features

train3 = train_m3.merge(train_rag[["order_id"] + rag_features], on="order_id", how="inner")
for dim in rag_features:
    train3[f'{dim}_mentioned'] = (train3[dim] != 0.5).astype(int)
train3 = train3.fillna(0.5)

X_train3 = train3[all_feats3].astype(float).values
y_train3 = train3["churn"].values
spw3     = (y_train3==0).sum() / (y_train3==1).sum()

sc3         = StandardScaler()
X_train3_sc = sc3.fit_transform(X_train3)

p3 = hp3["M3_XGB"]
xgb3 = XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=p3["max_depth"], min_child_weight=p3["min_child_weight"],
    subsample=p3["subsample"], colsample_bytree=p3["colsample_bytree"],
    reg_alpha=p3["reg_alpha"], reg_lambda=p3["reg_lambda"],
    scale_pos_weight=spw3,
    eval_metric="aucpr", random_state=42, verbosity=0, nthread=1
)
xgb3.fit(X_train3_sc, y_train3)

print("\n=== Feature Importance M3 XGB (top 5) ===")
fi3 = sorted(zip(all_feats3, xgb3.feature_importances_), key=lambda x: -x[1])
for feat, imp in fi3[:5]:
    print(f"  {feat:<30} : {imp:.4f}")