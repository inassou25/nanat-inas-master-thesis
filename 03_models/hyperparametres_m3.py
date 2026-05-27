import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import faiss
import json
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from tqdm import tqdm

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# BLOC 1 — CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
val        = pd.read_csv(save_path + "split_val.csv",
                         parse_dates=["order_purchase_timestamp"])

# 1500 premières observations chronologiques du train pool
train = train_pool.iloc[:1500].copy()

print(f"Train M3 : {len(train)} obs | churn : {train['churn'].mean():.1%}")
print(f"Val      : {len(val)} obs | churn : {val['churn'].mean():.1%}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 2 — FEATURES STRUCTURÉES
# ════════════════════════════════════════════════════════════════════════════
struct_features = [
    "recency_days", "frequency", "monetary",
    "n_items", "n_sellers", "freight_ratio",
    "delivery_delay_days", "is_late", "processing_time_days",
    "n_installments", "payment_value",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    "product_weight_g"
]

# Target encoding — calculé sur train uniquement
state_means      = train.groupby("customer_state")["churn"].mean()
cat_means        = train.groupby("product_category_name_english")["churn"].mean()
train_median     = train[struct_features].median()
train_churn_mean = train["churn"].mean()

for df in [train, val]:
    df["state_encoded"]    = df["customer_state"].map(state_means)
    df["category_encoded"] = df["product_category_name_english"].map(cat_means)

feats1         = struct_features + ["state_encoded", "category_encoded"]
rag_features   = ["livraison","qualite_produit","service_client","prix"]
mentioned_features = [f'{d}_mentioned' for d in rag_features]
all_feats3     = feats1 + rag_features + mentioned_features

def impute_struct(df):
    df = df.copy()
    for col in struct_features:
        df[col] = df[col].fillna(train_median[col])
    df["state_encoded"]    = df["state_encoded"].fillna(train_churn_mean)
    df["category_encoded"] = df["category_encoded"].fillna(train_churn_mean)
    return df
# ════════════════════════════════════════════════════════════════════════════
# BLOC 5 — CHARGEMENT SCORES RAG DÉJÀ CALCULÉS
# ════════════════════════════════════════════════════════════════════════════
train_rag = pd.read_csv(save_path + "m3_train_rag.csv")
val_rag   = pd.read_csv(save_path + "m3_val_rag.csv")
print(f"Train RAG : {len(train_rag)} obs")
print(f"Val RAG   : {len(val_rag)} obs")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 6 — CONSTRUCTION DES FEATURES M3
# ════════════════════════════════════════════════════════════════════════════
def build_features(df, rag_df):
    merged = df.merge(rag_df[["order_id"] + rag_features],
                      on="order_id", how="inner")
    for dim in rag_features:
        merged[f'{dim}_mentioned'] = (merged[dim] != 0.5).astype(int)
    merged = impute_struct(merged)
    for col in rag_features + mentioned_features:
        merged[col] = merged[col].fillna(0.5)
    return merged

train3 = build_features(train, train_rag)
val3   = build_features(val,   val_rag)

X_train3 = train3[all_feats3].astype(float).values
X_val3   = val3[all_feats3].astype(float).values
y_train3 = train3["churn"].values
y_val3   = val3["churn"].values

print(f"\nTrain3 : {len(train3)} obs | Val3 : {len(val3)} obs")
print(f"NaN train : {np.isnan(X_train3).sum()} | NaN val : {np.isnan(X_val3).sum()}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 7 — GRID SEARCH
# ════════════════════════════════════════════════════════════════════════════
param_grid_lr = [
    {"C": 1.0},
    {"C": 0.1},
    {"C": 0.01},
]

param_grid_xgb = [
    {"max_depth": 3, "min_child_weight": 5,  "reg_lambda": 2.0, "reg_alpha": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 3, "min_child_weight": 10, "reg_lambda": 5.0, "reg_alpha": 1.0, "subsample": 0.7, "colsample_bytree": 0.7},
    {"max_depth": 4, "min_child_weight": 5,  "reg_lambda": 2.0, "reg_alpha": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "min_child_weight": 10, "reg_lambda": 5.0, "reg_alpha": 1.0, "subsample": 0.7, "colsample_bytree": 0.7},
]

spw = (y_train3==0).sum() / (y_train3==1).sum()

sc3         = StandardScaler()
X_train3_sc = sc3.fit_transform(X_train3)
X_val3_sc   = sc3.transform(X_val3)

def overfitting_check(y_tr, p_tr, y_v, p_v, label):
    gap = average_precision_score(y_tr, p_tr) - average_precision_score(y_v, p_v)
    status = "⚠️ overfitting" if gap > 0.05 else "✅ ok"
    print(f"  [{label}] Train : {average_precision_score(y_tr, p_tr):.4f} | "
          f"Val : {average_precision_score(y_v, p_v):.4f} | Gap : {gap:.4f} {status}")

# LR
print("\n" + "="*60)
print("M3 — GRID SEARCH LR")
print("="*60)
best_lr_score, best_lr_params, best_lr = -1, None, None
for params in param_grid_lr:
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                             random_state=42, C=params["C"])
    lr.fit(X_train3_sc, y_train3)
    score = average_precision_score(y_val3, lr.predict_proba(X_val3_sc)[:, 1])
    print(f"  C={params['C']} → val PR-AUC : {score:.4f}")
    if score > best_lr_score:
        best_lr_score, best_lr_params, best_lr = score, params, lr
print(f"  → Meilleur : {best_lr_params} (val PR-AUC : {best_lr_score:.4f})")
overfitting_check(y_train3, best_lr.predict_proba(X_train3_sc)[:,1],
                  y_val3,   best_lr.predict_proba(X_val3_sc)[:,1], "M3 LR")

# XGB — nthread=1 pour éviter le segfault Python 3.13
print("\n" + "="*60)
print("M3 — GRID SEARCH XGB")
print("="*60)
best_xgb_score, best_xgb_params, best_xgb = -1, None, None
for params in param_grid_xgb:
    xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=params["max_depth"],
    min_child_weight=params["min_child_weight"],
    subsample=params["subsample"],
    colsample_bytree=params["colsample_bytree"],
    reg_alpha=params["reg_alpha"],
    reg_lambda=params["reg_lambda"],
    scale_pos_weight=spw,
    eval_metric="aucpr",
    random_state=42,
    verbosity=0,
    nthread=1  # ← crucial pour Python 3.13 ARM64
)
    xgb.fit(X_train3_sc, y_train3)
    score = average_precision_score(y_val3, xgb.predict_proba(X_val3_sc)[:, 1])
    print(f"  depth={params['max_depth']} mcw={params['min_child_weight']} "
          f"λ={params['reg_lambda']} → val PR-AUC : {score:.4f}")
    if score > best_xgb_score:
        best_xgb_score, best_xgb_params, best_xgb = score, params, xgb
print(f"  → Meilleur : {best_xgb_params} (val PR-AUC : {best_xgb_score:.4f})")
overfitting_check(y_train3, best_xgb.predict_proba(X_train3_sc)[:,1],
                  y_val3,   best_xgb.predict_proba(X_val3_sc)[:,1], "M3 XGB")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 8 — RÉCAPITULATIF + SAUVEGARDE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RÉCAPITULATIF VAL — M3")
print("="*60)
print(f"M3 LR  : PR-AUC = {best_lr_score:.4f} | "
      f"ROC-AUC = {roc_auc_score(y_val3, best_lr.predict_proba(X_val3_sc)[:,1]):.4f} | "
      f"params : {best_lr_params}")
print(f"M3 XGB : PR-AUC = {best_xgb_score:.4f} | "
      f"ROC-AUC = {roc_auc_score(y_val3, best_xgb.predict_proba(X_val3_sc)[:,1]):.4f} | "
      f"params : {best_xgb_params}")

hyperparams_m3 = {
    "M3_LR":  best_lr_params,
    "M3_XGB": best_xgb_params,
    "sc3_mean": sc3.mean_.tolist(),
    "sc3_std":  sc3.scale_.tolist(),
}
with open(save_path + "hyperparams_m3.json", "w") as f:
    json.dump(hyperparams_m3, f, indent=2)
print("\n✅ Hyperparamètres sauvegardés : hyperparams_m3.json")