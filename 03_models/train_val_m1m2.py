import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# BLOC 1 — CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════
train = pd.read_csv(save_path + "split_train_pool.csv",
                    parse_dates=["order_purchase_timestamp"])
val   = pd.read_csv(save_path + "split_val.csv",
                    parse_dates=["order_purchase_timestamp"])

print(f"Train : {len(train)} obs | churn : {train['churn'].mean():.1%}")
print(f"Val   : {len(val)} obs | churn : {val['churn'].mean():.1%}")

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

# Target encoding — calculé sur train uniquement, appliqué sur val
state_means = train.groupby("customer_state")["churn"].mean()
cat_means   = train.groupby("product_category_name_english")["churn"].mean()
for df in [train, val]:
    df["state_encoded"]    = df["customer_state"].map(state_means)
    df["category_encoded"] = df["product_category_name_english"].map(cat_means)

feats1       = struct_features + ["state_encoded", "category_encoded"]
train_median = train[struct_features].median()
train_churn_mean = train["churn"].mean()

def impute(df):
    df = df.copy()
    for col in struct_features:
        df[col] = df[col].fillna(train_median[col])
    df["state_encoded"]    = df["state_encoded"].fillna(train_churn_mean)
    df["category_encoded"] = df["category_encoded"].fillna(train_churn_mean)
    return df

X_train_s = impute(train)[feats1].values
X_val_s   = impute(val)[feats1].values
y_train   = train["churn"].values
y_val     = val["churn"].values

# ════════════════════════════════════════════════════════════════════════════
# BLOC 3 — GRILLES HYPERPARAMÈTRES
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

# ════════════════════════════════════════════════════════════════════════════
# BLOC 4 — FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════
def grid_search_lr(X_tr, y_tr, X_v, y_v):
    best_score, best_params, best_model = -1, None, None
    print("  Grid search LR...")
    for params in param_grid_lr:
        lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                                random_state=42, C=params["C"])
        lr.fit(X_tr, y_tr)
        score = average_precision_score(y_v, lr.predict_proba(X_v)[:, 1])
        print(f"    C={params['C']} → val PR-AUC : {score:.4f}")
        if score > best_score:
            best_score, best_params, best_model = score, params, lr
    print(f"  → Meilleur : {best_params} (val PR-AUC : {best_score:.4f})")
    return best_model, best_params

def grid_search_xgb(X_tr, y_tr, X_v, y_v, spw):
    best_score, best_params, best_model = -1, None, None
    print("  Grid search XGB...")
    for params in param_grid_xgb:
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            scale_pos_weight=spw,
            eval_metric="aucpr", random_state=42, verbosity=0
        )
        xgb.fit(X_tr, y_tr)
        score = average_precision_score(y_v, xgb.predict_proba(X_v)[:, 1])
        print(f"    depth={params['max_depth']} mcw={params['min_child_weight']} λ={params['reg_lambda']} → val PR-AUC : {score:.4f}")
        if score > best_score:
            best_score, best_params, best_model = score, params, xgb
    print(f"  → Meilleur : {best_params} (val PR-AUC : {best_score:.4f})")
    return best_model, best_params

def evaluate(y_true, y_proba, label):
    prauc = average_precision_score(y_true, y_proba)
    roc   = roc_auc_score(y_true, y_proba)
    print(f"  {label} → PR-AUC : {prauc:.4f} | ROC-AUC : {roc:.4f}")
    return prauc

def overfitting_check(y_tr, proba_tr, y_v, proba_v, label):
    prauc_tr = average_precision_score(y_tr, proba_tr)
    prauc_v  = average_precision_score(y_v,  proba_v)
    gap      = prauc_tr - prauc_v
    print(f"  [{label}] Train : {prauc_tr:.4f} | Val : {prauc_v:.4f} | Gap : {gap:.4f}", end="")
    print(" ⚠️ overfitting" if gap > 0.05 else " ✅ ok")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 5 — M1 : STRUCTURÉ SEUL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M1 — STRUCTURÉ SEUL")
print("="*60)

spw1 = (y_train==0).sum() / (y_train==1).sum()

sc1          = StandardScaler()
X_tr1_sc     = sc1.fit_transform(X_train_s)
X_val1_sc    = sc1.transform(X_val_s)

# LR
lr1, lr1_params = grid_search_lr(X_tr1_sc, y_train, X_val1_sc, y_val)
overfitting_check(y_train, lr1.predict_proba(X_tr1_sc)[:,1],
                  y_val,   lr1.predict_proba(X_val1_sc)[:,1], "M1 LR")

# XGB — pas de StandardScaler pour XGB
xgb1, xgb1_params = grid_search_xgb(X_train_s, y_train, X_val_s, y_val, spw1)
overfitting_check(y_train, xgb1.predict_proba(X_train_s)[:,1],
                  y_val,   xgb1.predict_proba(X_val_s)[:,1], "M1 XGB")

# Sauvegarde des probabilités val pour comparaison finale
val_probas_m1_lr  = lr1.predict_proba(X_val1_sc)[:, 1]
val_probas_m1_xgb = xgb1.predict_proba(X_val_s)[:, 1]

print(f"\nRésultats val M1 :")
evaluate(y_val, val_probas_m1_lr,  "LR")
evaluate(y_val, val_probas_m1_xgb, "XGB")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 6 — M2 : STRUCTURÉ + EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M2 — STRUCTURÉ + EMBEDDINGS")
print("="*60)

# Chargement modèle embedding
print("\nChargement du modèle d'embedding...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def prepare_text(df):
    title   = df["review_comment_title"].fillna("")
    message = df["review_comment_message"].fillna("")
    return (title + " " + message).str.strip().tolist()

print("Génération embeddings train...")
X_train_emb = model.encode(prepare_text(train), batch_size=64, show_progress_bar=True)
print("Génération embeddings val...")
X_val_emb   = model.encode(prepare_text(val),   batch_size=64, show_progress_bar=True)

print(f"Dimension embeddings : {X_train_emb.shape}")

# Sauvegarde embeddings — réutilisables pour le code final
np.save(save_path + "final_X_train_emb.npy", X_train_emb)
np.save(save_path + "final_X_val_emb.npy",   X_val_emb)
print("✅ Embeddings sauvegardés")

# Normalisation structuré + embeddings séparément
sc2s = StandardScaler()
sc2e = StandardScaler()

X_tr2 = np.hstack([sc2s.fit_transform(X_train_s), sc2e.fit_transform(X_train_emb)])
X_v2  = np.hstack([sc2s.transform(X_val_s),       sc2e.transform(X_val_emb)])
spw2  = (y_train==0).sum() / (y_train==1).sum()

# LR
lr2, lr2_params = grid_search_lr(X_tr2, y_train, X_v2, y_val)
overfitting_check(y_train, lr2.predict_proba(X_tr2)[:,1],
                  y_val,   lr2.predict_proba(X_v2)[:,1], "M2 LR")

# XGB
xgb2, xgb2_params = grid_search_xgb(X_tr2, y_train, X_v2, y_val, spw2)
overfitting_check(y_train, xgb2.predict_proba(X_tr2)[:,1],
                  y_val,   xgb2.predict_proba(X_v2)[:,1], "M2 XGB")

val_probas_m2_lr  = lr2.predict_proba(X_v2)[:, 1]
val_probas_m2_xgb = xgb2.predict_proba(X_v2)[:, 1]

print(f"\nRésultats val M2 :")
evaluate(y_val, val_probas_m2_lr,  "LR")
evaluate(y_val, val_probas_m2_xgb, "XGB")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 7 — RÉCAPITULATIF VAL + SAUVEGARDE HYPERPARAMÈTRES
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RÉCAPITULATIF VAL — M1 et M2")
print("="*60)
print(f"M1 LR  : PR-AUC = {average_precision_score(y_val, val_probas_m1_lr):.4f}  | params : {lr1_params}")
print(f"M1 XGB : PR-AUC = {average_precision_score(y_val, val_probas_m1_xgb):.4f}  | params : {xgb1_params}")
print(f"M2 LR  : PR-AUC = {average_precision_score(y_val, val_probas_m2_lr):.4f}  | params : {lr2_params}")
print(f"M2 XGB : PR-AUC = {average_precision_score(y_val, val_probas_m2_xgb):.4f}  | params : {xgb2_params}")

# Sauvegarde hyperparamètres pour le code final
import json
hyperparams = {
    "M1_LR":  lr1_params,
    "M1_XGB": xgb1_params,
    "M2_LR":  lr2_params,
    "M2_XGB": xgb2_params,
    "sc1_mean": sc1.mean_.tolist(),
    "sc1_std":  sc1.scale_.tolist(),
    "sc2s_mean": sc2s.mean_.tolist(),
    "sc2s_std":  sc2s.scale_.tolist(),
    "sc2e_mean": sc2e.mean_.tolist(),
    "sc2e_std":  sc2e.scale_.tolist(),
}
with open(save_path + "hyperparams_m1_m2.json", "w") as f:
    json.dump(hyperparams, f, indent=2)
print("\n✅ Hyperparamètres sauvegardés : hyperparams_m1_m2.json")
print("✅ Embeddings sauvegardés : final_X_train_emb.npy / final_X_val_emb.npy")