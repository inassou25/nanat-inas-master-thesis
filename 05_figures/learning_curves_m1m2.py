import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES SPLITS
# ════════════════════════════════════════════════════════════════════════════
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
val        = pd.read_csv(save_path + "split_val.csv",
                         parse_dates=["order_purchase_timestamp"])

print(f"Train pool : {len(train_pool)} obs | churn : {train_pool['churn'].mean():.1%}")
print(f"Val fixe   : {len(val)} obs | churn : {val['churn'].mean():.1%}")

# ════════════════════════════════════════════════════════════════════════════
# FEATURES STRUCTURÉES
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

# Target encoding calculé sur train_pool complet
state_means = train_pool.groupby("customer_state")["churn"].mean()
cat_means   = train_pool.groupby("product_category_name_english")["churn"].mean()

for df in [train_pool, val]:
    df["state_encoded"]    = df["customer_state"].map(state_means)
    df["category_encoded"] = df["product_category_name_english"].map(cat_means)

feats1   = struct_features + ["state_encoded", "category_encoded"]
X_val_s  = val[feats1].fillna(train_pool[feats1].median())
y_val    = val["churn"].values

# ════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS VAL — générés une seule fois
# ════════════════════════════════════════════════════════════════════════════
print("\nChargement modèle embedding...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def prepare_text(df):
    title   = df["review_comment_title"].fillna("")
    message = df["review_comment_message"].fillna("")
    return (title + " " + message).str.strip().tolist()

print("Génération embeddings val...")
X_val_emb = embedder.encode(prepare_text(val), batch_size=64, show_progress_bar=True)
np.save(save_path + "lc_X_val_emb.npy", X_val_emb)
print("✅ Embeddings val sauvegardés")

# ════════════════════════════════════════════════════════════════════════════
# LEARNING CURVE — tailles à tester
# ════════════════════════════════════════════════════════════════════════════
train_sizes = [500, 1000, 2000, 3500, 5000, 7000, 10000, 15000, 20000, 27000]
train_sizes = [s for s in train_sizes if s <= len(train_pool)]
print(f"\nTailles testées : {train_sizes}")

results = []
print(f"\n{'Size':>7} | {'M1 LR':>7} | {'M1 XGB':>7} | {'M2 LR':>7} | {'M2 XGB':>7}")
print("-" * 45)

for size in train_sizes:

    tr   = train_pool.iloc[:size].copy()
    y_tr = tr["churn"].values
    spw  = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    # ── Features structurées ─────────────────────────────────────────────
    X_tr_s = tr[feats1].fillna(train_pool[feats1].median())
    sc_s   = StandardScaler()
    X_tr_s_sc  = sc_s.fit_transform(X_tr_s)
    X_val_s_sc = sc_s.transform(X_val_s)

    # ── M1 LR ─────────────────────────────────────────────────────────────
    lr1 = LogisticRegression(class_weight="balanced", C=1.0,
                              max_iter=1000, random_state=42)
    lr1.fit(X_tr_s_sc, y_tr)
    m1_lr = average_precision_score(y_val, lr1.predict_proba(X_val_s_sc)[:, 1])

    # ── M1 XGB ────────────────────────────────────────────────────────────
    xgb1 = XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr", random_state=42, verbosity=0
    )
    xgb1.fit(X_tr_s_sc, y_tr)
    m1_xgb = average_precision_score(y_val, xgb1.predict_proba(X_val_s_sc)[:, 1])

    # ── Embeddings train ─────────────────────────────────────────────────
    print(f"  Embeddings train size={size}...")
    X_tr_emb = embedder.encode(prepare_text(tr), batch_size=64,
                                show_progress_bar=False)
    sc_e         = StandardScaler()
    X_tr_e_sc    = sc_e.fit_transform(X_tr_emb)
    X_val_e_sc   = sc_e.transform(X_val_emb)

    X_tr_c  = np.hstack([X_tr_s_sc,  X_tr_e_sc])
    X_val_c = np.hstack([X_val_s_sc, X_val_e_sc])

    # ── M2 LR ─────────────────────────────────────────────────────────────
    lr2 = LogisticRegression(class_weight="balanced", C=1.0,
                              max_iter=1000, random_state=42)
    lr2.fit(X_tr_c, y_tr)
    m2_lr = average_precision_score(y_val, lr2.predict_proba(X_val_c)[:, 1])

    # ── M2 XGB ────────────────────────────────────────────────────────────
    xgb2 = XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr", random_state=42, verbosity=0
    )
    xgb2.fit(X_tr_c, y_tr)
    m2_xgb = average_precision_score(y_val, xgb2.predict_proba(X_val_c)[:, 1])

    print(f"{size:>7} | {m1_lr:>7.4f} | {m1_xgb:>7.4f} | {m2_lr:>7.4f} | {m2_xgb:>7.4f}")
    results.append({
        "train_size": size,
        "M1_LR": m1_lr, "M1_XGB": m1_xgb,
        "M2_LR": m2_lr, "M2_XGB": m2_xgb
    })

# ════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE ET FIGURE
# ════════════════════════════════════════════════════════════════════════════
df_lc = pd.DataFrame(results)
df_lc.to_csv(save_path + "lc_m1_m2_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gauche — M1
axes[0].plot(df_lc["train_size"], df_lc["M1_LR"],
             marker="o", color="#d62728", linewidth=2, label="M1 LR")
axes[0].plot(df_lc["train_size"], df_lc["M1_XGB"],
             marker="s", color="#d62728", linewidth=2,
             linestyle="--", label="M1 XGB")
axes[0].set_title("Learning curve M1 — Structuré seul", fontsize=13)
axes[0].set_xlabel("Taille du train set", fontsize=12)
axes[0].set_ylabel("PR-AUC (val fixe)", fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.2, 0.6)

# Droite — M2
axes[1].plot(df_lc["train_size"], df_lc["M2_LR"],
             marker="o", color="#1f77b4", linewidth=2, label="M2 LR")
axes[1].plot(df_lc["train_size"], df_lc["M2_XGB"],
             marker="s", color="#1f77b4", linewidth=2,
             linestyle="--", label="M2 XGB")
axes[1].set_title("Learning curve M2 — Structuré + Embeddings", fontsize=13)
axes[1].set_xlabel("Taille du train set", fontsize=12)
axes[1].set_ylabel("PR-AUC (val fixe)", fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.5, 1.0)

plt.suptitle("Learning curves — évaluation sur val fixe (5831 obs)", fontsize=14)
plt.tight_layout()
plt.savefig(save_path + "lc_m1_m2.png", dpi=150)
print("\n✅ Figure sauvegardée : lc_m1_m2.png")
plt.show()

print("\n" + "="*50)
print("RÉSULTATS FINAUX")
print("="*50)
print(df_lc.to_string(index=False))