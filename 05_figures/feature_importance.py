import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ── Chargement ────────────────────────────────────────────────────────────────
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
train_rag  = pd.read_csv(save_path + "m3_train_rag_stratified.csv")
hp12       = json.load(open(save_path + "hyperparams_m1_m2.json"))
hp3        = json.load(open(save_path + "hyperparams_m3.json"))

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
feats1             = struct_features + ["state_encoded", "category_encoded"]
all_feats3         = feats1 + rag_features + mentioned_features

# ── Target encoding M1 ───────────────────────────────────────────────────────
state_means_12  = train_pool.groupby("customer_state")["churn"].mean()
cat_means_12    = train_pool.groupby("product_category_name_english")["churn"].mean()
train_pool["state_encoded"]    = train_pool["customer_state"].map(state_means_12)
train_pool["category_encoded"] = train_pool["product_category_name_english"].map(cat_means_12)

X_train_m1 = train_pool[feats1].fillna(
    train_pool[feats1].median()).astype(float).values
y_train_m1 = train_pool["churn"].values
spw1       = (y_train_m1==0).sum() / (y_train_m1==1).sum()

# ── M1 XGB ────────────────────────────────────────────────────────────────────
p1 = hp12["M1_XGB"]
xgb1 = XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=p1["max_depth"], min_child_weight=p1["min_child_weight"],
    subsample=p1["subsample"], colsample_bytree=p1["colsample_bytree"],
    reg_alpha=p1["reg_alpha"], reg_lambda=p1["reg_lambda"],
    scale_pos_weight=spw1,
    eval_metric="aucpr", random_state=42, verbosity=0, nthread=1
)
xgb1.fit(X_train_m1, y_train_m1)

fi_m1 = pd.DataFrame({
    "feature":    feats1,
    "importance": xgb1.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

# ── Target encoding M3 ───────────────────────────────────────────────────────
train_m3, _ = train_test_split(
    train_pool, train_size=1500,
    stratify=train_pool["churn"], random_state=42
)
train_m3 = train_m3.sort_values("order_purchase_timestamp").reset_index(drop=True)

state_means_3 = train_m3.groupby("customer_state")["churn"].mean()
cat_means_3   = train_m3.groupby("product_category_name_english")["churn"].mean()
train_m3["state_encoded"]    = train_m3["customer_state"].map(state_means_3)
train_m3["category_encoded"] = train_m3["product_category_name_english"].map(cat_means_3)

train3 = train_m3.merge(train_rag[["order_id"] + rag_features],
                        on="order_id", how="inner")
for dim in rag_features:
    train3[f'{dim}_mentioned'] = (train3[dim] != 0.5).astype(int)

for col in struct_features + ["state_encoded","category_encoded"]:
    train3[col] = train3[col].fillna(train_m3[col].median()
                                     if col in struct_features
                                     else train_m3["churn"].mean())
for col in rag_features + mentioned_features:
    train3[col] = train3[col].fillna(0.5)

X_train3 = train3[all_feats3].astype(float).values
y_train3 = train3["churn"].values
spw3     = (y_train3==0).sum() / (y_train3==1).sum()

sc3         = StandardScaler()
X_train3_sc = sc3.fit_transform(X_train3)

# ── M3 XGB ────────────────────────────────────────────────────────────────────
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

fi_m3 = pd.DataFrame({
    "feature":    all_feats3,
    "importance": xgb3.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

# ── Labels français avec nom technique entre parenthèses ─────────────────────
labels = {
    "is_late":                   "Retard de livraison (is_late)",
    "n_items":                   "Nombre d'articles (n_items)",
    "delivery_delay_days":       "Délai de retard en jours (delivery_delay_days)",
    "n_sellers":                 "Nombre de vendeurs (n_sellers)",
    "processing_time_days":      "Délai de traitement (processing_time_days)",
    "category_encoded":          "Catégorie produit (category_encoded)",
    "recency_days":              "Récence en jours (recency_days)",
    "product_weight_g":          "Poids du produit (product_weight_g)",
    "state_encoded":             "État client (state_encoded)",
    "payment_type_voucher":      "Paiement voucher",
    "payment_value":             "Montant payé (payment_value)",
    "monetary":                  "Montant total (monetary)",
    "freight_ratio":             "Ratio frais de port (freight_ratio)",
    "n_installments":            "Nombre de mensualités (n_installments)",
    "payment_type_boleto":       "Paiement boleto",
    "payment_type_credit_card":  "Paiement carte de crédit",
    "payment_type_debit_card":   "Paiement carte de débit",
    "frequency":                 "Fréquence d'achat (frequency)",
    "livraison":                 "Score RAG — Livraison",
    "qualite_produit":           "Score RAG — Qualité produit",
    "service_client":            "Score RAG — Service client",
    "prix":                      "Score RAG — Prix",
    "livraison_mentioned":       "Livraison mentionnée (binaire)",
    "qualite_produit_mentioned": "Qualité produit mentionnée (binaire)",
    "service_client_mentioned":  "Service client mentionné (binaire)",
    "prix_mentioned":            "Prix mentionné (binaire)",
}

fi_m1["label"] = fi_m1["feature"].map(labels).fillna(fi_m1["feature"])
fi_m3["label"] = fi_m3["feature"].map(labels).fillna(fi_m3["feature"])

top10_m1 = fi_m1.head(10)
top10_m3 = fi_m3.head(10)

# Vérification chiffres
print("=== Top 5 M1 XGB ===")
for _, r in top10_m1.head(5).iterrows():
    print(f"  {r['feature']:<30} : {r['importance']:.4f}")

print("\n=== Top 5 M3 XGB ===")
for _, r in top10_m3.head(5).iterrows():
    print(f"  {r['feature']:<30} : {r['importance']:.4f}")

# ── FIGURE 1 — M1 XGB ────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))

colors_m1 = ["#c0392b" if row["feature"] == "is_late"
              else "#95a5a6" for _, row in top10_m1.iterrows()]

bars1 = ax1.barh(top10_m1["label"][::-1],
                  top10_m1["importance"][::-1],
                  color=colors_m1[::-1], edgecolor="white", height=0.6)

for bar, val in zip(bars1, top10_m1["importance"][::-1]):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9, color="#333333")

ax1.set_xlabel("Importance (gain)", fontsize=12)
ax1.set_title("Figure 1 — Importance des variables — M1 XGBoost\n"
              "(Données structurées seules — Top 10)",
              fontsize=13, pad=12)
ax1.set_xlim(0, top10_m1["importance"].max() * 1.20)
ax1.grid(True, axis="x", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(save_path + "figure1_fi_m1_final.png", dpi=150, bbox_inches="tight")
print("\n✅ Figure 1 sauvegardée")
plt.show()

# ── FIGURE 2 — M3 XGB ────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))

rag_feats_set = set(rag_features + mentioned_features)
colors_m3 = ["#2980b9" if row["feature"] in rag_feats_set
              else "#95a5a6" for _, row in top10_m3.iterrows()]

bars2 = ax2.barh(top10_m3["label"][::-1],
                  top10_m3["importance"][::-1],
                  color=colors_m3[::-1], edgecolor="white", height=0.6)

for bar, val in zip(bars2, top10_m3["importance"][::-1]):
    ax2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9, color="#333333")

ax2.set_xlabel("Importance (gain)", fontsize=12)
ax2.set_title("Figure 2 — Importance des variables — M3 XGBoost\n"
              "(Données structurées + RAG — Top 10)",
              fontsize=13, pad=12)
ax2.set_xlim(0, top10_m3["importance"].max() * 1.20)
ax2.grid(True, axis="x", alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2980b9", label="Variables issues du pipeline RAG"),
    Patch(facecolor="#95a5a6", label="Variables structurées")
]
ax2.legend(handles=legend_elements, loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig(save_path + "figure2_fi_m3_final.png", dpi=150, bbox_inches="tight")
print("✅ Figure 2 sauvegardée")
plt.show()

# ── ANNEXE H — tableaux complets ─────────────────────────────────────────────
print("\n" + "="*60)
print("ANNEXE H — Feature importance complète M1 XGB")
print("="*60)
for _, r in fi_m1[fi_m1["importance"] > 0].iterrows():
    print(f"  {r['label']:<45} : {r['importance']:.4f}")

print("\n" + "="*60)
print("ANNEXE H — Feature importance complète M3 XGB")
print("="*60)
for _, r in fi_m3[fi_m3["importance"] > 0].iterrows():
    print(f"  {r['label']:<45} : {r['importance']:.4f}")