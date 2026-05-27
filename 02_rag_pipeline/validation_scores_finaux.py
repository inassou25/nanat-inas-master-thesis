import pandas as pd
import numpy as np
from scipy import stats

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ── Chargement ────────────────────────────────────────────────────────────────
# On utilise le train pool et les nouveaux scores RAG
train      = pd.read_csv(save_path + "split_train_pool.csv")
train_rag  = pd.read_csv(save_path + "m3_train_rag_stratified.csv")


# ── Jointure ──────────────────────────────────────────────────────────────────
df = train.merge(
    train_rag[['order_id', 'livraison', 'qualite_produit', 'service_client', 'prix']],
    on='order_id', how='inner'
)

print(f"Dataset : {len(df)} lignes")

# ── Corrélation scores RAG vs note de review ──────────────────────────────────
print(f"\n=== Corrélation scores RAG vs note de review ===")
rag_dims = ['livraison', 'qualite_produit', 'service_client', 'prix']
for dim in rag_dims:
    corr, pval = stats.pearsonr(df[dim], df['review_score'])
    print(f"{dim:20} : r = {corr:.3f} (p = {pval:.2e})")

# ── Corrélation scores RAG vs churn ───────────────────────────────────────────
print(f"\n=== Corrélation scores RAG vs churn ===")
for dim in rag_dims:
    corr, pval = stats.pointbiserialr(df['churn'], df[dim])
    print(f"{dim:20} : r = {corr:.3f} (p = {pval:.2e})")

# ── Moyenne des scores par note de review ─────────────────────────────────────
print(f"\n=== Moyenne des scores par note de review ===")
print(df.groupby('review_score')[rag_dims].mean().round(3))

# ── Moyenne des scores par churn ──────────────────────────────────────────────
print(f"\n=== Moyenne des scores par churn ===")
print(df.groupby('churn')[rag_dims].mean().round(3))