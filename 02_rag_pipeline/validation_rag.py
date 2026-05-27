import pandas as pd
import numpy as np
from scipy import stats

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ── Chargement ────────────────────────────────────────────────────────────────
train = pd.read_csv(save_path + "train.csv")
train_rag = pd.read_csv(save_path + "train_rag_scores_7k.csv")

# ── Jointure pour avoir les scores RAG + la note de review ───────────────────
df = train.merge(train_rag[['order_id', 'livraison', 'qualite_produit', 
                              'service_client', 'prix']], on='order_id', how='inner')

print(f"Dataset : {len(df)} lignes")
print(f"\n=== Corrélation scores RAG vs note de review ===")

rag_dims = ['livraison', 'qualite_produit', 'service_client', 'prix']
for dim in rag_dims:
    corr, pval = stats.pearsonr(df[dim], df['review_score'])
    print(f"{dim:20} : r = {corr:.3f} (p = {pval:.2e})")

print(f"\n=== Corrélation scores RAG vs churn ===")
for dim in rag_dims:
    corr, pval = stats.pointbiserialr(df['churn'], df[dim])
    print(f"{dim:20} : r = {corr:.3f} (p = {pval:.2e})")

print(f"\n=== Moyenne des scores par note de review ===")
print(df.groupby('review_score')[rag_dims].mean().round(3))

print(f"\n=== Moyenne des scores par churn ===")
print(df.groupby('churn')[rag_dims].mean().round(3))

#Toutes les p-values sont quasi nulles — les corrélations sont statistiquement significatives.
#Toutes négatives — logique. Plus le score est bas, plus le client a churné. livraison à -0.589 et qualite_produit à -0.486 sont particulièrement forts.
#La table par note de review
#C'est le résultat le plus parlant. Regarde livraison :

#Note 1 → score moyen 0.198
#Note 5 → score moyen 0.747

#La progression est parfaitement monotone pour livraison et qualite_produit. Les scores GPT capturent fidèlement la réalité — ce n'est pas du bruit.