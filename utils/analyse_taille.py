import pandas as pd

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"
train = pd.read_csv(save_path + "split_train_pool.csv")

train["text_length"] = (
    train["review_comment_title"].fillna("") + " " + 
    train["review_comment_message"].fillna("")
).str.strip().str.len()

print(f"Longueur moyenne : {train['text_length'].mean():.0f} caractères")
print(f"Médiane          : {train['text_length'].median():.0f} caractères")
print(f"75e percentile   : {train['text_length'].quantile(0.75):.0f} caractères")
print(f"90e percentile   : {train['text_length'].quantile(0.90):.0f} caractères")
print(f"% reviews < 200 car : {(train['text_length'] < 200).mean()*100:.1f}%")
print(f"% reviews < 100 car : {(train['text_length'] < 100).mean()*100:.1f}%")

import sys
sys.path.append("/Users/inasnanat/Desktop/Mémoire/code/")
from knowledge_base import knowledge_base

lengths = [len(p["text"]) for p in knowledge_base]

import numpy as np
print(f"Nombre de passages     : {len(lengths)}")
print(f"Longueur moyenne       : {np.mean(lengths):.0f} caractères")
print(f"Médiane                : {np.median(lengths):.0f} caractères")
print(f"Min                    : {min(lengths)} caractères")
print(f"Max                    : {max(lengths)} caractères")
print(f"75e percentile         : {np.percentile(lengths, 75):.0f} caractères")
print(f"% passages < 200 car   : {sum(l < 200 for l in lengths)/len(lengths)*100:.1f}%")
print(f"% passages >= 200 car  : {sum(l >= 200 for l in lengths)/len(lengths)*100:.1f}%")

print("\nPassages > 200 caractères :")
for p in knowledge_base:
    if len(p["text"]) > 200:
        print(f"  {p['id']} ({len(p['text'])} car) : {p['text'][:50]}...")


import pandas as pd

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"
train_rag = pd.read_csv(save_path + "m3_train_rag_stratified.csv")

# Recréer les binaires comme dans le pipeline
for dim in ["livraison", "qualite_produit", "service_client", "prix"]:
    train_rag[f"{dim}_mentioned"] = (train_rag[dim] != 0.5).astype(int)

print("% de reviews où la dimension est mentionnée :")
for dim in ["livraison", "qualite_produit", "service_client", "prix"]:
    pct = train_rag[f"{dim}_mentioned"].mean() * 100
    print(f"  {dim:<20} : {pct:.1f}%")