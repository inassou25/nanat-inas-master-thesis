import pandas as pd

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

train_pool = pd.read_csv(save_path + "split_train_pool.csv")
val        = pd.read_csv(save_path + "split_val.csv")
test       = pd.read_csv(save_path + "split_test.csv")

train_rag  = pd.read_csv(save_path + "m3_train_rag_stratified.csv")
val_rag    = pd.read_csv(save_path + "m3_val_rag.csv")
# test_rag n'existe pas encore — sera créé dans le code test final

print(f"Train pool : {len(train_pool)}")
print(f"Val        : {len(val)}")
print(f"Test       : {len(test)}")
print(f"Train RAG  : {len(train_rag)}")
print(f"Val RAG    : {len(val_rag)}")

print(f"\nOverlap train/test : {len(set(train_pool['order_id']) & set(test['order_id']))} (doit être 0)")
print(f"Overlap val/test   : {len(set(val['order_id']) & set(test['order_id']))} (doit être 0)")
print(f"Overlap train/val  : {len(set(train_pool['order_id']) & set(val['order_id']))} (doit être 0)")

print(f"\nTrain RAG aligné avec train pool : {set(train_rag['order_id']).issubset(set(train_pool['order_id']))} (doit être True)")
print(f"Val RAG aligné avec val          : {set(val_rag['order_id']).issubset(set(val['order_id']))} (doit être True)")

import pandas as pd
df = pd.read_csv("/Users/inasnanat/Desktop/Mémoire/code/split_train_pool.csv")
val = pd.read_csv("/Users/inasnanat/Desktop/Mémoire/code/split_val.csv")
test = pd.read_csv("/Users/inasnanat/Desktop/Mémoire/code/split_test.csv")
print(len(df) + len(val) + len(test))

#########################################################
print("c'est là le bon truc ")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
train_rag  = pd.read_csv(save_path + "m3_train_rag_stratified.csv")

# Sous-échantillon M3 — même paramètres que dans le pipeline
train_m3, _ = train_test_split(
    train_pool, train_size=1500,
    stratify=train_pool["churn"], random_state=42
)

# Vérification
print(f"Taille train_m3 : {len(train_m3)}")
print(f"Taille train_rag : {len(train_rag)}")

overlap = set(train_m3["order_id"]) & set(train_rag["order_id"])
print(f"Overlap order_id : {len(overlap)} (doit être 1500)")
print(f"Taux churn train_m3 : {train_m3['churn'].mean():.3f}")
print(f"Taux churn train_rag : {train_rag['churn'].mean():.3f}")