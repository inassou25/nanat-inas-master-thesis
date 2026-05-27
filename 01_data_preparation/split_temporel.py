import pandas as pd
import numpy as np

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# SPLIT TEMPOREL STRICT — sur tout le dataset avec texte
# ════════════════════════════════════════════════════════════════════════════

df_master = pd.read_csv(save_path + "master_table.csv",
                        parse_dates=["order_purchase_timestamp"])

# Garder uniquement les observations avec texte
df_text = df_master[
    df_master["review_comment_message"].notna() &
    (df_master["review_comment_message"].str.strip() != "") &
    df_master["review_score"].notna()
].copy()

# Tri chronologique strict
df_text = df_text.sort_values("order_purchase_timestamp").reset_index(drop=True)

n = len(df_text)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_pool = df_text.iloc[:train_end].copy()
val        = df_text.iloc[train_end:val_end].copy()
test       = df_text.iloc[val_end:].copy()

# Vérifications
print(f"Total observations : {n}")
print(f"\nTrain pool : {len(train_pool)} obs ({len(train_pool)/n:.1%})")
print(f"  Période : {train_pool['order_purchase_timestamp'].min().date()} "
      f"→ {train_pool['order_purchase_timestamp'].max().date()}")
print(f"  Churn : {train_pool['churn'].mean():.1%}")

print(f"\nVal fixe : {len(val)} obs ({len(val)/n:.1%})")
print(f"  Période : {val['order_purchase_timestamp'].min().date()} "
      f"→ {val['order_purchase_timestamp'].max().date()}")
print(f"  Churn : {val['churn'].mean():.1%}")

print(f"\nTest définitif : {len(test)} obs ({len(test)/n:.1%})")
print(f"  Période : {test['order_purchase_timestamp'].min().date()} "
      f"→ {test['order_purchase_timestamp'].max().date()}")
print(f"  Churn : {test['churn'].mean():.1%}")

# Sauvegarde
train_pool.to_csv(save_path + "split_train_pool.csv", index=False)
val.to_csv(save_path        + "split_val.csv",        index=False)
test.to_csv(save_path       + "split_test.csv",       index=False)

print(f"\n✅ Splits sauvegardés")
print(f"   split_train_pool.csv — {len(train_pool)} lignes")
print(f"   split_val.csv        — {len(val)} lignes")
print(f"   split_test.csv       — {len(test)} lignes")
print(f"\n🔒 TEST SET GELÉ — ne plus y toucher jusqu'à l'évaluation finale")