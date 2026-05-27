import pandas as pd
import numpy as np
import json

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES RÉSULTATS
# ════════════════════════════════════════════════════════════════════════════
results    = pd.read_csv(save_path + "final_results.csv")
thresholds = json.load(open(save_path + "optimal_thresholds.json"))
hp12       = json.load(open(save_path + "hyperparams_m1_m2.json"))
hp3        = json.load(open(save_path + "hyperparams_m3.json"))

# ════════════════════════════════════════════════════════════════════════════
# TABLEAU 1 — RÉSULTATS COMPLETS PAR MODÈLE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("RÉSULTATS FINAUX — TEST SET (5832 observations, churn ~19.2%)")
print("="*80)
print(f"{'Méthode':<35} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'Recall':>8} {'Precision':>9} {'Seuil':>7}")
print("-" * 80)

for _, row in results.iterrows():
    name = row["label"].replace("TEST — ", "")
    print(f"{name:<35} {row['pr_auc']:>8.4f} {row['roc_auc']:>8.4f} "
          f"{row['f1']:>8.4f} {row['recall']:>8.4f} {row['precision']:>9.4f} "
          f"{row['threshold']:>7.3f}")

# ════════════════════════════════════════════════════════════════════════════
# TABLEAU 2 — MEILLEURS MODÈLES PAR CONDITION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("MEILLEURS MODÈLES PAR CONDITION")
print("="*80)
print(f"{'Condition':<10} {'Modèle':<30} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8}")
print("-" * 65)

conditions = {
    "M1": results[results["label"].str.contains("M1")].sort_values("pr_auc", ascending=False).iloc[0],
    "M2": results[results["label"].str.contains("M2")].sort_values("pr_auc", ascending=False).iloc[0],
    "M3": results[results["label"].str.contains("M3")].sort_values("pr_auc", ascending=False).iloc[0],
    "M4": results[results["label"].str.contains("M4")].iloc[0],
}

for cond, row in conditions.items():
    name = row["label"].replace("TEST — ", "")
    print(f"{cond:<10} {name:<30} {row['pr_auc']:>8.4f} {row['roc_auc']:>8.4f} {row['f1']:>8.4f}")

# ════════════════════════════════════════════════════════════════════════════
# TABLEAU 3 — HYPERPARAMÈTRES RETENUS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("HYPERPARAMÈTRES RETENUS (sélectionnés sur val par grid search PR-AUC)")
print("="*80)

print(f"\nM1 LR  : C = {hp12['M1_LR']['C']}")
print(f"M1 XGB : max_depth={hp12['M1_XGB']['max_depth']} | "
      f"min_child_weight={hp12['M1_XGB']['min_child_weight']} | "
      f"reg_lambda={hp12['M1_XGB']['reg_lambda']} | "
      f"reg_alpha={hp12['M1_XGB']['reg_alpha']} | "
      f"subsample={hp12['M1_XGB']['subsample']} | "
      f"colsample_bytree={hp12['M1_XGB']['colsample_bytree']}")

print(f"\nM2 LR  : C = {hp12['M2_LR']['C']}")
print(f"M2 XGB : max_depth={hp12['M2_XGB']['max_depth']} | "
      f"min_child_weight={hp12['M2_XGB']['min_child_weight']} | "
      f"reg_lambda={hp12['M2_XGB']['reg_lambda']} | "
      f"reg_alpha={hp12['M2_XGB']['reg_alpha']} | "
      f"subsample={hp12['M2_XGB']['subsample']} | "
      f"colsample_bytree={hp12['M2_XGB']['colsample_bytree']}")

print(f"\nM3 LR  : C = {hp3['M3_LR']['C']}")
print(f"M3 XGB : max_depth={hp3['M3_XGB']['max_depth']} | "
      f"min_child_weight={hp3['M3_XGB']['min_child_weight']} | "
      f"reg_lambda={hp3['M3_XGB']['reg_lambda']} | "
      f"reg_alpha={hp3['M3_XGB']['reg_alpha']} | "
      f"subsample={hp3['M3_XGB']['subsample']} | "
      f"colsample_bytree={hp3['M3_XGB']['colsample_bytree']}")

print(f"\nM4 Zero-shot : GPT-4o mini | temperature=0 | aucun entraînement")

# ════════════════════════════════════════════════════════════════════════════
# TABLEAU 4 — SEUILS OPTIMAUX
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SEUILS OPTIMAUX (trouvés sur val, appliqués sur test)")
print("="*80)
print(f"{'Méthode':<12} {'Seuil':>8} {'F1 val':>10}")
print("-" * 35)
for name, vals in thresholds.items():
    print(f"{name:<12} {vals['threshold']:>8.3f} {vals['f1_val']:>10.4f}")

# ════════════════════════════════════════════════════════════════════════════
# TABLEAU 5 — BOOTSTRAP IC 95%
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("BOOTSTRAP PR-AUC IC 95% (1000 itérations)")
print("="*80)

# Recharger les probabilités et recalculer le bootstrap
from sklearn.metrics import average_precision_score
from sklearn.utils import resample

probas_12 = pd.read_csv(save_path + "test_probas_m1_m2.csv")
probas_3  = pd.read_csv(save_path + "test_probas_m3.csv")
probas_4  = pd.read_csv(save_path + "test_probas_m4.csv")

bootstrap_data = {
    "M1 LR":  (probas_12["churn"].values, probas_12["M1_LR"].values),
    "M1 XGB": (probas_12["churn"].values, probas_12["M1_XGB"].values),
    "M2 LR":  (probas_12["churn"].values, probas_12["M2_LR"].values),
    "M2 XGB": (probas_12["churn"].values, probas_12["M2_XGB"].values),
    "M3 LR":  (probas_3["churn"].values,  probas_3["M3_LR"].values),
    "M3 XGB": (probas_3["churn"].values,  probas_3["M3_XGB"].values),
    "M4 ZS":  (probas_4["churn"].values,  probas_4["churn_proba"].values),
}

print(f"{'Méthode':<12} {'PR-AUC':>8} {'IC 95%':>12}")
print("-" * 35)
for name, (y_true, y_proba) in bootstrap_data.items():
    scores = []
    for _ in range(1000):
        idx = resample(range(len(y_true)), random_state=None)
        try:
            scores.append(average_precision_score(
                np.array(y_true)[idx], np.array(y_proba)[idx]))
        except:
            pass
    mean = np.mean(scores)
    ci   = np.std(scores) * 1.96
    print(f"{name:<12} {mean:>8.3f} {'± '+f'{ci:.3f}':>12}")


import pandas as pd
save_path = "/Users/inasnanat/Desktop/Mémoire/code/"
df = pd.read_csv(save_path + "option_b_results.csv")
print(df[["label","mae","rmse","spearman"]].to_string(index=False))