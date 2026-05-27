import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# BLOC 1 — CHARGEMENT DES PROBABILITÉS TEST
# ════════════════════════════════════════════════════════════════════════════
probas_12 = pd.read_csv(save_path + "test_probas_m1_m2.csv")
probas_3  = pd.read_csv(save_path + "test_probas_m3.csv")
probas_4  = pd.read_csv(save_path + "test_probas_m4.csv")

print(f"M1/M2 : {len(probas_12)} obs")
print(f"M3    : {len(probas_3)} obs")
print(f"M4    : {len(probas_4)} obs")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 2 — CALIBRATION CURVE
# ════════════════════════════════════════════════════════════════════════════
models = {
    "M1 XGB (Structuré seul)":        (probas_12["churn"].values, probas_12["M1_XGB"].values),
    "M2 LR (Structuré + Embeddings)": (probas_12["churn"].values, probas_12["M2_LR"].values),
    "M3 XGB (Structuré + RAG)":       (probas_3["churn"].values,  probas_3["M3_XGB"].values),
    "M4 Zero-shot (GPT-4o mini)":     (probas_4["churn"].values,  probas_4["churn_proba"].values),
}

colors = {
    "M1 XGB (Structuré seul)":        "#d62728",
    "M2 LR (Structuré + Embeddings)": "#1f77b4",
    "M3 XGB (Structuré + RAG)":       "#2ca02c",
    "M4 Zero-shot (GPT-4o mini)":     "#ff7f0e",
}

fig, ax = plt.subplots(figsize=(10, 8))

# Diagonale parfaite
ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
        linewidth=1.5, label="Calibration parfaite")

for label, (y_true, y_proba) in models.items():
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=10, strategy="uniform"
    )
    ax.plot(mean_pred, fraction_pos,
            marker="o", linewidth=2,
            color=colors[label], label=label)

ax.set_xlabel("Probabilité prédite moyenne", fontsize=13)
ax.set_ylabel("Fraction de positifs observés", fontsize=13)
ax.set_title("Courbe de calibration — Comparaison des 4 conditions\n(test set, 5832 observations)", fontsize=14)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(save_path + "calibration_curve.png", dpi=150)
print("✅ Calibration curve sauvegardée")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# BLOC 3 — COURBE DE GAIN CUMULATIF
# ════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Ligne aléatoire
ax2.plot([0, 1], [0, 1], linestyle="--", color="gray",
         linewidth=1.5, label="Modèle aléatoire")

for label, (y_true, y_proba) in models.items():
    # Tri par probabilité décroissante
    sorted_idx   = np.argsort(y_proba)[::-1]
    y_sorted     = y_true[sorted_idx]
    total_pos    = y_true.sum()
    n            = len(y_true)

    # Pourcentage de la population ciblée vs % de churners capturés
    pct_pop      = np.arange(1, n + 1) / n
    pct_captured = np.cumsum(y_sorted) / total_pos

    ax2.plot(pct_pop, pct_captured,
             linewidth=2, color=colors[label], label=label)

ax2.set_xlabel("Proportion de clients ciblés (%)", fontsize=13)
ax2.set_ylabel("Proportion de churners capturés (%)", fontsize=13)
ax2.set_title("Courbe de gain cumulatif — Comparaison des 4 conditions\n(test set, 5832 observations)", fontsize=14)
ax2.legend(fontsize=11, loc="lower right")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Annotations utiles — exemple à 20% de la population
ax2.axvline(x=0.20, color="black", linestyle=":", alpha=0.5)
ax2.text(0.21, 0.05, "20% clients\nciblés", fontsize=9, alpha=0.7)

plt.tight_layout()
plt.savefig(save_path + "cumulative_gain_curve.png", dpi=150)
print("✅ Courbe de gain cumulatif sauvegardée")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# BLOC 4 — TABLEAU DE GAIN À 10%, 20%, 30%
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("GAIN CUMULATIF — % churners capturés par seuil de ciblage")
print("="*60)
print(f"{'Méthode':<35} {'10%':>6} {'20%':>6} {'30%':>6} {'50%':>6}")
print("-" * 60)

for label, (y_true, y_proba) in models.items():
    sorted_idx   = np.argsort(y_proba)[::-1]
    y_sorted     = y_true[sorted_idx]
    total_pos    = y_true.sum()
    n            = len(y_true)
    pct_captured = np.cumsum(y_sorted) / total_pos

    t10 = pct_captured[int(n * 0.10) - 1]
    t20 = pct_captured[int(n * 0.20) - 1]
    t30 = pct_captured[int(n * 0.30) - 1]
    t50 = pct_captured[int(n * 0.50) - 1]

    print(f"{label:<35} {t10:>6.1%} {t20:>6.1%} {t30:>6.1%} {t50:>6.1%}")