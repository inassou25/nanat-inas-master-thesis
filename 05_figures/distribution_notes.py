import pandas as pd
import matplotlib.pyplot as plt

save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

df = pd.read_csv(save_path + "split_train_pool.csv")

print("Distribution des notes de review :")
print(df["review_score"].value_counts().sort_index())
print(f"\nEn pourcentage :")
print((df["review_score"].value_counts(normalize=True).sort_index() * 100).round(1))

fig, ax = plt.subplots(figsize=(8, 5))
counts = df["review_score"].value_counts().sort_index()
ax.bar(counts.index, counts.values, color=["#d62728","#d62728","#aec7e8","#1f77b4","#1f77b4"],
       edgecolor="white", width=0.6)
ax.set_xlabel("Note de satisfaction", fontsize=13)
ax.set_ylabel("Nombre d'observations", fontsize=13)
ax.set_title("Distribution des notes de review — Train pool", fontsize=14)
ax.set_xticks([1, 2, 3, 4, 5])
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(save_path + "distribution_notes.png", dpi=150)
print("\n✅ Figure sauvegardée")
plt.show()