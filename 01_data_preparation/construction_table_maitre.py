import pandas as pd
import numpy as np

# ── 1. Chargement des tables ──────────────────────────────────────────────────
path = "/Users/inasnanat/Desktop/olist/"

orders        = pd.read_csv(path + "olist_orders_dataset.csv")
order_items   = pd.read_csv(path + "olist_order_items_dataset.csv")
order_reviews = pd.read_csv(path + "olist_order_reviews_dataset.csv")
order_pays    = pd.read_csv(path + "olist_order_payments_dataset.csv")
customers     = pd.read_csv(path + "olist_customers_dataset.csv")
products      = pd.read_csv(path + "olist_products_dataset.csv")
sellers       = pd.read_csv(path + "olist_sellers_dataset.csv")

# ── 2. Conversion des dates ───────────────────────────────────────────────────
date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

# ── 3. Garder uniquement les commandes livrées ────────────────────────────────
orders = orders[orders["order_status"] == "delivered"].copy()

# ── 4. Features depuis order_items (agrégé par commande) ─────────────────────
items_agg = order_items.groupby("order_id").agg(
    n_items            = ("order_item_id", "count"),
    n_sellers          = ("seller_id", "nunique"),
    total_price        = ("price", "sum"),
    total_freight      = ("freight_value", "sum")
).reset_index()

# ── 5. Features depuis order_payments (agrégé par commande) ──────────────────
pays_agg = order_pays.groupby("order_id").agg(
    n_installments     = ("payment_installments", "max"),
    payment_type       = ("payment_type", lambda x: x.mode()[0]),
    payment_value      = ("payment_value", "sum")
).reset_index()

# ── 6. Review (une review par commande — garder la plus récente si doublon) ───
order_reviews["review_creation_date"] = pd.to_datetime(
    order_reviews["review_creation_date"]
)
reviews_clean = (
    order_reviews
    .sort_values("review_creation_date", ascending=False)
    .drop_duplicates(subset="order_id", keep="first")
)[["order_id", "review_score", "review_comment_title", "review_comment_message"]]

# ── 7. Jointures ──────────────────────────────────────────────────────────────
df = (
    orders
    .merge(customers,    on="customer_id",  how="left")
    .merge(items_agg,    on="order_id",     how="left")
    .merge(pays_agg,     on="order_id",     how="left")
    .merge(reviews_clean, on="order_id",    how="left")
)

# ── 8. Features temporelles ───────────────────────────────────────────────────
df["delivery_delay_days"] = (
    df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
).dt.days  # positif = retard, négatif = en avance

df["processing_time_days"] = (
    df["order_delivered_carrier_date"] - df["order_approved_at"]
).dt.days

# ── 9. Variable cible ─────────────────────────────────────────────────────────
df["churn"] = (df["review_score"] <= 2).astype(int)

# ── 10. Vérification rapide ───────────────────────────────────────────────────
print(f"Shape : {df.shape}")
print(f"Churners : {df['churn'].sum()} ({df['churn'].mean():.1%})")
print(f"Missing review_score : {df['review_score'].isna().sum()}")
print(df.dtypes)

df = df[df["review_score"].notna()].copy()
print(f"Shape après exclusion missing reviews : {df.shape}")
print(f"Churners : {df['churn'].sum()} ({df['churn'].mean():.1%})")

# ── Features RFM + comportementales ──────────────────────────────────────────
reference_date = df["order_purchase_timestamp"].max()

# Récence : jours entre l'achat et la date de référence
df["recency_days"] = (
    reference_date - df["order_purchase_timestamp"]
).dt.days

# Fréquence : toujours 1 dans Olist, mais on la garde pour la forme
df["frequency"] = 1

# Montant : total dépensé (price + freight)
df["monetary"] = df["total_price"] + df["total_freight"]

# Ratio freight : part des frais de port dans le total
df["freight_ratio"] = df["total_freight"] / df["monetary"]

# Retard binaire
df["is_late"] = (df["delivery_delay_days"] > 0).astype(int)

# Vérification
print(df[["recency_days", "monetary", "freight_ratio", 
          "delivery_delay_days", "is_late"]].describe())

# ── Encodage payment_type (one-hot) ──────────────────────────────────────────
df = pd.get_dummies(df, columns=["payment_type"], drop_first=False)

# ── customer_state : on le garde en label (trop de modalités pour one-hot) ───
# On le transformera en target encoding au moment du modèle

# ── Sélection des colonnes finales ───────────────────────────────────────────
cols_to_keep = [
    # Identifiants
    "order_id", "customer_unique_id",
    # Temporel
    "order_purchase_timestamp",
    # RFM
    "recency_days", "frequency", "monetary",
    # Comportemental
    "n_items", "n_sellers", "freight_ratio",
    "delivery_delay_days", "is_late", "processing_time_days",
    "n_installments",
    # Paiement
    "payment_value",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    # Géographique
    "customer_state",
    # Texte
    "review_comment_title", "review_comment_message",
    # Cible
    "review_score", "churn"
]

df_master = df[cols_to_keep].copy()

print(f"Table maître : {df_master.shape}")
print(df_master.head(3))

# ── Chargement des tables manquantes ─────────────────────────────────────────
products     = pd.read_csv(path + "olist_products_dataset.csv")
translation  = pd.read_csv(path + "product_category_name_translation.csv")
order_items  = pd.read_csv(path + "olist_order_items_dataset.csv")

# ── Jointure produit + traduction ─────────────────────────────────────────────
products = products.merge(translation, on="product_category_name", how="left")

# ── Garder une seule ligne par commande (premier item) ────────────────────────
items_product = (
    order_items
    .sort_values("order_item_id")
    .drop_duplicates(subset="order_id", keep="first")
)[["order_id", "product_id"]]

# ── Jointure avec les infos produit ──────────────────────────────────────────
items_product = items_product.merge(
    products[["product_id", "product_category_name_english", "product_weight_g"]],
    on="product_id",
    how="left"
)

# ── Intégration dans df_master ────────────────────────────────────────────────
df_master = df_master.merge(items_product, on="order_id", how="left")

# ── Sauvegarde finale ─────────────────────────────────────────────────────────
df_master.to_csv("/Users/inasnanat/Desktop/Mémoire/code/master_table.csv", index=False)
print(f"✅ master_table.csv sauvegardée — {df_master.shape[0]} lignes, {df_master.shape[1]} colonnes")

# ── Split temporel ────────────────────────────────────────────────────────────
df_master = df_master.sort_values("order_purchase_timestamp").reset_index(drop=True)

n = len(df_master)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train = df_master.iloc[:train_end].copy()
val   = df_master.iloc[train_end:val_end].copy()
test  = df_master.iloc[val_end:].copy()

print(f"Train : {len(train)} lignes ({len(train)/n:.1%}) — churn : {train['churn'].mean():.1%}")
print(f"Val   : {len(val)} lignes ({len(val)/n:.1%}) — churn : {val['churn'].mean():.1%}")
print(f"Test  : {len(test)} lignes ({len(test)/n:.1%}) — churn : {test['churn'].mean():.1%}")

print(f"\nTrain : {train['order_purchase_timestamp'].min().date()} → {train['order_purchase_timestamp'].max().date()}")
print(f"Val   : {val['order_purchase_timestamp'].min().date()} → {val['order_purchase_timestamp'].max().date()}")
print(f"Test  : {test['order_purchase_timestamp'].min().date()} → {test['order_purchase_timestamp'].max().date()}")

# ── Sauvegarde des splits ──────────────────────────────────────────────────────
save_path = "/Users/inasnanat/Desktop/Mémoire/code/"
train.to_csv(save_path + "train.csv", index=False)
val.to_csv(save_path + "val.csv", index=False)
test.to_csv(save_path + "test.csv", index=False)
print("\n✅ Splits sauvegardés")