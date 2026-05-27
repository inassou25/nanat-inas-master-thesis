import os
import pandas as pd
import numpy as np
import json
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# BLOC 1 — CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
val        = pd.read_csv(save_path + "split_val.csv",
                         parse_dates=["order_purchase_timestamp"])

# Train M1/M2 : tout le train pool
# Train M3    : 1500 premières observations

train_m3, _ = train_test_split(
    train_pool, train_size=1500,
    stratify=train_pool["churn"], random_state=42
)
train_m3 = train_m3.sort_values("order_purchase_timestamp").reset_index(drop=True)

print(f"Train pool : {len(train_pool)} obs | churn : {train_pool['churn'].mean():.1%}")
print(f"Val        : {len(val)} obs | churn : {val['churn'].mean():.1%}")
print(f"Train M3   : {len(train_m3)} obs | churn : {train_m3['churn'].mean():.1%}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 2 — FEATURES STRUCTURÉES
# ════════════════════════════════════════════════════════════════════════════
struct_features = [
    "recency_days", "frequency", "monetary",
    "n_items", "n_sellers", "freight_ratio",
    "delivery_delay_days", "is_late", "processing_time_days",
    "n_installments", "payment_value",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    "product_weight_g"
]

# Target encoding M1/M2 — calculé sur train_pool
state_means_12   = train_pool.groupby("customer_state")["churn"].mean()
cat_means_12     = train_pool.groupby("product_category_name_english")["churn"].mean()
train_median_12  = train_pool[struct_features].median()
churn_mean_12    = train_pool["churn"].mean()

# Target encoding M3 — calculé sur train_m3
state_means_3    = train_m3.groupby("customer_state")["churn"].mean()
cat_means_3      = train_m3.groupby("product_category_name_english")["churn"].mean()
train_median_3   = train_m3[struct_features].median()
churn_mean_3     = train_m3["churn"].mean()

def apply_encoding(df, state_means, cat_means):
    df = df.copy()
    df["state_encoded"]    = df["customer_state"].map(state_means)
    df["category_encoded"] = df["product_category_name_english"].map(cat_means)
    return df

def impute(df, median, churn_mean):
    df = df.copy()
    for col in struct_features:
        df[col] = df[col].fillna(median[col])
    df["state_encoded"]    = df["state_encoded"].fillna(churn_mean)
    df["category_encoded"] = df["category_encoded"].fillna(churn_mean)
    return df

feats1 = struct_features + ["state_encoded", "category_encoded"]

# ════════════════════════════════════════════════════════════════════════════
# BLOC 3 — FONCTION SEUIL OPTIMAL
# ════════════════════════════════════════════════════════════════════════════
def find_best_threshold(y_true, y_proba):
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    best_idx  = f1_scores.argmax()
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

# ════════════════════════════════════════════════════════════════════════════
# BLOC 4 — M1 : STRUCTURÉ SEUL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M1 — SEUIL OPTIMAL SUR VAL")
print("="*60)

# Charger hyperparamètres
with open(save_path + "hyperparams_m1_m2.json") as f:
    hp = json.load(f)

# Préparer features
train_m1 = apply_encoding(train_pool, state_means_12, cat_means_12)
val_m1   = apply_encoding(val,        state_means_12, cat_means_12)
train_m1 = impute(train_m1, train_median_12, churn_mean_12)
val_m1   = impute(val_m1,   train_median_12, churn_mean_12)

X_train_m1 = train_m1[feats1].astype(float).values
X_val_m1   = val_m1[feats1].astype(float).values
y_train_m1 = train_m1["churn"].values
y_val_m1   = val_m1["churn"].values

# LR
sc1_lr      = StandardScaler()
X_tr1_sc    = sc1_lr.fit_transform(X_train_m1)
X_v1_sc     = sc1_lr.transform(X_val_m1)
lr1 = LogisticRegression(class_weight="balanced", max_iter=1000,
                          random_state=42, C=hp["M1_LR"]["C"])
lr1.fit(X_tr1_sc, y_train_m1)
proba_val_m1_lr  = lr1.predict_proba(X_v1_sc)[:, 1]
thresh_m1_lr, f1_m1_lr = find_best_threshold(y_val_m1, proba_val_m1_lr)
print(f"M1 LR  → seuil optimal : {thresh_m1_lr:.3f} | F1 val : {f1_m1_lr:.4f}")

# XGB
spw1 = (y_train_m1==0).sum() / (y_train_m1==1).sum()
xgb1_p = hp["M1_XGB"]
xgb1 = XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=xgb1_p["max_depth"],
    min_child_weight=xgb1_p["min_child_weight"],
    subsample=xgb1_p["subsample"],
    colsample_bytree=xgb1_p["colsample_bytree"],
    reg_alpha=xgb1_p["reg_alpha"],
    reg_lambda=xgb1_p["reg_lambda"],
    scale_pos_weight=spw1,
    eval_metric="aucpr", random_state=42, verbosity=0, nthread=1
)
xgb1.fit(X_train_m1, y_train_m1)
proba_val_m1_xgb = xgb1.predict_proba(X_val_m1)[:, 1]
thresh_m1_xgb, f1_m1_xgb = find_best_threshold(y_val_m1, proba_val_m1_xgb)
print(f"M1 XGB → seuil optimal : {thresh_m1_xgb:.3f} | F1 val : {f1_m1_xgb:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 5 — M2 : STRUCTURÉ + EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M2 — SEUIL OPTIMAL SUR VAL")
print("="*60)

X_train_emb = np.load(save_path + "final_X_train_emb.npy")
X_val_emb   = np.load(save_path + "final_X_val_emb.npy")

sc2s = StandardScaler()
sc2e = StandardScaler()
X_tr2 = np.hstack([sc2s.fit_transform(X_train_m1), sc2e.fit_transform(X_train_emb)])
X_v2  = np.hstack([sc2s.transform(X_val_m1),       sc2e.transform(X_val_emb)])

# LR
lr2 = LogisticRegression(class_weight="balanced", max_iter=1000,
                          random_state=42, C=hp["M2_LR"]["C"])
lr2.fit(X_tr2, y_train_m1)
proba_val_m2_lr  = lr2.predict_proba(X_v2)[:, 1]
thresh_m2_lr, f1_m2_lr = find_best_threshold(y_val_m1, proba_val_m2_lr)
print(f"M2 LR  → seuil optimal : {thresh_m2_lr:.3f} | F1 val : {f1_m2_lr:.4f}")

# XGB
spw2  = (y_train_m1==0).sum() / (y_train_m1==1).sum()
xgb2_p = hp["M2_XGB"]
xgb2 = XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=xgb2_p["max_depth"],
    min_child_weight=xgb2_p["min_child_weight"],
    subsample=xgb2_p["subsample"],
    colsample_bytree=xgb2_p["colsample_bytree"],
    reg_alpha=xgb2_p["reg_alpha"],
    reg_lambda=xgb2_p["reg_lambda"],
    scale_pos_weight=spw2,
    eval_metric="aucpr", random_state=42, verbosity=0, nthread=1
)
xgb2.fit(X_tr2, y_train_m1)
proba_val_m2_xgb = xgb2.predict_proba(X_v2)[:, 1]
thresh_m2_xgb, f1_m2_xgb = find_best_threshold(y_val_m1, proba_val_m2_xgb)
print(f"M2 XGB → seuil optimal : {thresh_m2_xgb:.3f} | F1 val : {f1_m2_xgb:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 6 — M3 : STRUCTURÉ + RAG
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M3 — SEUIL OPTIMAL SUR VAL")
print("="*60)

with open(save_path + "hyperparams_m3.json") as f:
    hp3 = json.load(f)

train_rag = pd.read_csv(save_path + "m3_train_rag_stratified.csv")
val_rag   = pd.read_csv(save_path + "m3_val_rag.csv")

rag_features       = ["livraison","qualite_produit","service_client","prix"]
mentioned_features = [f'{d}_mentioned' for d in rag_features]
all_feats3         = feats1 + rag_features + mentioned_features

def build_m3_features(df, rag_df, state_means, cat_means, median, churn_mean):
    merged = df.merge(rag_df[["order_id"] + rag_features], on="order_id", how="inner")
    for dim in rag_features:
        merged[f'{dim}_mentioned'] = (merged[dim] != 0.5).astype(int)
    merged = apply_encoding(merged, state_means, cat_means)
    merged = impute(merged, median, churn_mean)
    for col in rag_features + mentioned_features:
        merged[col] = merged[col].fillna(0.5)
    return merged

train3 = build_m3_features(train_m3, train_rag, state_means_3, cat_means_3,
                            train_median_3, churn_mean_3)
val3   = build_m3_features(val,       val_rag,   state_means_3, cat_means_3,
                            train_median_3, churn_mean_3)

X_train3 = train3[all_feats3].astype(float).values
X_val3   = val3[all_feats3].astype(float).values
y_train3 = train3["churn"].values
y_val3   = val3["churn"].values

sc3         = StandardScaler()
X_train3_sc = sc3.fit_transform(X_train3)
X_val3_sc   = sc3.transform(X_val3)

# LR
lr3 = LogisticRegression(class_weight="balanced", max_iter=1000,
                          random_state=42, C=hp3["M3_LR"]["C"])
lr3.fit(X_train3_sc, y_train3)
proba_val_m3_lr  = lr3.predict_proba(X_val3_sc)[:, 1]
thresh_m3_lr, f1_m3_lr = find_best_threshold(y_val3, proba_val_m3_lr)
print(f"M3 LR  → seuil optimal : {thresh_m3_lr:.3f} | F1 val : {f1_m3_lr:.4f}")

# XGB
spw3  = (y_train3==0).sum() / (y_train3==1).sum()
xgb3_p = hp3["M3_XGB"]
xgb3 = XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=xgb3_p["max_depth"],
    min_child_weight=xgb3_p["min_child_weight"],
    subsample=xgb3_p["subsample"],
    colsample_bytree=xgb3_p["colsample_bytree"],
    reg_alpha=xgb3_p["reg_alpha"],
    reg_lambda=xgb3_p["reg_lambda"],
    scale_pos_weight=spw3,
    eval_metric="aucpr", random_state=42, verbosity=0, nthread=1
)
xgb3.fit(X_train3_sc, y_train3)
proba_val_m3_xgb = xgb3.predict_proba(X_val3_sc)[:, 1]
thresh_m3_xgb, f1_m3_xgb = find_best_threshold(y_val3, proba_val_m3_xgb)
print(f"M3 XGB → seuil optimal : {thresh_m3_xgb:.3f} | F1 val : {f1_m3_xgb:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 7 — M4 : ZERO-SHOT GPT
# 500 dernières observations du val — respect de la temporalité
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M4 — ZERO-SHOT GPT — SEUIL OPTIMAL SUR VAL (500 obs)")
print("="*60)

val_zs = val.iloc[-500:].copy()
print(f"Val zero-shot : {len(val_zs)} obs | churn : {val_zs['churn'].mean():.1%}")

def call_openai_with_retry(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=10, timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Retry {attempt+1}/{max_retries} — attente {wait}s")
            time.sleep(wait)
    return None

def predict_churn_zeroshot(review_text):
    prompt = f"""You are analyzing a Brazilian e-commerce customer review to predict whether the customer will churn.

REVIEW: "{review_text[:300]}"

What is the probability (between 0.0 and 1.0) that this customer will NOT return to the platform?
- 0.0 = very likely to return (satisfied)
- 1.0 = very unlikely to return (dissatisfied)

Reply ONLY with a single number between 0.0 and 1.0, nothing else."""

    response = call_openai_with_retry(prompt)
    if response is None:
        return 0.5
    try:
        return min(max(float(response.strip()), 0.0), 1.0)
    except:
        return 0.5

def prepare_text(row):
    text = str(row["review_comment_message"])
    if pd.notna(row["review_comment_title"]):
        text = str(row["review_comment_title"]) + " " + text
    return text

# Reprise automatique
zs_val_path = save_path + "m4_val_zeroshot.csv"
if os.path.exists(zs_val_path):
    done_df  = pd.read_csv(zs_val_path)
    done_ids = set(done_df["order_id"].tolist())
    print(f"Déjà traités : {len(done_ids)} / {len(val_zs)}")
else:
    done_df  = pd.DataFrame()
    done_ids = set()

remaining = val_zs[~val_zs["order_id"].isin(done_ids)].copy()
print(f"Restants : {len(remaining)}")

results_zs = []
for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Zero-shot val"):
    score = predict_churn_zeroshot(prepare_text(row))
    results_zs.append({"order_id": row["order_id"],
                        "churn": row["churn"],
                        "churn_proba": score})

    if len(results_zs) % 50 == 0:
        df_partial = pd.DataFrame(results_zs)
        if not done_df.empty:
            df_partial = pd.concat([done_df, df_partial], ignore_index=True)
        df_partial.to_csv(zs_val_path, index=False)

df_new = pd.DataFrame(results_zs)
if not done_df.empty:
    val_zs_results = pd.concat([done_df, df_new], ignore_index=True)
else:
    val_zs_results = df_new
val_zs_results.to_csv(zs_val_path, index=False)
print(f"✅ Zero-shot val sauvegardé : {len(val_zs_results)} observations")

y_val_zs    = val_zs_results["churn"].values
proba_val_zs = val_zs_results["churn_proba"].values
thresh_m4, f1_m4 = find_best_threshold(y_val_zs, proba_val_zs)
print(f"M4 ZS  → seuil optimal : {thresh_m4:.3f} | F1 val : {f1_m4:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 8 — SAUVEGARDE DES SEUILS
# ════════════════════════════════════════════════════════════════════════════
thresholds = {
    "M1_LR":  {"threshold": thresh_m1_lr,  "f1_val": f1_m1_lr},
    "M1_XGB": {"threshold": thresh_m1_xgb, "f1_val": f1_m1_xgb},
    "M2_LR":  {"threshold": thresh_m2_lr,  "f1_val": f1_m2_lr},
    "M2_XGB": {"threshold": thresh_m2_xgb, "f1_val": f1_m2_xgb},
    "M3_LR":  {"threshold": thresh_m3_lr,  "f1_val": f1_m3_lr},
    "M3_XGB": {"threshold": thresh_m3_xgb, "f1_val": f1_m3_xgb},
    "M4_ZS":  {"threshold": thresh_m4,     "f1_val": f1_m4},
}
with open(save_path + "optimal_thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

print("\n" + "="*60)
print("RÉCAPITULATIF — SEUILS OPTIMAUX SUR VAL")
print("="*60)
for name, vals in thresholds.items():
    print(f"{name:8} → seuil : {vals['threshold']:.3f} | F1 val : {vals['f1_val']:.4f}")
print("\n✅ Seuils sauvegardés : optimal_thresholds.json")