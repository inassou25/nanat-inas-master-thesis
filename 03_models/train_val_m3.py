import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import faiss
import json
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from tqdm import tqdm

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# BLOC 1 — CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
val        = pd.read_csv(save_path + "split_val.csv",
                         parse_dates=["order_purchase_timestamp"])

# 1500 premières observations chronologiques du train pool
from sklearn.model_selection import train_test_split
train, _ = train_test_split(
    train_pool, train_size=1500,
    stratify=train_pool["churn"], random_state=42
)
train = train.sort_values("order_purchase_timestamp").reset_index(drop=True)

print(f"Train M3 : {len(train)} obs | churn : {train['churn'].mean():.1%}")
print(f"Val      : {len(val)} obs | churn : {val['churn'].mean():.1%}")

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

# Target encoding — calculé sur train uniquement
state_means      = train.groupby("customer_state")["churn"].mean()
cat_means        = train.groupby("product_category_name_english")["churn"].mean()
train_median     = train[struct_features].median()
train_churn_mean = train["churn"].mean()

for df in [train, val]:
    df["state_encoded"]    = df["customer_state"].map(state_means)
    df["category_encoded"] = df["product_category_name_english"].map(cat_means)

feats1         = struct_features + ["state_encoded", "category_encoded"]
rag_features   = ["livraison","qualite_produit","service_client","prix"]
mentioned_features = [f'{d}_mentioned' for d in rag_features]
all_feats3     = feats1 + rag_features + mentioned_features

def impute_struct(df):
    df = df.copy()
    for col in struct_features:
        df[col] = df[col].fillna(train_median[col])
    df["state_encoded"]    = df["state_encoded"].fillna(train_churn_mean)
    df["category_encoded"] = df["category_encoded"].fillna(train_churn_mean)
    return df

# ════════════════════════════════════════════════════════════════════════════
# BLOC 3 — MODÈLE EMBEDDING + INDEX FAISS (k=3, justifié par sensitivity_k.py)
# ════════════════════════════════════════════════════════════════════════════
print("\nChargement modèle embedding...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

from knowledge_base import knowledge_base
kb_texts      = [p["text"] for p in knowledge_base]
kb_embeddings = embedder.encode(kb_texts, show_progress_bar=False)
kb_embeddings = kb_embeddings / np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
dimension     = kb_embeddings.shape[1]
index_faiss   = faiss.IndexFlatIP(dimension)
index_faiss.add(kb_embeddings.astype(np.float32))
print(f"Index FAISS : {index_faiss.ntotal} passages")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 4 — FONCTIONS RAG
# ════════════════════════════════════════════════════════════════════════════
def retrieve(query, k=3):
    query_emb = embedder.encode([query])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    scores, indices = index_faiss.search(query_emb.astype(np.float32), k)
    return [knowledge_base[i] for i in indices[0]]

def safe_float(val, default=0.5):
    if val is None:
        return default
    try:
        return float(val)
    except:
        return default

def call_openai_with_retry(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=50,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Retry {attempt+1}/{max_retries} — attente {wait}s")
            time.sleep(wait)
    return None

def extract_rag_scores(review_text, k=3):
    retrieved = retrieve(review_text, k=k)
    context   = "\n".join([
        f"- [{p['dimension']}] {p['text'][:200]}"
        for p in retrieved
    ])
    prompt = f"""You are analyzing a Brazilian e-commerce customer review.
Based on the academic context below, score the review on 4 dimensions from 0.0 (very negative) to 1.0 (very positive). Use 0.5 if the dimension is not mentioned.

ACADEMIC CONTEXT:
{context}

REVIEW: "{review_text[:200]}"

Reply ONLY with a JSON object on one line, no explanation:
{{"livraison":0.5,"qualite_produit":0.5,"service_client":0.5,"prix":0.5}}"""

    response = call_openai_with_retry(prompt)
    if response is None:
        return {"livraison":0.5,"qualite_produit":0.5,
                "service_client":0.5,"prix":0.5}

    response = response.replace("```json","").replace("```","").strip()
    response = response.replace(": null",": 0.5").replace(":null",":0.5")

    try:
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("{") and "livraison" in line:
                raw = json.loads(line)
                return {dim: safe_float(raw.get(dim))
                        for dim in ["livraison","qualite_produit","service_client","prix"]}
        start = response.find("{")
        end   = response.rfind("}") + 1
        raw   = json.loads(response[start:end])
        return {dim: safe_float(raw.get(dim))
                for dim in ["livraison","qualite_produit","service_client","prix"]}
    except Exception as e:
        print(f"PARSE ERROR: {e} | {response[:100]}")
        return {"livraison":0.5,"qualite_produit":0.5,
                "service_client":0.5,"prix":0.5}

def prepare_text(row):
    text = str(row["review_comment_message"])
    if pd.notna(row["review_comment_title"]):
        text = str(row["review_comment_title"]) + " " + text
    return text

def extract_scores_batch(df, split_name, save_csv):
    """Extraction RAG avec reprise automatique et sauvegarde toutes les 100 obs."""
    if os.path.exists(save_csv):
        done_df  = pd.read_csv(save_csv)
        done_ids = set(done_df["order_id"].tolist())
        print(f"  Déjà traités : {len(done_ids)} / {len(df)}")
    else:
        done_df  = pd.DataFrame()
        done_ids = set()

    remaining = df[~df["order_id"].isin(done_ids)].copy()
    print(f"  Restants : {len(remaining)}")

    if len(remaining) == 0:
        print(f"  ✅ {split_name} déjà complet — aucun appel GPT")
        return done_df

    results = []
    for _, row in tqdm(remaining.iterrows(), total=len(remaining),
                       desc=f"RAG {split_name}"):
        scores             = extract_rag_scores(prepare_text(row))
        scores["order_id"] = row["order_id"]
        scores["churn"]    = row["churn"]
        results.append(scores)

        if len(results) % 100 == 0:
            df_partial = pd.DataFrame(results)
            if not done_df.empty:
                df_partial = pd.concat([done_df, df_partial], ignore_index=True)
            df_partial.to_csv(save_csv, index=False)

    df_new = pd.DataFrame(results)
    final  = pd.concat([done_df, df_new], ignore_index=True) if not done_df.empty else df_new
    final.to_csv(save_csv, index=False)
    print(f"  ✅ {split_name} complet : {len(final)} observations")
    return final

# ════════════════════════════════════════════════════════════════════════════
# BLOC 5 — EXTRACTION RAG
# Train : 1500 obs → ~$1.50
# Val   : 5831 obs → ~$6
# Total : ~$7.50
# ════════════════════════════════════════════════════════════════════════════
print("\nExtraction RAG train (1500 reviews)...")
train_rag = extract_scores_batch(
    train, "train",
    save_path + "m3_train_rag_stratified.csv"
)

print("\nExtraction RAG val (5831 reviews)...")
val_rag = extract_scores_batch(
    val, "val",
    save_path + "m3_val_rag.csv"
)

# ════════════════════════════════════════════════════════════════════════════
# BLOC 6 — CONSTRUCTION DES FEATURES M3
# ════════════════════════════════════════════════════════════════════════════
def build_features(df, rag_df):
    merged = df.merge(rag_df[["order_id"] + rag_features],
                      on="order_id", how="inner")
    for dim in rag_features:
        merged[f'{dim}_mentioned'] = (merged[dim] != 0.5).astype(int)
    merged = impute_struct(merged)
    for col in rag_features + mentioned_features:
        merged[col] = merged[col].fillna(0.5)
    return merged

train3 = build_features(train, train_rag)
val3   = build_features(val,   val_rag)

X_train3 = train3[all_feats3].astype(float).values
X_val3   = val3[all_feats3].astype(float).values
y_train3 = train3["churn"].values
y_val3   = val3["churn"].values

print(f"Train3 : {len(train3)} obs | Val3 : {len(val3)} obs")
print(f"NaN train : {np.isnan(X_train3).sum()} | NaN val : {np.isnan(X_val3).sum()}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 7 — GRID SEARCH
# ════════════════════════════════════════════════════════════════════════════
param_grid_lr = [
    {"C": 1.0},
    {"C": 0.1},
    {"C": 0.01},
    {"C": 0.001},  
]

param_grid_xgb = [
    {"max_depth": 3, "min_child_weight": 5,  "reg_lambda": 2.0, "reg_alpha": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 3, "min_child_weight": 10, "reg_lambda": 5.0, "reg_alpha": 1.0, "subsample": 0.7, "colsample_bytree": 0.7},
    {"max_depth": 4, "min_child_weight": 5,  "reg_lambda": 2.0, "reg_alpha": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "min_child_weight": 10, "reg_lambda": 5.0, "reg_alpha": 1.0, "subsample": 0.7, "colsample_bytree": 0.7},
    {"max_depth": 2, "min_child_weight": 15, "reg_lambda": 10.0, "reg_alpha": 2.0, "subsample": 0.6, "colsample_bytree": 0.6},
]

spw = (y_train3==0).sum() / (y_train3==1).sum()

sc3          = StandardScaler()
X_train3_sc  = sc3.fit_transform(X_train3)
X_val3_sc    = sc3.transform(X_val3)

def overfitting_check(y_tr, p_tr, y_v, p_v, label):
    gap = average_precision_score(y_tr, p_tr) - average_precision_score(y_v, p_v)
    status = "⚠️ overfitting" if gap > 0.05 else "✅ ok"
    print(f"  [{label}] Train : {average_precision_score(y_tr, p_tr):.4f} | "
          f"Val : {average_precision_score(y_v, p_v):.4f} | Gap : {gap:.4f} {status}")

# LR
print("\n" + "="*60)
print("M3 — GRID SEARCH LR")
print("="*60)
best_lr_score, best_lr_params, best_lr = -1, None, None
for params in param_grid_lr:
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                             random_state=42, C=params["C"])
    lr.fit(X_train3_sc, y_train3)
    score = average_precision_score(y_val3, lr.predict_proba(X_val3_sc)[:, 1])
    print(f"  C={params['C']} → val PR-AUC : {score:.4f}")
    if score > best_lr_score:
        best_lr_score, best_lr_params, best_lr = score, params, lr
print(f"  → Meilleur : {best_lr_params} (val PR-AUC : {best_lr_score:.4f})")
overfitting_check(y_train3, best_lr.predict_proba(X_train3_sc)[:,1],
                  y_val3,   best_lr.predict_proba(X_val3_sc)[:,1], "M3 LR")

# XGB
print("\nM3 — GRID SEARCH XGB")
print("="*60)
best_xgb_score, best_xgb_params, best_xgb = -1, None, None
for params in param_grid_xgb:
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=spw,
        eval_metric="aucpr", random_state=42, verbosity=0
    )
    xgb.fit(X_train3_sc, y_train3)
    score = average_precision_score(y_val3, xgb.predict_proba(X_val3_sc)[:, 1])
    print(f"  depth={params['max_depth']} mcw={params['min_child_weight']} "
          f"λ={params['reg_lambda']} → val PR-AUC : {score:.4f}")
    if score > best_xgb_score:
        best_xgb_score, best_xgb_params, best_xgb = score, params, xgb
print(f"  → Meilleur : {best_xgb_params} (val PR-AUC : {best_xgb_score:.4f})")
overfitting_check(y_train3, best_xgb.predict_proba(X_train3_sc)[:,1],
                  y_val3,   best_xgb.predict_proba(X_val3_sc)[:,1], "M3 XGB")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 8 — RÉCAPITULATIF + SAUVEGARDE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RÉCAPITULATIF VAL — M3")
print("="*60)
print(f"M3 LR  : PR-AUC = {best_lr_score:.4f}  | ROC-AUC = {roc_auc_score(y_val3, best_lr.predict_proba(X_val3_sc)[:,1]):.4f} | params : {best_lr_params}")
print(f"M3 XGB : PR-AUC = {best_xgb_score:.4f}  | ROC-AUC = {roc_auc_score(y_val3, best_xgb.predict_proba(X_val3_sc)[:,1]):.4f} | params : {best_xgb_params}")

hyperparams_m3 = {
    "M3_LR":  best_lr_params,
    "M3_XGB": best_xgb_params,
    "sc3_mean": sc3.mean_.tolist(),
    "sc3_std":  sc3.scale_.tolist(),
}
with open(save_path + "hyperparams_m3.json", "w") as f:
    json.dump(hyperparams_m3, f, indent=2)
print("\n✅ Hyperparamètres sauvegardés : hyperparams_m3.json")
print("✅ Scores RAG sauvegardés : m3_train_rag.csv / m3_val_rag.csv")