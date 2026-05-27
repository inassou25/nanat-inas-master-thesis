import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import json
import time
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_recall_curve, f1_score,
                              recall_score, precision_score)
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tqdm import tqdm
from knowledge_base import knowledge_base

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# ════════════════════════════════════════════════════════════════════════════
# BLOC 1 — CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])
val        = pd.read_csv(save_path + "split_val.csv",
                         parse_dates=["order_purchase_timestamp"])
test       = pd.read_csv(save_path + "split_test.csv",
                         parse_dates=["order_purchase_timestamp"])

# Train M3 — 1500 obs stratifiées
train_m3, _ = train_test_split(
    train_pool, train_size=1500,
    stratify=train_pool["churn"], random_state=42
)
train_m3 = train_m3.sort_values("order_purchase_timestamp").reset_index(drop=True)

print(f"Train pool : {len(train_pool)} | churn : {train_pool['churn'].mean():.1%}")
print(f"Train M3   : {len(train_m3)} | churn : {train_m3['churn'].mean():.1%}")
print(f"Val        : {len(val)} | churn : {val['churn'].mean():.1%}")
print(f"Test       : {len(test)} | churn : {test['churn'].mean():.1%}")

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

# Target encoding M1/M2 — sur train_pool
state_means_12  = train_pool.groupby("customer_state")["churn"].mean()
cat_means_12    = train_pool.groupby("product_category_name_english")["churn"].mean()
train_median_12 = train_pool[struct_features].median()
churn_mean_12   = train_pool["churn"].mean()

# Target encoding M3 — sur train_m3
state_means_3   = train_m3.groupby("customer_state")["churn"].mean()
cat_means_3     = train_m3.groupby("product_category_name_english")["churn"].mean()
train_median_3  = train_m3[struct_features].median()
churn_mean_3    = train_m3["churn"].mean()

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

# Préparer train/val/test pour M1/M2
train_12 = impute(apply_encoding(train_pool, state_means_12, cat_means_12),
                  train_median_12, churn_mean_12)
val_12   = impute(apply_encoding(val,        state_means_12, cat_means_12),
                  train_median_12, churn_mean_12)
test_12  = impute(apply_encoding(test,       state_means_12, cat_means_12),
                  train_median_12, churn_mean_12)

X_train_12 = train_12[feats1].astype(float).values
X_val_12   = val_12[feats1].astype(float).values
X_test_12  = test_12[feats1].astype(float).values
y_train_12 = train_pool["churn"].values
y_val_12   = val["churn"].values
y_test     = test["churn"].values

# ════════════════════════════════════════════════════════════════════════════
# BLOC 3 — CHARGEMENT HYPERPARAMÈTRES ET SEUILS
# ════════════════════════════════════════════════════════════════════════════
with open(save_path + "hyperparams_m1_m2.json") as f:
    hp12 = json.load(f)
with open(save_path + "hyperparams_m3.json") as f:
    hp3 = json.load(f)
with open(save_path + "optimal_thresholds.json") as f:
    thresholds = json.load(f)

print("\n✅ Hyperparamètres et seuils chargés")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 4 — FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════
def evaluate_test(y_true, y_proba, threshold, label):
    roc   = roc_auc_score(y_true, y_proba)
    prauc = average_precision_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    print(f"\n=== {label} ===")
    print(f"ROC-AUC   : {roc:.4f}")
    print(f"PR-AUC    : {prauc:.4f}")
    print(f"F1        : {f1:.4f} (seuil val : {threshold:.3f})")
    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")
    return {"label": label, "roc_auc": roc, "pr_auc": prauc,
            "f1": f1, "recall": rec, "precision": prec,
            "threshold": threshold}

def bootstrap_prauc(y_true, y_proba, label, n=1000):
    scores = []
    for _ in range(n):
        idx = resample(range(len(y_true)), random_state=None)
        try:
            scores.append(average_precision_score(
                np.array(y_true)[idx], np.array(y_proba)[idx]))
        except:
            pass
    mean = np.mean(scores)
    ci   = np.std(scores) * 1.96
    print(f"  Bootstrap PR-AUC ({label}) : {mean:.3f} ± {ci:.3f} [IC 95%]")
    return mean, ci

def permutation_test(y_true, proba1, proba2, label, n=1000):
    obs_diff = average_precision_score(y_true, proba2) - \
               average_precision_score(y_true, proba1)
    count = 0
    for _ in range(n):
        mask = np.random.rand(len(y_true)) > 0.5
        p = np.where(mask, proba1, proba2)
        q = np.where(mask, proba2, proba1)
        count += (average_precision_score(y_true, q) -
                  average_precision_score(y_true, p)) >= obs_diff
    p_value = count / n
    print(f"  {label} : Δ={obs_diff:+.4f} | p={p_value:.3f}")
    return obs_diff, p_value

all_results  = []
all_probas   = {}  # pour calibration curve et robustesse

# ════════════════════════════════════════════════════════════════════════════
# BLOC 5 — M1 : STRUCTURÉ SEUL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M1 — STRUCTURÉ SEUL")
print("="*60)

spw1 = (y_train_12==0).sum() / (y_train_12==1).sum()

# LR
sc1_lr      = StandardScaler()
X_tr1_sc    = sc1_lr.fit_transform(X_train_12)
X_v1_sc     = sc1_lr.transform(X_val_12)
X_te1_sc    = sc1_lr.transform(X_test_12)

lr1 = LogisticRegression(class_weight="balanced", max_iter=1000,
                          random_state=42, C=hp12["M1_LR"]["C"])
lr1.fit(X_tr1_sc, y_train_12)
proba_test_m1_lr = lr1.predict_proba(X_te1_sc)[:, 1]
res = evaluate_test(y_test, proba_test_m1_lr,
                    thresholds["M1_LR"]["threshold"], "TEST — M1 LR")
all_results.append(res)
all_probas["M1_LR"] = proba_test_m1_lr

# XGB
xgb1_p = hp12["M1_XGB"]
xgb1   = XGBClassifier(
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
xgb1.fit(X_train_12, y_train_12)
proba_test_m1_xgb = xgb1.predict_proba(X_test_12)[:, 1]
res = evaluate_test(y_test, proba_test_m1_xgb,
                    thresholds["M1_XGB"]["threshold"], "TEST — M1 XGB")
all_results.append(res)
all_probas["M1_XGB"] = proba_test_m1_xgb

# ════════════════════════════════════════════════════════════════════════════
# BLOC 6 — M2 : STRUCTURÉ + EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M2 — STRUCTURÉ + EMBEDDINGS")
print("="*60)

X_train_emb = np.load(save_path + "final_X_train_emb.npy")
X_val_emb   = np.load(save_path + "final_X_val_emb.npy")

print("Génération embeddings test...")
embedder    = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",
                                   local_files_only=True)

def prepare_text(df):
    title   = df["review_comment_title"].fillna("")
    message = df["review_comment_message"].fillna("")
    return (title + " " + message).str.strip().tolist()

X_test_emb = embedder.encode(prepare_text(test), batch_size=64,
                              show_progress_bar=True)
np.save(save_path + "final_X_test_emb.npy", X_test_emb)
print("✅ Embeddings test sauvegardés")

sc2s = StandardScaler()
sc2e = StandardScaler()
X_tr2  = np.hstack([sc2s.fit_transform(X_train_12), sc2e.fit_transform(X_train_emb)])
X_v2   = np.hstack([sc2s.transform(X_val_12),       sc2e.transform(X_val_emb)])
X_te2  = np.hstack([sc2s.transform(X_test_12),      sc2e.transform(X_test_emb)])
spw2   = (y_train_12==0).sum() / (y_train_12==1).sum()

# LR
lr2 = LogisticRegression(class_weight="balanced", max_iter=1000,
                          random_state=42, C=hp12["M2_LR"]["C"])
lr2.fit(X_tr2, y_train_12)
proba_test_m2_lr = lr2.predict_proba(X_te2)[:, 1]
res = evaluate_test(y_test, proba_test_m2_lr,
                    thresholds["M2_LR"]["threshold"], "TEST — M2 LR")
all_results.append(res)
all_probas["M2_LR"] = proba_test_m2_lr

# XGB
xgb2_p = hp12["M2_XGB"]
xgb2   = XGBClassifier(
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
xgb2.fit(X_tr2, y_train_12)
proba_test_m2_xgb = xgb2.predict_proba(X_te2)[:, 1]
res = evaluate_test(y_test, proba_test_m2_xgb,
                    thresholds["M2_XGB"]["threshold"], "TEST — M2 XGB")
all_results.append(res)
all_probas["M2_XGB"] = proba_test_m2_xgb

# ════════════════════════════════════════════════════════════════════════════
# BLOC 7 — M3 : STRUCTURÉ + RAG
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M3 — STRUCTURÉ + RAG")
print("="*60)

rag_features       = ["livraison","qualite_produit","service_client","prix"]
mentioned_features = [f'{d}_mentioned' for d in rag_features]
all_feats3         = feats1 + rag_features + mentioned_features

train_rag = pd.read_csv(save_path + "m3_train_rag_stratified.csv")
val_rag   = pd.read_csv(save_path + "m3_val_rag.csv")

def build_m3(df, rag_df, state_means, cat_means, median, churn_mean):
    merged = df.merge(rag_df[["order_id"] + rag_features],
                      on="order_id", how="inner")
    for dim in rag_features:
        merged[f'{dim}_mentioned'] = (merged[dim] != 0.5).astype(int)
    merged = apply_encoding(merged, state_means, cat_means)
    merged = impute(merged, median, churn_mean)
    for col in rag_features + mentioned_features:
        merged[col] = merged[col].fillna(0.5)
    return merged

train3 = build_m3(train_m3, train_rag, state_means_3,
                  cat_means_3, train_median_3, churn_mean_3)
val3   = build_m3(val, val_rag, state_means_3,
                  cat_means_3, train_median_3, churn_mean_3)

# Extraction RAG test — avec reprise automatique
print("\nExtraction RAG test (5832 reviews)...")

kb_texts      = [p["text"] for p in knowledge_base]
emb_kb        = embedder.encode(kb_texts, show_progress_bar=False)
emb_kb        = emb_kb / np.linalg.norm(emb_kb, axis=1, keepdims=True)
index_faiss   = faiss.IndexFlatIP(emb_kb.shape[1])
index_faiss.add(emb_kb.astype(np.float32))

def retrieve(query, k=3):
    q = embedder.encode([query])
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    _, idx = index_faiss.search(q.astype(np.float32), k)
    return [knowledge_base[i] for i in idx[0]]

def safe_float(v, default=0.5):
    try:
        return float(v) if v is not None else default
    except:
        return default

def call_openai(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=50, timeout=30
            )
            return r.choices[0].message.content
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def extract_rag(text, k=3):
    retrieved = retrieve(text, k=k)
    context   = "\n".join([f"- [{p['dimension']}] {p['text'][:200]}"
                            for p in retrieved])
    prompt = f"""You are analyzing a Brazilian e-commerce customer review.
Based on the academic context below, score the review on 4 dimensions from 0.0 (very negative) to 1.0 (very positive). Use 0.5 if the dimension is not mentioned.

ACADEMIC CONTEXT:
{context}

REVIEW: "{text[:200]}"

Reply ONLY with a JSON object on one line, no explanation:
{{"livraison":0.5,"qualite_produit":0.5,"service_client":0.5,"prix":0.5}}"""

    resp = call_openai(prompt)
    if resp is None:
        return {d: 0.5 for d in rag_features}
    resp = resp.replace("```json","").replace("```","").strip()
    resp = resp.replace(": null",": 0.5").replace(":null",":0.5")
    try:
        import json as json_lib
        for line in resp.split("\n"):
            line = line.strip()
            if line.startswith("{") and "livraison" in line:
                raw = json_lib.loads(line)
                return {d: safe_float(raw.get(d)) for d in rag_features}
        raw = json_lib.loads(resp[resp.find("{"):resp.rfind("}")+1])
        return {d: safe_float(raw.get(d)) for d in rag_features}
    except:
        return {d: 0.5 for d in rag_features}

def prepare_row(row):
    text = str(row["review_comment_message"])
    if pd.notna(row["review_comment_title"]):
        text = str(row["review_comment_title"]) + " " + text
    return text

def extract_batch(df, split_name, save_csv):
    if os.path.exists(save_csv):
        done_df  = pd.read_csv(save_csv)
        done_ids = set(done_df["order_id"].tolist())
        print(f"  Déjà traités : {len(done_ids)} / {len(df)}")
    else:
        done_df  = pd.DataFrame()
        done_ids = set()
    remaining = df[~df["order_id"].isin(done_ids)].copy()
    if len(remaining) == 0:
        print(f"  ✅ {split_name} déjà complet")
        return done_df
    results = []
    for _, row in tqdm(remaining.iterrows(), total=len(remaining),
                       desc=f"RAG {split_name}"):
        scores             = extract_rag(prepare_row(row))
        scores["order_id"] = row["order_id"]
        scores["churn"]    = row["churn"]
        results.append(scores)
        if len(results) % 100 == 0:
            tmp = pd.DataFrame(results)
            if not done_df.empty:
                tmp = pd.concat([done_df, tmp], ignore_index=True)
            tmp.to_csv(save_csv, index=False)
    final = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True) \
            if not done_df.empty else pd.DataFrame(results)
    final.to_csv(save_csv, index=False)
    print(f"  ✅ {split_name} complet : {len(final)} obs")
    return final

test_rag = extract_batch(test, "test", save_path + "m3_test_rag.csv")

test3 = build_m3(test, test_rag, state_means_3,
                 cat_means_3, train_median_3, churn_mean_3)

X_train3 = train3[all_feats3].astype(float).values
X_val3   = val3[all_feats3].astype(float).values
X_test3  = test3[all_feats3].astype(float).values
y_train3 = train3["churn"].values
y_test3  = test3["churn"].values

sc3         = StandardScaler()
X_train3_sc = sc3.fit_transform(X_train3)
X_val3_sc   = sc3.transform(X_val3)
X_test3_sc  = sc3.transform(X_test3)
spw3        = (y_train3==0).sum() / (y_train3==1).sum()

# LR
lr3 = LogisticRegression(class_weight="balanced", max_iter=1000,
                          random_state=42, C=hp3["M3_LR"]["C"])
lr3.fit(X_train3_sc, y_train3)
proba_test_m3_lr = lr3.predict_proba(X_test3_sc)[:, 1]
res = evaluate_test(y_test3, proba_test_m3_lr,
                    thresholds["M3_LR"]["threshold"], "TEST — M3 LR")
all_results.append(res)
all_probas["M3_LR"] = proba_test_m3_lr

# XGB
xgb3_p = hp3["M3_XGB"]
xgb3   = XGBClassifier(
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
proba_test_m3_xgb = xgb3.predict_proba(X_test3_sc)[:, 1]
res = evaluate_test(y_test3, proba_test_m3_xgb,
                    thresholds["M3_XGB"]["threshold"], "TEST — M3 XGB")
all_results.append(res)
all_probas["M3_XGB"] = proba_test_m3_xgb

# Feature importance XGBoost M3
print("\n=== Feature importance XGBoost M3 ===")
for feat, imp in sorted(zip(all_feats3, xgb3.feature_importances_),
                         key=lambda x: -x[1])[:10]:
    print(f"  {feat} : {imp:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 8 — M4 : ZERO-SHOT GPT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M4 — ZERO-SHOT GPT")
print("="*60)

def predict_zeroshot(text):
    prompt = f"""You are analyzing a Brazilian e-commerce customer review to predict whether the customer will churn.

REVIEW: "{text[:300]}"

What is the probability (between 0.0 and 1.0) that this customer will NOT return to the platform?
- 0.0 = very likely to return (satisfied)
- 1.0 = very unlikely to return (dissatisfied)

Reply ONLY with a single number between 0.0 and 1.0, nothing else."""
    resp = call_openai(prompt)
    if resp is None:
        return 0.5
    try:
        return min(max(float(resp.strip()), 0.0), 1.0)
    except:
        return 0.5

zs_path = save_path + "m4_test_zeroshot.csv"
if os.path.exists(zs_path):
    done_zs  = pd.read_csv(zs_path)
    done_ids = set(done_zs["order_id"].tolist())
    print(f"Déjà traités : {len(done_ids)} / {len(test)}")
else:
    done_zs  = pd.DataFrame()
    done_ids = set()

remaining_zs = test[~test["order_id"].isin(done_ids)].copy()
results_zs   = []

for _, row in tqdm(remaining_zs.iterrows(), total=len(remaining_zs),
                   desc="Zero-shot test"):
    score = predict_zeroshot(prepare_row(row))
    results_zs.append({"order_id": row["order_id"],
                        "churn": row["churn"],
                        "churn_proba": score})
    if len(results_zs) % 100 == 0:
        tmp = pd.DataFrame(results_zs)
        if not done_zs.empty:
            tmp = pd.concat([done_zs, tmp], ignore_index=True)
        tmp.to_csv(zs_path, index=False)

df_zs = pd.DataFrame(results_zs)
final_zs = pd.concat([done_zs, df_zs], ignore_index=True) \
           if not done_zs.empty else df_zs
final_zs.to_csv(zs_path, index=False)
print(f"✅ Zero-shot test sauvegardé : {len(final_zs)} obs")

y_test_zs    = final_zs["churn"].values
proba_test_zs = final_zs["churn_proba"].values
res = evaluate_test(y_test_zs, proba_test_zs,
                    thresholds["M4_ZS"]["threshold"], "TEST — M4 Zero-shot")
all_results.append(res)
all_probas["M4_ZS"] = proba_test_zs

# ════════════════════════════════════════════════════════════════════════════
# BLOC 9 — TABLEAU COMPARATIF COMPLET
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TABLEAU COMPARATIF COMPLET — TEST SET")
print("="*60)
df_results = pd.DataFrame(all_results)
print(df_results.to_string(index=False))

print("\n" + "="*60)
print("MEILLEURS MODÈLES PAR CONDITION")
print("="*60)
best = {
    "M1": max([r for r in all_results if "M1" in r["label"]],
              key=lambda x: x["pr_auc"]),
    "M2": max([r for r in all_results if "M2" in r["label"]],
              key=lambda x: x["pr_auc"]),
    "M3": max([r for r in all_results if "M3" in r["label"]],
              key=lambda x: x["pr_auc"]),
    "M4": next(r for r in all_results if "M4" in r["label"])
}
for k, v in best.items():
    print(f"{k} → {v['label']} | PR-AUC : {v['pr_auc']:.4f} | "
          f"F1 : {v['f1']:.4f} | ROC-AUC : {v['roc_auc']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 10 — BOOTSTRAP IC 95%
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BOOTSTRAP PR-AUC IC 95%")
print("="*60)
bootstrap_res = {}
for label, proba in all_probas.items():
    y = y_test if "M4" not in label else y_test_zs
    if "M3" in label:
        y = y_test3
    mean, ci = bootstrap_prauc(y, proba, label)
    bootstrap_res[label] = (mean, ci)

# ════════════════════════════════════════════════════════════════════════════
# BLOC 11 — TESTS DE PERMUTATION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TESTS DE PERMUTATION — SIGNIFICATIVITÉ")
print("="*60)
permutation_test(y_test, all_probas["M1_XGB"], all_probas["M2_LR"],
                 "M1 XGB → M2 LR (gain embeddings)")
permutation_test(y_test3, all_probas["M1_XGB"][:len(y_test3)],
                 all_probas["M3_XGB"],
                 "M1 XGB → M3 XGB (gain RAG)")
permutation_test(y_test3, all_probas["M2_LR"][:len(y_test3)],
                 all_probas["M3_XGB"],
                 "M2 LR → M3 XGB (embeddings vs RAG)")

# ════════════════════════════════════════════════════════════════════════════
# BLOC 12 — SAUVEGARDE FINALE
# ════════════════════════════════════════════════════════════════════════════
df_results.to_csv(save_path + "final_results.csv", index=False)

probas_df = pd.DataFrame({
    "order_id":    test["order_id"].values,
    "churn":       y_test,
    "M1_LR":       all_probas["M1_LR"],
    "M1_XGB":      all_probas["M1_XGB"],
    "M2_LR":       all_probas["M2_LR"],
    "M2_XGB":      all_probas["M2_XGB"],
})
probas_df.to_csv(save_path + "test_probas_m1_m2.csv", index=False)

probas_m3_df = pd.DataFrame({
    "order_id": test3["order_id"].values,
    "churn":    y_test3,
    "M3_LR":    all_probas["M3_LR"],
    "M3_XGB":   all_probas["M3_XGB"],
})
probas_m3_df.to_csv(save_path + "test_probas_m3.csv", index=False)

probas_zs_df = final_zs[["order_id","churn","churn_proba"]].copy()
probas_zs_df.to_csv(save_path + "test_probas_m4.csv", index=False)

print("\n✅ Résultats sauvegardés : final_results.csv")
print("✅ Probabilités sauvegardées : test_probas_m1_m2.csv / test_probas_m3.csv / test_probas_m4.csv")