import sys
sys.path.append("/Users/inasnanat/Desktop/Mémoire/code/")

import pandas as pd
import numpy as np
import faiss
import json
import os
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from knowledge_base import knowledge_base

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
save_path = "/Users/inasnanat/Desktop/Mémoire/code/"

# Chargement — on utilise le train pool
train_pool = pd.read_csv(save_path + "split_train_pool.csv",
                         parse_dates=["order_purchase_timestamp"])


sample, _ = train_test_split(
    train_pool, train_size=200,
    stratify=train_pool["churn"], random_state=42
)
print(f"Échantillon : {len(sample)} reviews | churn : {sample['churn'].mean():.1%}")

# Modèle embedding + FAISS
print("Chargement modèle embedding...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",
                                local_files_only=True)

kb_texts      = [p["text"] for p in knowledge_base]
kb_embeddings = embedder.encode(kb_texts, show_progress_bar=False)
kb_embeddings = kb_embeddings / np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
dimension     = kb_embeddings.shape[1]
index_faiss   = faiss.IndexFlatIP(dimension)
index_faiss.add(kb_embeddings.astype(np.float32))
print(f"Index FAISS : {index_faiss.ntotal} passages")

def retrieve(query, k):
    query_emb = embedder.encode([query])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    scores, indices = index_faiss.search(query_emb.astype(np.float32), k)
    return [knowledge_base[i] for i in indices[0]]

def safe_float(val, default=0.5):
    try:
        return float(val) if val is not None else default
    except:
        return default

def call_openai_with_retry(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=50, timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def extract_rag_scores(review_text, k):
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
        return {d: 0.5 for d in ["livraison","qualite_produit","service_client","prix"]}

    response = response.replace("```json","").replace("```","").strip()
    response = response.replace(": null",": 0.5").replace(":null",":0.5")

    try:
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("{") and "livraison" in line:
                raw = json.loads(line)
                return {d: safe_float(raw.get(d))
                        for d in ["livraison","qualite_produit","service_client","prix"]}
        start = response.find("{")
        end   = response.rfind("}") + 1
        raw   = json.loads(response[start:end])
        return {d: safe_float(raw.get(d))
                for d in ["livraison","qualite_produit","service_client","prix"]}
    except:
        return {d: 0.5 for d in ["livraison","qualite_produit","service_client","prix"]}

def prepare_text(row):
    text = str(row["review_comment_message"])
    if pd.notna(row["review_comment_title"]):
        text = str(row["review_comment_title"]) + " " + text
    return text

# Valeurs de k à tester
k_values = [1, 2, 3, 5, 7, 10]
results  = []

for k in k_values:
    print(f"\n{'='*40}")
    print(f"Test k = {k}")

    scores_list = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"k={k}"):
        text   = prepare_text(row)
        scores = extract_rag_scores(text, k=k)
        scores["order_id"]     = row["order_id"]
        scores["review_score"] = row["review_score"]
        scores["churn"]        = row["churn"]
        scores_list.append(scores)

    df_k = pd.DataFrame(scores_list)

    rag_dims   = ["livraison","qualite_produit","service_client","prix"]
    row_result = {"k": k}

    print(f"\n  Corrélation avec review_score :")
    for dim in rag_dims:
        corr, pval = stats.pearsonr(df_k[dim], df_k["review_score"])
        row_result[f"corr_score_{dim}"] = round(corr, 3)
        print(f"    {dim:20} : r = {corr:.3f} (p = {pval:.2e})")

    print(f"\n  Corrélation avec churn :")
    for dim in rag_dims:
        corr, pval = stats.pointbiserialr(df_k["churn"], df_k[dim])
        row_result[f"corr_churn_{dim}"] = round(corr, 3)
        print(f"    {dim:20} : r = {corr:.3f} (p = {pval:.2e})")

    df_k.to_csv(save_path + f"sensitivity_k{k}_scores.csv", index=False)
    results.append(row_result)

# Tableau récapitulatif
df_results = pd.DataFrame(results)
print(f"\n{'='*60}")
print("TABLEAU RÉCAPITULATIF")
print(f"{'='*60}")
print(df_results.to_string(index=False))
df_results.to_csv(save_path + "sensitivity_k_results.csv", index=False)

# K optimal
df_results["mean_corr_score"] = df_results[
    [f"corr_score_{dim}" for dim in rag_dims]
].mean(axis=1)
best_k = df_results.loc[df_results["mean_corr_score"].idxmax(), "k"]
print(f"\n✅ Meilleur k : k = {best_k}")