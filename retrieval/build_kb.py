# retrieval/build_kb.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
import networkx as nx

# ---------- robust paths ----------
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJ_ROOT / "data" / "processed"
RETR_DIR = PROJ_ROOT / "data" / "artifacts" / "retrieval"
RETR_DIR.mkdir(parents=True, exist_ok=True)

def load_split(name: str) -> List[Dict[str, Any]]:
    p = DATA_PROCESSED / f"{name}.json"
    print(f"[INFO] Loading {p} ...")
    return json.loads(p.read_text(encoding="utf-8"))

def make_doc(record: Dict[str, Any]) -> Dict[str, Any]:
    title = record.get("subject", "") or ""
    product = record.get("product", "") or ""
    category = record.get("category", "") or ""
    subcat = record.get("subcategory", "") or ""
    tags = " ".join(record.get("tags", []) or [])
    res_code = record.get("resolution_code", "") or ""
    kbs = " ".join(record.get("kb_articles_helpful", []) or record.get("kb_articles_viewed", []) or [])
    resolution = record.get("resolution", "") or ""
    desc = record.get("description", "") or ""
    text = " ".join([title, product, category, subcat, tags, res_code, kbs, resolution, desc]).strip()

    return {
        "doc_id": record.get("ticket_id"),
        "product": product,
        "category": category,
        "subcategory": subcat,
        "resolution_code": res_code,
        "quality": float(record.get("resolution_helpful") is True),
        "text": text
    }

def build_docs() -> pd.DataFrame:
    all_recs = []
    for split in ["train", "val", "test"]:
        all_recs.extend(load_split(split))
    print(f"[INFO] Building documents from tickets...")
    docs = [make_doc(r) for r in all_recs if r.get("ticket_id")]
    df = pd.DataFrame(docs).dropna(subset=["doc_id"]).reset_index(drop=True)
    print(f"[INFO] Docs built: {len(df)}")
    return df

def build_tfidf(texts: List[str]):
    vec = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True
    )
    X = vec.fit_transform(texts)
    return X, vec

def save_index(X, vec, meta: pd.DataFrame):
    # save sparse matrix
    sparse.save_npz(RETR_DIR / "tfidf_index.npz", X)
    # save vectorizer
    joblib.dump(vec, RETR_DIR / "tfidf_vectorizer.joblib", compress=3)
    # save metadata as parquet (what /retrieve needs!)
    meta.to_parquet(RETR_DIR / "docs_meta.parquet", index=False)
    print(f"[OK] Saved index to {RETR_DIR}")

def build_graph(meta: pd.DataFrame):
    print("[INFO] Building Product-Issue-Solution graph...")
    G = nx.Graph()
    for _, row in meta.iterrows():
        prod = f"PRODUCT::{row['product']}"
        cat = f"CATEGORY::{row['category']}"
        sub = f"SUBCATEGORY::{row.get('subcategory') or ''}"
        sol = f"RESCODE::{row.get('resolution_code') or ''}"
        doc = f"DOC::{row['doc_id']}"
        for node in [prod, cat, sub, sol, doc]:
            if node and node not in G:
                G.add_node(node)
        # link relationships
        if prod and cat: G.add_edge(prod, cat)
        if cat and sub:  G.add_edge(cat, sub)
        if doc and sub:  G.add_edge(doc, sub)
        if doc and sol:  G.add_edge(doc, sol)
        if doc and prod: G.add_edge(doc, prod)
    out = RETR_DIR / "support_graph.graphml"
    nx.write_graphml(G, out)
    print(f"[OK] Graph saved: {out} (open in Gephi/Cytoscape)")

def main():
    df = build_docs()
    X, vec = build_tfidf(df["text"].fillna("").tolist())
    save_index(X, vec, df.drop(columns=["text"]))
    build_graph(df.drop(columns=["text"]))
    print("[DONE] KB build complete.")

if __name__ == "__main__":
    main()
