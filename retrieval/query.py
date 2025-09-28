# retrieval/query.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from scipy import sparse
import joblib

# ---------- Robust absolute paths ----------
PROJ_ROOT = Path(__file__).resolve().parents[1]
RETR_DIR = PROJ_ROOT / "data" / "artifacts" / "retrieval"

# We will reuse the production inference loader (works in /classify)
from models.infer import load_deployed_model

def _load_retrieval_artifacts():
    """Load TF-IDF matrix, vectorizer, and document metadata."""
    X = sparse.load_npz(RETR_DIR / "tfidf_index.npz").tocsr()
    vec = joblib.load(RETR_DIR / "tfidf_vectorizer.joblib")
    meta = pd.read_parquet(RETR_DIR / "docs_meta.parquet")
    return X, vec, meta

def _predict_category_with_infer(ticket: Dict[str, Any]) -> Tuple[str, float]:
    """Use the same inference path as /classify (stable)."""
    model = load_deployed_model()
    out = model.predict(ticket)
    return out["category"], float(out["confidence"])

def _make_query_text(ticket: Dict[str, Any], predicted_category: str) -> str:
    title   = (ticket.get("subject") or "").strip()
    product = (ticket.get("product") or "").strip()
    tags    = " ".join(ticket.get("tags", []) or [])
    desc    = (ticket.get("description") or "").strip()
    category = predicted_category or (ticket.get("category") or "")
    return " ".join([title, product, category, tags, desc]).strip()

def _rank(
    X_all: sparse.csr_matrix,
    vec,
    meta: pd.DataFrame,
    query_text: str,
    predicted_category: str,
    k: int
) -> List[Dict[str, Any]]:
    """Vectorize query, category-filter candidates, score, and return top-k."""
    qvec = vec.transform([query_text])              # 1 x V, tiny; safe to toarray()
    # Category filter
    cand_idx = meta.index[meta["category"] == predicted_category].to_numpy()
    if cand_idx.size == 0:
        cand_idx = meta.index.to_numpy()
    X_cand = X_all[cand_idx]                        # Nc x V

    # Cosine == dot for l2-normalized tf-idf
    scores = (qvec @ X_cand.T).toarray().ravel()
    topk_local = np.argsort(-scores)[:k]
    top_rows = cand_idx[topk_local]

    results: List[Dict[str, Any]] = []
    for i, ridx in enumerate(top_rows):
        r = meta.iloc[ridx]
        results.append({
            "doc_id": r["doc_id"],
            "product": r.get("product"),
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "resolution_code": r.get("resolution_code"),
            "quality": float(r.get("quality", 0.0)),
            "score": float(scores[topk_local[i]]),
            "preview": "\n".join([
                f"[TITLE] {r.get('doc_id')}",
                f"[PRODUCT] {r.get('product')}",
                f"[CATEGORY] {r.get('category')}::{r.get('subcategory')}",
                f"[RES_CODE] {r.get('resolution_code')}",
            ])
        })
    return results

def retrieve(ticket_json_path: Path, k: int = 5) -> Dict[str, Any]:
    """Main retrieval routine used by CLI and FastAPI."""
    # Load artifacts
    X_all, tfidf_vec, meta = _load_retrieval_artifacts()

    # Load ticket
    ticket = json.loads(Path(ticket_json_path).read_text(encoding="utf-8"))

    # Predict category via stable inference stack
    pred_cat, conf = _predict_category_with_infer(ticket)

    # Build query and rank
    query_text = _make_query_text(ticket, pred_cat)
    results = _rank(X_all, tfidf_vec, meta, query_text, pred_cat, k)

    return {
        "predicted_category": pred_cat,
        "predicted_confidence": conf,
        "results": results,
    }

if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticket", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    print(_json.dumps(retrieve(Path(args.ticket), args.k), indent=2))
