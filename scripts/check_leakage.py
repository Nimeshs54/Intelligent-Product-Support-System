"""
Check for data leakage between train/val/test splits.
Logs overlap counts.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, logging
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)

PROC = Path("data/processed")
FIELDS = ["subject","description","error_logs","stack_trace"]

def hrow(r: pd.Series) -> str:
    s = "||".join(str(r.get(c,"") or "") for c in FIELDS)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def load(name: str) -> pd.DataFrame:
    logging.info(f"Loading split: {name}")
    df = pd.read_json(PROC / f"{name}.json", orient="records")
    for c in FIELDS:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna("")
    df["text_hash"] = df.apply(hrow, axis=1)
    logging.info(f"{name} size: {len(df)} rows")
    return df[["text_hash","ticket_id","category"]]

def main():
    logging.info("=== Checking for leakage between splits ===")
    tr = load("train"); va = load("val"); te = load("test")
    tr_h, va_h, te_h = set(tr.text_hash), set(va.text_hash), set(te.text_hash)

    overlap_tv = len(tr_h & va_h)
    overlap_tt = len(tr_h & te_h)
    overlap_vt = len(va_h & te_h)

    report = {
        "train": len(tr_h),
        "val": len(va_h),
        "test": len(te_h),
        "overlap": {
            "train∩val": overlap_tv,
            "train∩test": overlap_tt,
            "val∩test": overlap_vt
        }
    }
    logging.info("=== Overlap Report ===")
    logging.info(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
