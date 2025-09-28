"""
Remake train/val/test splits with group deduplication on text hash.
Ensures duplicates don't cross splits.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, logging
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)

RAW = Path("data/support_tickets.json")
OUT = Path("data/processed")
FIELDS = ["subject","description","error_logs","stack_trace"]

def hrow(r) -> str:
    s = "||".join(str(r.get(c,"") or "") for c in FIELDS)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading dataset from {args.raw}")
    records = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    df = pd.DataFrame(records)

    for c in FIELDS:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna("")
    df["text_hash"] = df.apply(hrow, axis=1)

    logging.info(f"Total tickets: {len(df)}")
    logging.info("Performing group split...")

    # First split: train vs (val+test)
    gss = GroupShuffleSplit(n_splits=1, train_size=args.train, random_state=args.seed)
    train_idx, vt_idx = next(gss.split(df, groups=df["text_hash"]))
    df_train, df_vt = df.iloc[train_idx], df.iloc[vt_idx]

    # Second split: val vs test
    vt_size = args.val + args.test
    val_ratio = args.val / vt_size if vt_size > 0 else 0.5
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=args.seed+1)
    val_idx, test_idx = next(gss2.split(df_vt, groups=df_vt["text_hash"]))
    df_val, df_test = df_vt.iloc[val_idx], df_vt.iloc[test_idx]

    # Save
    (out/"train.json").write_text(df_train.to_json(orient="records", force_ascii=False), encoding="utf-8")
    (out/"val.json").write_text(df_val.to_json(orient="records", force_ascii=False), encoding="utf-8")
    (out/"test.json").write_text(df_test.to_json(orient="records", force_ascii=False), encoding="utf-8")

    logging.info("=== Split Sizes ===")
    logging.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    th_t, th_v, th_s = set(df_train.text_hash), set(df_val.text_hash), set(df_test.text_hash)
    logging.info("=== Overlap Check ===")
    logging.info(f"train∩val={len(th_t & th_v)}, train∩test={len(th_t & th_s)}, val∩test={len(th_v & th_s)}")

if __name__ == "__main__":
    main()
