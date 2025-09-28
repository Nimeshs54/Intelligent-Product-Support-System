from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

REQ_COLS = [
    "ticket_id","created_at","subject","description","category"
]

def df_from_json(path: Path) -> pd.DataFrame:
    txt = path.read_text(encoding="utf-8").strip()
    if txt.startswith("["):
        return pd.read_json(path, orient="records", lines=False)
    return pd.read_json(path, orient="records", lines=True)

def check_split(path: Path) -> Dict[str, Any]:
    df = df_from_json(path)
    missing_cols = [c for c in REQ_COLS if c not in df.columns]
    issues = {}

    if missing_cols:
        issues["missing_columns"] = missing_cols

    # nulls
    nulls = {c:int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    if nulls:
        issues["null_counts"] = nulls

    # basic stats
    issues["n_rows"] = len(df)
    issues["n_cols"] = len(df.columns)
    issues["category_counts"] = df["category"].value_counts().to_dict() if "category" in df else {}

    return issues

def run_all(in_dir: Path, out_path: Path):
    out = {}
    for split in ["train","val","test"]:
        p = in_dir / f"{split}.json"
        out[split] = check_split(p)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[DATACHECK] report -> {out_path}")
