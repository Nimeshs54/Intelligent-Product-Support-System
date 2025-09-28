import argparse, json, math
from pathlib import Path
from collections import defaultdict
import pandas as pd

from feature_store.schema import validate_records_json, dq_validate_dataframe


def read_json_records(path: Path):
    """
    Read a JSON array file robustly across Windows encodings.
    Tries UTF-8 first, then UTF-8 with BOM, then cp1252/latin-1 as last resort.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            return json.loads(text)
        except UnicodeDecodeError as e:
            last_err = e
        except json.JSONDecodeError:
            # Decoding succeeded but JSON failed → raise immediately
            raise
    raise last_err or RuntimeError("Unable to read file with common encodings")


def _to_df(records):
    # Normalize lists for DataFrame, parse datetimes
    df = pd.json_normalize(records)
    for col in ["created_at", "updated_at", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def stratified_temporal_split(df: pd.DataFrame, key_cols=("category", "subcategory"), ratios=(0.7, 0.15, 0.15)):
    """Stratify by (category, subcategory) then sort by created_at within each stratum and split by time."""
    assert abs(sum(ratios) - 1.0) < 1e-9
    strata = defaultdict(list)
    for i, row in df.iterrows():
        k = tuple((row.get(c) if c in df.columns else None) for c in key_cols)
        strata[k].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in strata.items():
        # sort by created_at; missing created_at go first (oldest)
        idxs_sorted = sorted(
            idxs,
            key=lambda i: (pd.Timestamp.min if pd.isna(df.loc[i, "created_at"]) else df.loc[i, "created_at"])
        )
        n = len(idxs_sorted)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_idx += idxs_sorted[:n_train]
        val_idx   += idxs_sorted[n_train:n_train + n_val]
        test_idx  += idxs_sorted[n_train + n_val:]

    return df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/support_tickets.json")
    ap.add_argument("--outdir", dest="out_dir", default="data/processed")
    ap.add_argument("--fail_on_validation", action="store_true", help="Exit non-zero if DQ checks fail")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load JSON with robust reader
    records = read_json_records(in_path)

    # 2) Pydantic validation (hard fail on schema errors)
    _ = validate_records_json(records)

    # 3) DataFrame + Data-Quality checks
    df = _to_df(records)
    dq_report = dq_validate_dataframe(df)

    if args.fail_on_validation and not dq_report.get("success", True):
        (out_dir / "validation_report.json").write_text(json.dumps(dq_report, indent=2), encoding="utf-8")
        raise SystemExit("DQ validation failed. See validation_report.json")

    # 4) Split (stratified + temporal)
    train_df, val_df, test_df = stratified_temporal_split(df)

    # 5) Save (UTF-8, no ASCII escaping)
    train_df.to_json(out_dir / "train.json", orient="records", force_ascii=False)
    val_df.to_json(out_dir / "val.json", orient="records", force_ascii=False)
    test_df.to_json(out_dir / "test.json", orient="records", force_ascii=False)
    (out_dir / "validation_report.json").write_text(json.dumps(dq_report, indent=2), encoding="utf-8")

    print(f"✓ Wrote splits to {out_dir}/ (train/val/test.json)")
    print(f"✓ Validation report: {out_dir}/validation_report.json")


if __name__ == "__main__":
    main()
