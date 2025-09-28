"""
Run anomaly scan over precomputed analytics and save a JSON report.

Improvements:
- If analytics Parquet files are missing, compute them first by importing
  analytics.compute_metrics and running its main().
- Robust JSON serialization (Timestamps, numpy scalars, datetimes, etc.)
  — uses .dt.strftime('%Y-%m-%d') for date-like columns (no .isoformat on .dt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from monitoring.anomaly_rules import (
    detect_volume_spikes,
    detect_category_spikes,
    detect_sentiment_shift,
    retrieval_failure_proxy,
)

ANALYTICS_DIR = Path("data/artifacts/analytics")
DEFAULT_OUT = Path("data/artifacts/anomalies/report.json")

REQUIRED_FILES = [
    "daily_total.parquet",
    "daily_by_category.parquet",
    "daily_sentiment.parquet",
    "retrieval_daily.parquet",
]


def _ensure_analytics_exist():
    missing = [f for f in REQUIRED_FILES if not (ANALYTICS_DIR / f).exists()]
    if not missing:
        return
    print(f"[INFO] Missing analytics files: {missing} — computing now…")
    from analytics.compute_metrics import main as compute_main  # local import to avoid cycle
    compute_main()
    still_missing = [f for f in REQUIRED_FILES if not (ANALYTICS_DIR / f).exists()]
    if still_missing:
        raise FileNotFoundError(f"Analytics build did not produce: {still_missing}")


def _to_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types."""
    # pandas timestamp / numpy datetime
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        # Return date string; if it has time, keep date part
        ts = pd.Timestamp(obj)
        if ts.tz is not None:
            ts = ts.tz_convert(None)
        return ts.strftime("%Y-%m-%d")

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return [_to_serializable(x) for x in obj.tolist()]

    # pandas Series/DataFrame
    if isinstance(obj, pd.Series):
        return [_to_serializable(x) for x in obj.to_list()]
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        for col in out.columns:
            # Try to coerce to datetime; if many NaT, leave as-is
            if pd.api.types.is_datetime64_any_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
                coerced = pd.to_datetime(out[col], errors="coerce", utc=False)
                # If we got at least one valid datetime after coercion, convert the whole col
                if coerced.notna().any():
                    try:
                        # if tz-aware, make tz-naive
                        if getattr(coerced.dt, "tz", None) is not None:
                            coerced = coerced.dt.tz_convert(None)
                    except Exception:
                        # tz-naive path
                        pass
                    # Write as 'YYYY-MM-DD'
                    out[col] = coerced.dt.strftime("%Y-%m-%d")
        # Convert to records, then recurse on items (to handle any remaining numpy types)
        return [_to_serializable(rec) for rec in out.to_dict(orient="records")]

    # dict / list
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]

    # plain python types
    return obj


def load_analytics() -> Dict[str, pd.DataFrame]:
    _ensure_analytics_exist()
    daily_total = pd.read_parquet(ANALYTICS_DIR / "daily_total.parquet")
    daily_by_category = pd.read_parquet(ANALYTICS_DIR / "daily_by_category.parquet")
    daily_sentiment = pd.read_parquet(ANALYTICS_DIR / "daily_sentiment.parquet")
    retrieval_metrics = pd.read_parquet(ANALYTICS_DIR / "retrieval_daily.parquet")
    return {
        "daily_total": daily_total,
        "daily_by_category": daily_by_category,
        "daily_sentiment": daily_sentiment,
        "retrieval_metrics": retrieval_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--ret_fail_threshold", type=float, default=0.10)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    print("[INFO] Loading analytics tables…")
    tables = load_analytics()

    print("[INFO] Detecting volume spikes…")
    vol = detect_volume_spikes(tables["daily_total"], window=args.window, z_threshold=args.z)

    print("[INFO] Detecting category spikes…")
    cat = detect_category_spikes(tables["daily_by_category"], window=args.window, z_threshold=args.z)

    print("[INFO] Detecting sentiment shifts…")
    sent = detect_sentiment_shift(tables["daily_sentiment"], window=args.window, z_threshold=args.z)

    print("[INFO] Checking retrieval failure proxy…")
    ret_fail = retrieval_failure_proxy(
        tables["retrieval_metrics"], window=args.window, threshold=args.ret_fail_threshold
    )

    report = {
        "params": {
            "window": args.window,
            "z_threshold": args.z,
            "retrieval_failure_threshold": args.ret_fail_threshold,
        },
        "volume_spikes": vol,
        "category_spikes": cat,
        "sentiment_shift": sent,
        "retrieval_failure": ret_fail,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing JSON report to {out_path} …")
    out_path.write_text(json.dumps(_to_serializable(report), indent=2), encoding="utf-8")
    print("[OK] Anomaly report saved.")


if __name__ == "__main__":
    main()
