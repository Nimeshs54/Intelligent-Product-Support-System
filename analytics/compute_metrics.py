"""
Compute core analytics tables required by monitoring:
- daily_total.parquet
- daily_by_category.parquet
- daily_sentiment.parquet
- retrieval_daily.parquet

Reads the processed splits from data/processed/{train,val,test}.json,
concatenates them, and writes Parquet files to data/artifacts/analytics/.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import json
import pandas as pd

PROC_DIR = Path("data/processed")
OUT_DIR = Path("data/artifacts/analytics")


def _read_split(name: str) -> pd.DataFrame:
    p = PROC_DIR / f"{name}.json"
    records = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame.from_records(records)


def _load_all() -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        df = _read_split(split)
        df["split"] = split
        parts.append(df)
    all_df = pd.concat(parts, ignore_index=True)

    # normalize created_at to date
    all_df["created_at"] = pd.to_datetime(all_df["created_at"], errors="coerce")
    all_df["created_date"] = all_df["created_at"].dt.tz_localize(None).dt.date
    # Some fields may be missing in temp dataset; guard them:
    for col in ("category", "customer_sentiment"):
        if col not in all_df.columns:
            all_df[col] = None
    # helpfulness proxy fields
    for col in ("kb_articles_viewed", "kb_articles_helpful", "auto_suggestion_accepted"):
        if col not in all_df.columns:
            all_df[col] = None
    return all_df


def _safe_len_list(x):
    if isinstance(x, list):
        return len(x)
    return 0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_all()

    # 1) Daily total volume
    daily_total = (
        df.groupby("created_date", dropna=False)
          .size()
          .rename("count")
          .reset_index()
    )

    # 2) Daily by category
    daily_by_category = (
        df.groupby(["created_date", "category"], dropna=False)
          .size()
          .rename("count")
          .reset_index()
    )

    # 3) Daily sentiment (normalize to small set)
    def _norm_sent(s):
        if not isinstance(s, str):
            return "unknown"
        s = s.lower()
        if "frustrat" in s or "angry" in s or "upset" in s or "unhappy" in s:
            return "negative"
        if "happy" in s or "satisf" in s or "great" in s or "good" in s:
            return "positive"
        return "neutral"

    df["sentiment_norm"] = df["customer_sentiment"].apply(_norm_sent)
    daily_sentiment = (
        df.groupby(["created_date", "sentiment_norm"], dropna=False)
          .size()
          .rename("count")
          .reset_index()
    )

    # 4) Retrieval failure proxy per day (no live logs -> heuristic proxy)
    # Define "retrieval success" if any helpful KB article exists OR auto_suggestion_accepted is True
    viewed_len = df["kb_articles_viewed"].apply(_safe_len_list)
    helpful_len = df["kb_articles_helpful"].apply(_safe_len_list)
    auto_acc = df["auto_suggestion_accepted"].fillna(False).astype(bool)
    success = (helpful_len > 0) | (auto_acc == True)
    df["_retrieval_success"] = success.astype(int)

    retrieval_daily = (
        df.groupby("created_date", dropna=False)["_retrieval_success"]
          .agg(total="count", success="sum")
          .reset_index()
    )
    retrieval_daily["failure"] = retrieval_daily["total"] - retrieval_daily["success"]
    retrieval_daily["failure_rate"] = retrieval_daily["failure"] / retrieval_daily["total"]
    retrieval_daily = retrieval_daily[["created_date", "total", "success", "failure", "failure_rate"]]

    # Write the four required Parquet files
    (OUT_DIR / "daily_total.parquet").write_bytes(daily_total.to_parquet(index=False))
    (OUT_DIR / "daily_by_category.parquet").write_bytes(daily_by_category.to_parquet(index=False))
    (OUT_DIR / "daily_sentiment.parquet").write_bytes(daily_sentiment.to_parquet(index=False))
    (OUT_DIR / "retrieval_daily.parquet").write_bytes(retrieval_daily.to_parquet(index=False))

    print(f"[OK] Analytics saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
