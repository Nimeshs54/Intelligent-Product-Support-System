from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass
class SpikeConfig:
    window: int = 14
    z_threshold: float = 3


def _prep_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """
    Normalize to a daily, monotonic series:
      - handle tz-aware and tz-naive dates
      - fill missing days with 0
    """
    # Keep only the columns we need and drop empties
    d = df[[date_col, value_col]].dropna(subset=[date_col])

    # to_datetime handles both date and datetime strings
    idx = pd.to_datetime(d[date_col], errors="coerce")
    # If tz-aware, convert to naive; if tz-naive, leave as-is
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except (TypeError, AttributeError):
        # tz-naive path (nothing to convert)
        pass

    s = pd.Series(d[value_col].astype(float).values, index=idx)
    # Ensure sorted daily frequency and fill missing days with 0
    s = s.sort_index()
    # If the index is date-only (no time), it’s still ok for asfreq
    s = s.asfreq("D", fill_value=0.0)
    return s


def detect_spikes(series: pd.Series, cfg: SpikeConfig) -> pd.DataFrame:
    """
    Rolling mean/std; flag points with z-score > threshold as anomalies.
    Uses ddof=0 to avoid NaN std on constant windows.
    """
    s = series.copy()
    minp = max(3, cfg.window // 2)
    roll_mean = s.rolling(cfg.window, min_periods=minp).mean()
    roll_std = s.rolling(cfg.window, min_periods=minp).std(ddof=0).replace(0.0, np.nan)
    z = (s - roll_mean) / roll_std
    flags = (z > cfg.z_threshold).fillna(False)

    out = pd.DataFrame({
        "date": s.index,
        "value": s.values,
        "rolling_mean": roll_mean.values,
        "rolling_std": roll_std.values,
        "zscore": z.values,
        "is_anomaly": flags.values,
    })
    return out


def detect_volume_spikes(daily_total: pd.DataFrame, window=14, z_threshold=3) -> pd.DataFrame:
    s = _prep_series(daily_total, "created_date", "count")
    return detect_spikes(s, SpikeConfig(window=window, z_threshold=z_threshold))


def detect_category_spikes(daily_by_category: pd.DataFrame, window=14, z_threshold=3) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for cat, g in daily_by_category.groupby("category", dropna=False):
        s = _prep_series(g, "created_date", "count")
        out[str(cat)] = detect_spikes(s, SpikeConfig(window=window, z_threshold=z_threshold))
    return out


def detect_sentiment_shift(daily_sentiment: pd.DataFrame, window=14, z_threshold=3) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sent, g in daily_sentiment.groupby("sentiment_norm", dropna=False):
        s = _prep_series(g, "created_date", "count")
        out[str(sent)] = detect_spikes(s, SpikeConfig(window=window, z_threshold=z_threshold))
    return out


def retrieval_failure_proxy(timeseries_scores: pd.DataFrame, window=14, threshold=0.10) -> pd.DataFrame:
    """
    Given a daily time series of 'failure_rate' (0..1), flag days exceeding threshold
    and also days that spike relative to rolling mean.

    Expected columns:
      - created_date (date or datetime)
      - failure_rate (float 0..1)
    """
    if timeseries_scores.empty:
        return pd.DataFrame(columns=["date", "failure_rate", "rolling_mean", "zscore", "is_anomaly"])

    d = timeseries_scores.copy()
    d = d.dropna(subset=["created_date"])
    idx = pd.to_datetime(d["created_date"], errors="coerce")
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except (TypeError, AttributeError):
        pass

    s = pd.Series(d["failure_rate"].astype(float).values, index=idx).sort_index()
    s = s.asfreq("D", fill_value=0.0)

    minp = max(3, window // 2)
    roll = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std(ddof=0).replace(0.0, np.nan)
    z = (s - roll) / std
    hard_flag = (s >= threshold).fillna(False)
    soft_flag = (z > 3).fillna(False)
    flags = hard_flag | soft_flag

    out = pd.DataFrame({
        "date": s.index,
        "failure_rate": s.values,
        "rolling_mean": roll.values,
        "zscore": z.values,
        "is_anomaly": flags.values,
    })
    return out
