# models/infer.py
"""
Inference loader that uses the native XGBoost Booster JSON (no custom pickles).
Assumes feature artifacts were built with feature_store/build_features.py and
are loadable via joblib. We install shims so old pickles that referenced
__main__.TextConcatenator can be loaded without retraining.

Additionally, if the loaded ColumnTransformer has a text branch that
accidentally interprets multiple text columns as multiple samples (causing
row-dimension mismatch on hstack), we detect that error and fall back to a
manual, robust transform that guarantees 1-sample behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse as sp
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from models.pickle_shims import TextConcatenator as ShimTextConcatenator  # compat

FEAT_DIR = Path("data/artifacts/features")
MODEL_DIR = Path("data/artifacts/models")

TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]
CAT_COLS = [
    "product",
    "product_module",
    "language",
    "region",
    "priority",
    "severity",
    "customer_tier",
    "channel",
    "environment",
]
BOOL_COLS = [
    "contains_error_code",
    "contains_stack_trace",
    "weekend_ticket",
    "after_hours",
    "known_issue",
    "bug_report_filed",
    "auto_suggestion_accepted",
    "resolution_helpful",
    "escalated",
]
NUM_COLS = [
    "previous_tickets",
    "account_age_days",
    "account_monthly_value",
    "similar_issues_last_30_days",
    "product_version_age_days",
    "ticket_text_length",
    "response_count",
    "attachments_count",
    "affected_users",
    "resolution_time_hours",
    "resolution_attempts",
    "agent_experience_months",
    "transferred_count",
    "satisfaction_score",
]


def _install_unpickle_shims():
    """Inject a compatible TextConcatenator into __main__ for old pickles."""
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "TextConcatenator"):
        setattr(main_mod, "TextConcatenator", ShimTextConcatenator)


def _join_text(df: pd.DataFrame, cols: List[str]) -> str:
    vals = []
    for c in cols:
        v = df.iloc[0][c] if c in df.columns else ""
        v = "" if v is None else str(v)
        vals.append(v.strip())
    return " ".join([v for v in vals if v])


def _manual_transform_single_sample(featurizer, df_row: pd.DataFrame) -> sp.csr_matrix:
    """
    Manually apply each fitted sub-transformer for a single-sample dataframe.
    This avoids ColumnTransformer's hstack when a text branch misreads shape.
    """
    pieces: List[sp.csr_matrix] = []

    # featurizer is usually a ColumnTransformer
    # Each item: (name, transformer, columns)
    for name, trans, cols in featurizer.transformers_:
        if trans == "drop":
            continue

        # Collect the data slice for this transformer as we intend it.
        # Ensure df_slice is 1-sample DataFrame with all required cols present.
        if isinstance(cols, (list, tuple)):
            cols_list = list(cols)
        elif cols is None:
            cols_list = []
        else:
            cols_list = [cols]

        df_slice = pd.DataFrame({c: df_row[c] if c in df_row.columns else "" for c in cols_list})

        out = None

        # If it's a Pipeline, we need the final estimator and feed the right shape.
        if isinstance(trans, Pipeline):
            last_est = trans.steps[-1][1]
            # Handle typical text pipeline ending with TfidfVectorizer
            if isinstance(last_est, TfidfVectorizer):
                # Force single string input
                doc = _join_text(df_slice, cols_list if cols_list else df_row.columns.tolist())
                vec = last_est
                out = vec.transform([doc])  # shape (1, n_features)

            else:
                # Non-text pipeline: just call transform(df_slice)
                Xt = trans.transform(df_slice)
                # Convert to CSR
                out = Xt if sp.issparse(Xt) else sp.csr_matrix(np.asarray(Xt))
        else:
            # Passthrough or estimator directly
            if trans == "passthrough":
                Xt = df_slice.values
                out = sp.csr_matrix(np.asarray(Xt))
            else:
                Xt = trans.transform(df_slice)
                out = Xt if sp.issparse(Xt) else sp.csr_matrix(np.asarray(Xt))

        if out is None:
            out = sp.csr_matrix((1, 0))
        # Ensure row dimension is exactly 1
        if out.shape[0] != 1:
            # Collapse to one row if something still went off (take first row)
            out = out[:1]
        pieces.append(out)

    # Handle remainder if needed (when set to 'passthrough' and columns unspecified)
    if hasattr(featurizer, "remainder") and featurizer.remainder == "passthrough":
        used = set()
        for _, _, cols in featurizer.transformers_:
            if isinstance(cols, (list, tuple)):
                used.update(cols)
            elif cols is not None:
                used.add(cols)
        rem_cols = [c for c in df_row.columns if c not in used]
        if rem_cols:
            Xt = df_row[rem_cols].values
            pieces.append(sp.csr_matrix(np.asarray(Xt)))

    if not pieces:
        return sp.csr_matrix((1, 0))

    X = sp.hstack(pieces).tocsr()
    return X


class LoadedModel:
    def __init__(self, booster: xgb.Booster, featurizer, le_cat):
        self.booster = booster
        self.featurizer = featurizer
        self.le_cat = le_cat
        self.name = "xgb_category_booster"

    def _row_from_ticket(self, ticket) -> Dict[str, Any]:
        def get(attr, default=None):
            if isinstance(ticket, dict):
                return ticket.get(attr, default)
            return getattr(ticket, attr, default)
        row = {c: get(c) for c in (TEXT_COLS + CAT_COLS + NUM_COLS + BOOL_COLS)}
        # sanitize
        for c in TEXT_COLS:
            row[c] = row.get(c) or ""
        for c in BOOL_COLS:
            row[c] = float(bool(row.get(c, False)))
        return row

    def predict(self, ticket) -> Dict[str, Any]:
        row = self._row_from_ticket(ticket)
        df_row = pd.DataFrame([row])

        # First try the regular path
        try:
            X = self.featurizer.transform(df_row)
        except ValueError as e:
            msg = str(e)
            if "incompatible row dimensions" in msg or "blocks[0,:] has incompatible row dimensions" in msg:
                # Use robust manual transform for 1-sample
                X = _manual_transform_single_sample(self.featurizer, df_row)
            else:
                raise

        dmat = xgb.DMatrix(X)
        proba = self.booster.predict(dmat)[0]
        idx = int(np.argmax(proba))
        category = self.le_cat.inverse_transform([idx])[0]
        confidence = float(proba[idx])

        # simple heuristics for demo
        text = (row.get("subject", "") + " " + row.get("description", "")).lower()
        sentiment = "frustrated" if any(w in text for w in ["error", "fail", "issue", "crash", "broken"]) else "neutral"
        priority = "high" if row.get("contains_error_code", 0.0) > 0.5 else "medium"
        return {
            "category": category,
            "subcategory": None,
            "priority": priority,
            "sentiment": sentiment,
            "confidence": confidence,
            "model": self.name,
            "top_features": [],
        }


def load_deployed_model() -> LoadedModel:
    # Install shims so joblib can resolve old __main__.TextConcatenator references
    _install_unpickle_shims()

    # Load artifacts
    featurizer = joblib.load(FEAT_DIR / "featurizer.joblib")
    le_cat = joblib.load(FEAT_DIR / "label_encoder_category.joblib")
    booster = xgb.Booster()
    booster.load_model((MODEL_DIR / "xgb_category.booster.json").as_posix())
    return LoadedModel(booster, featurizer, le_cat)
