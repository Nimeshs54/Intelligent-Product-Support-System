# models/pickle_shims.py
"""
Shims for unpickling old feature pipelines that referenced local classes
(e.g., __main__.TextConcatenator). We provide a compatible class that the
pipeline can import at unpickle time without breaking.

This avoids retraining artifacts just because the original class lived in a
script's __main__ at training time.
"""
from __future__ import annotations

from typing import Iterable, Optional, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextConcatenator(BaseEstimator, TransformerMixin):
    """
    Compatible shim for a text field concatenator used during training.

    Notes about unpickling:
    - scikit-learn may bypass __init__ during unpickle, so attributes
      might not exist. We therefore use getattr(...) with defaults and
      set them if missing.
    """

    def __init__(
        self,
        text_cols: Optional[Iterable[str]] = None,
        out_col: str = "__text__",
        sep: str = " ",
        strip: bool = True,
        **kwargs,  # absorb any unknown params saved in the pickle
    ):
        self.text_cols = list(text_cols) if text_cols is not None else None
        self.out_col = out_col
        self.sep = sep
        self.strip = strip

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Handle unpickle cases where attributes might be missing.
        text_cols = getattr(self, "text_cols", None)
        out_col = getattr(self, "out_col", "__text__")
        sep = getattr(self, "sep", " ")
        strip = getattr(self, "strip", True)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Default columns if not provided (covers typical ticket fields)
        if text_cols is None:
            candidates: List[str] = [
                "subject",
                "description",
                "error_logs",
                "stack_trace",
                "feedback_text",
            ]
            text_cols = [c for c in candidates if c in X.columns]
            try:
                self.text_cols = list(text_cols)
            except Exception:
                pass

        # Ensure each column exists
        for c in text_cols:
            if c not in X.columns:
                X[c] = ""

        parts = X[text_cols].fillna("").astype(str)
        if strip:
            # Column-wise strip (no applymap deprecation)
            parts = parts.apply(lambda col: col.str.strip())

        X = X.copy()
        X[out_col] = parts.apply(lambda r: sep.join([v for v in r if v]), axis=1)

        # Persist other attrs if they were missing during unpickle
        for name, value in [("out_col", out_col), ("sep", sep), ("strip", strip)]:
            if not hasattr(self, name):
                try:
                    setattr(self, name, value)
                except Exception:
                    pass

        return X
