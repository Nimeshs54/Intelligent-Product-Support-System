"""
Lightweight inference loader for the TEXT-ONLY Keras model.
Uses artifacts produced by: models.train_keras_textonly
 - data/artifacts/models/keras_text/model.keras
 - data/artifacts/models/keras_text/tfidf.joblib
 - data/artifacts/models/keras_text/svd.joblib
 - data/artifacts/models/keras_text/scaler.joblib
And the label encoder from the feature store:
 - data/artifacts/features/label_encoder_category.joblib
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import re
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

# ------------ Paths ------------
MODEL_ROOT = Path("data/artifacts/models/keras_text")
FEAT_DIR   = Path("data/artifacts/features")

TFIDF_PATH  = MODEL_ROOT / "tfidf.joblib"
SVD_PATH    = MODEL_ROOT / "svd.joblib"
SCALER_PATH = MODEL_ROOT / "scaler.joblib"
MODEL_PATH  = MODEL_ROOT / "model.keras"
LE_PATH     = FEAT_DIR   / "label_encoder_category.joblib"

TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]

# ------------ Utilities ------------

def _normalize_text(s: str | None) -> str:
    if not s:
        return ""
    s = str(s)
    s = s.replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _concat_text(ticket: Dict[str, Any]) -> str:
    """Concatenate the 4 text fields in a stable order, used at training time."""
    parts = []
    for col in TEXT_COLS:
        parts.append(_normalize_text(ticket.get(col, "")))
    return " ".join([p for p in parts if p])

# ------------ Loader ------------

class LoadedDLModel:
    def __init__(self, tfidf, svd, scaler, keras_model, label_encoder):
        self.tfidf = tfidf
        self.svd = svd
        self.scaler = scaler
        self.model = keras_model
        self.le = label_encoder
        self.name = "keras_textonly"

    def predict(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        # build text
        text = _concat_text(ticket)
        if not text:
            text = _normalize_text(ticket.get("subject", ""))  # still allow empty

        # vectorize -> svd -> scale
        X_tfidf = self.tfidf.transform([text])
        X_svd   = self.svd.transform(X_tfidf)
        X       = self.scaler.transform(X_svd)

        # predict
        proba = self.model.predict(X, verbose=0)[0]
        idx   = int(np.argmax(proba))
        category   = self.le.inverse_transform([idx])[0]
        confidence = float(proba[idx])

        # cheap heuristics to match classic output fields
        txt = (ticket.get("subject", "") or "") + " " + (ticket.get("description", "") or "")
        low = txt.lower()
        sentiment = "frustrated" if any(w in low for w in ["error", "fail", "failing", "issue", "bug"]) else "neutral"
        priority  = "high" if any(w in low for w in ["p0", "critical", "urgent", "severe"]) else "medium"

        return {
            "category": category,
            "subcategory": None,
            "priority": priority,
            "sentiment": sentiment,
            "confidence": confidence,
            "model": self.name,
            "top_features": [],  # not computed for the DL model in this minimal setup
        }

def load_dl_model() -> LoadedDLModel:
    # Load artifacts
    tfidf  = joblib.load(TFIDF_PATH)
    svd    = joblib.load(SVD_PATH)
    scaler = joblib.load(SCALER_PATH)
    keras_model = keras_load_model(MODEL_PATH)
    label_encoder = joblib.load(LE_PATH)

    return LoadedDLModel(tfidf, svd, scaler, keras_model, label_encoder)
