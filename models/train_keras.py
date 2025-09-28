"""
Train a simple Keras MLP on TF-IDF features with TruncatedSVD to control RAM.
This module exposes TextConcatenator in globals so joblib can unpickle older
featurizers safely.

Artifacts -> data/artifacts/models/keras/
  - model.keras
  - svd.joblib
  - scaler.joblib
  - label_encoder_category.joblib
  - train_history.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import load, dump

try:
    from models.pickle_shims import TextConcatenator  # noqa: F401
except Exception:
    class TextConcatenator:
        def __init__(self, text_cols=None, out_col="__concat__", sep=" "):
            self.text_cols = text_cols or ["subject", "description", "error_logs", "stack_trace"]
            self.out_col = out_col
            self.sep = sep
        def fit(self, X, y=None): return self
        def transform(self, X):
            import pandas as pd
            X = X.copy()
            for c in self.text_cols:
                if c not in X.columns:
                    X[c] = ""
            X[self.out_col] = (
                X[self.text_cols]
                .astype(str)
                .applymap(lambda s: s.strip())
                .agg(self.sep.join, axis=1)
            )
            return X

# ---- ML imports ----
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ---- Paths ----
ROOT = Path(".")
FEAT_DIR = ROOT / "data" / "artifacts" / "features"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "data" / "artifacts" / "models" / "keras"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Column groups (for normalization only) ----
TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]
CAT_COLS  = ["product", "product_module", "language", "region", "priority", "severity",
             "customer_tier", "channel", "environment"]
BOOL_COLS = ["contains_error_code", "contains_stack_trace", "weekend_ticket", "after_hours",
             "known_issue", "bug_report_filed", "auto_suggestion_accepted",
             "resolution_helpful", "escalated"]
NUM_COLS  = [
    "previous_tickets", "account_age_days", "account_monthly_value",
    "similar_issues_last_30_days", "product_version_age_days",
    "ticket_text_length", "response_count", "attachments_count", "affected_users",
    "resolution_time_hours", "resolution_attempts", "agent_experience_months",
    "transferred_count", "satisfaction_score"
]
ALL_FEAT_COLS = TEXT_COLS + CAT_COLS + NUM_COLS + BOOL_COLS  # for normalization only


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json_array(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    return pd.read_json(text)


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("[2/7] Loading splits…")
    train_p = PROC_DIR / "train.json"
    val_p   = PROC_DIR / "val.json"

    if not train_p.exists() or not val_p.exists():
        raise FileNotFoundError(f"Missing processed splits in {PROC_DIR}. Run: python -m pipelines.bootstrap")

    try:
        train_df = pd.read_json(train_p, lines=False)
        log(f"[READ] train.json: parsed as JSON array with {len(train_df)} rows")
    except ValueError:
        train_df = read_json_array(train_p)
        log(f"[READ] train.json: parsed via raw text, rows={len(train_df)}")

    try:
        val_df = pd.read_json(val_p, lines=False)
        log(f"[READ] val.json: parsed as JSON array with {len(val_df)} rows")
    except ValueError:
        val_df = read_json_array(val_p)
        log(f"[READ] val.json: parsed via raw text, rows={len(val_df)}")

    log(f"[INFO] Loaded train={len(train_df)}, val={len(val_df)}")
    return train_df, val_df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("[3/7] Normalizing/creating missing columns…")
    df = df.copy()

    # Ensure all columns exist (so featurizer sees a consistent schema)
    for c in ALL_FEAT_COLS + ["category"]:
        if c not in df.columns:
            if c in TEXT_COLS: df[c] = ""
            elif c in CAT_COLS: df[c] = "unknown"
            elif c in BOOL_COLS: df[c] = False
            elif c in NUM_COLS: df[c] = 0.0
            elif c == "category": df[c] = "Unknown"

    # Light cleanup for text/bool/num
    df[TEXT_COLS] = df[TEXT_COLS].astype(str).applymap(lambda s: s.strip())
    for c in BOOL_COLS:
        df[c] = df[c].astype(float)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def to_xy(featurizer, df: pd.DataFrame, le_cat) -> Tuple:
    """
    IMPORTANT CHANGE:
      Pass the FULL DataFrame to featurizer.transform(df)
      (do NOT slice to ALL_FEAT_COLS), because the saved featurizer
      selects its own columns internally. Slicing caused row-mismatch.
    """
    X = featurizer.transform(df)        # <-- FIXED: was df[ALL_FEAT_COLS]
    y = le_cat.transform(df["category"].astype(str))
    return X, y


def build_model(input_dim: int, hidden: int, dropout: float, lr: float = 1e-3) -> keras.Model:
    log("[5/7] Building model…")
    inp = layers.Input(shape=(input_dim,), name="features")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden // 2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(5, activation="softmax")(x)  # 5 classes

    model = keras.Model(inp, out)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--svd_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    log(f"[ENV] TensorFlow: {tf.__version__}, NumPy: {np.__version__}")

    # 1) Load artifacts
    log("[1/7] Loading artifacts…")
    featurizer = load(FEAT_DIR / "featurizer.joblib")
    le_cat     = load(FEAT_DIR / "label_encoder_category.joblib")
    classes    = list(le_cat.classes_)
    log("[OK] Featurizer + label encoder loaded")
    log(f"[INFO] Classes: {len(classes)} — {classes}")

    # 2) Load splits
    train_df, val_df = load_splits()

    if args.limit_train:
        train_df = train_df.iloc[: args.limit_train].reset_index(drop=True)
        log(f"[INFO] Limiting training rows to {len(train_df)}")

    # 3) Normalize columns (keep full frame; featurizer will pick what it needs)
    train_df = normalize_columns(train_df)
    val_df   = normalize_columns(val_df)

    # 4) Transform features (sparse), then SVD -> dense
    log("[4/7] Transforming features… (this can take a moment)")
    X_train_sp, y_train = to_xy(featurizer, train_df, le_cat)
    X_val_sp,   y_val   = to_xy(featurizer, val_df,   le_cat)
    log(f"[INFO] Shapes — X_train: {X_train_sp.shape}, X_val: {X_val_sp.shape}")

    svd_path = OUT_DIR / "svd.joblib"
    if svd_path.exists():
        svd = load(svd_path)
        log(f"[INFO] Loaded existing SVD: {svd_path}")
    else:
        log(f"[INFO] Fitting TruncatedSVD(n_components={args.svd_dim}) on train features…")
        svd = TruncatedSVD(n_components=args.svd_dim, n_iter=7, random_state=42)
        svd.fit(X_train_sp)
        dump(svd, svd_path, compress=3)
        log(f"[OK] Saved SVD to {svd_path}")

    X_train = svd.transform(X_train_sp).astype("float32")
    X_val   = svd.transform(X_val_sp).astype("float32")
    log(f"[INFO] After SVD — X_train: {X_train.shape}, X_val: {X_val.shape}")

    # Standardize
    scaler_path = OUT_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = load(scaler_path)
        log(f"[INFO] Loaded existing scaler: {scaler_path}")
    else:
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X_train)
        dump(scaler, scaler_path, compress=3)
        log(f"[OK] Saved scaler to {scaler_path}")

    X_train = scaler.transform(X_train).astype("float32")
    X_val   = scaler.transform(X_val).astype("float32")

    # 5) Build model
    model = build_model(input_dim=X_train.shape[1],
                        hidden=args.hidden,
                        dropout=args.dropout,
                        lr=args.lr)

    # 6) Train
    log("[6/7] Training…")
    ckpt_path = OUT_DIR / "model.keras"
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path.as_posix(), monitor="val_accuracy", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=cbs
    )

    # 7) Save artifacts
    log("[7/7] Saving artifacts…")
    dump(le_cat, OUT_DIR / "label_encoder_category.joblib", compress=3)
    (OUT_DIR / "train_history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2),
        encoding="utf-8"
    )
    log(f"[OK] Saved model to {ckpt_path}")
    log(f"[OK] Saved SVD + scaler + label encoder to {OUT_DIR}")
    log("[DONE] Keras training finished.")


if __name__ == "__main__":
    main()
