# models/train_keras_textonly.py
"""
Keras text-only classifier to compare with XGBoost without relying on the old
sklearn ColumnTransformer. Keeps things simple and RAM-safe.

Artifacts -> data/artifacts/models/keras_text/
  - model.keras
  - tfidf.joblib
  - svd.joblib
  - scaler.joblib
  - label_encoder_category.joblib (copied/loaded from features dir)
  - train_history.json
  - val_report.json  (accuracy + weighted F1)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


ROOT = Path(".")
PROC_DIR = ROOT / "data" / "processed"
FEAT_DIR = ROOT / "data" / "artifacts" / "features"   # for existing label encoder
OUT_DIR  = ROOT / "data" / "artifacts" / "models" / "keras_text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]


def log(msg: str) -> None:
    print(msg, flush=True)


def _read_json_array(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    return pd.read_json(text)


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("[2/7] Loading splits…")
    train_p = PROC_DIR / "train.json"
    val_p   = PROC_DIR / "val.json"
    if not train_p.exists() or not val_p.exists():
        raise FileNotFoundError(f"Missing processed splits in {PROC_DIR}. Run: python -m pipelines.bootstrap")

    # Try array-of-objects first; fallback to raw read if needed
    try:
        train_df = pd.read_json(train_p, lines=False)
        log(f"[READ] train.json as array: {len(train_df)} rows")
    except ValueError:
        train_df = _read_json_array(train_p)
        log(f"[READ] train.json via raw text: {len(train_df)} rows")

    try:
        val_df = pd.read_json(val_p, lines=False)
        log(f"[READ] val.json as array: {len(val_df)} rows")
    except ValueError:
        val_df = _read_json_array(val_p)
        log(f"[READ] val.json via raw text: {len(val_df)} rows")

    log(f"[INFO] Loaded train={len(train_df)}, val={len(val_df)}")
    return train_df, val_df


def prepare_text(df: pd.DataFrame) -> pd.Series:
    # Ensure text cols exist
    for c in TEXT_COLS:
        if c not in df.columns:
            df[c] = ""
    # Clean minimal
    for c in TEXT_COLS:
        df[c] = df[c].astype(str).str.strip()
    # Concatenate
    text = (
        df["subject"] + " \n " +
        df["description"] + " \n " +
        df["error_logs"] + " \n " +
        df["stack_trace"]
    )
    return text.fillna("")


def build_model(input_dim: int, hidden: int, dropout: float, lr: float = 1e-3, n_classes: int = 5) -> keras.Model:
    log("[5/7] Building model…")
    inp = layers.Input(shape=(input_dim,), name="features")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(max(64, hidden // 2), activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--svd_dim", type=int, default=256)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    log(f"[ENV] TensorFlow: {tf.__version__}, NumPy: {np.__version__}")

    # 1) Label encoder (to keep class order consistent with XGB)
    log("[1/7] Loading label encoder…")
    le_path = FEAT_DIR / "label_encoder_category.joblib"
    if not le_path.exists():
        raise FileNotFoundError(f"Missing {le_path}. Train the classic model first or run: python -m pipelines.bootstrap")
    le_cat = load(le_path)
    classes = list(le_cat.classes_)
    n_classes = len(classes)
    log(f"[OK] Label encoder loaded; classes={n_classes} — {classes}")

    # 2) Load splits
    train_df, val_df = load_splits()
    if args.limit_train:
        train_df = train_df.iloc[: args.limit_train].reset_index(drop=True)
        log(f"[INFO] Limiting training rows to {len(train_df)}")

    if len(train_df) < max(500, args.batch_size * 4) or len(val_df) < 500:
        log(f"[WARN] Very small splits (train={len(train_df)}, val={len(val_df)}). "
            f"This is okay for a demo but won’t be representative.")

    # 3) Prepare text + labels
    log("[3/7] Preparing text + labels…")
    tr_text = prepare_text(train_df)
    va_text = prepare_text(val_df)
    y_train = le_cat.transform(train_df["category"].astype(str))
    y_val   = le_cat.transform(val_df["category"].astype(str))

    # 4) TF-IDF -> SVD -> Standardize
    log("[4/7] Vectorizing text (TF-IDF)…")
    tfidf_path = OUT_DIR / "tfidf.joblib"
    if tfidf_path.exists():
        tfidf = load(tfidf_path)
        log(f"[INFO] Loaded existing TF-IDF: {tfidf_path}")
    else:
        tfidf = TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(1, 2),
            min_df=args.min_df,
            dtype=np.float32
        )
        tfidf.fit(tr_text)
        dump(tfidf, tfidf_path, compress=3)
        log(f"[OK] Saved TF-IDF to {tfidf_path}")

    X_train_sp = tfidf.transform(tr_text)
    X_val_sp   = tfidf.transform(va_text)
    log(f"[INFO] TF-IDF shapes — train: {X_train_sp.shape}, val: {X_val_sp.shape}")

    svd_path = OUT_DIR / "svd.joblib"
    if svd_path.exists():
        svd = load(svd_path)
        log(f"[INFO] Loaded existing SVD: {svd_path}")
    else:
        log(f"[INFO] Fitting TruncatedSVD(n_components={args.svd_dim})…")
        svd = TruncatedSVD(n_components=args.svd_dim, n_iter=7, random_state=42)
        svd.fit(X_train_sp)
        dump(svd, svd_path, compress=3)
        log(f"[OK] Saved SVD to {svd_path}")

    X_train = svd.transform(X_train_sp).astype("float32")
    X_val   = svd.transform(X_val_sp).astype("float32")
    log(f"[INFO] After SVD — train: {X_train.shape}, val: {X_val.shape}")

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
    model = build_model(
        input_dim=X_train.shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        n_classes=n_classes
    )

    # 6) Train
    log("[6/7] Training…")
    ckpt_path = OUT_DIR / "model.keras"
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path.as_posix(), monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=cbs
    )

    # 7) Save + quick eval
    log("[7/7] Saving artifacts + quick eval…")
    (OUT_DIR / "train_history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2),
        encoding="utf-8"
    )
    # Keep a copy of label encoder alongside (helpful for deployment)
    dump(le_cat, OUT_DIR / "label_encoder_category.joblib", compress=3)

    # Eval
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    acc = float(accuracy_score(y_val, y_val_pred))
    f1w = float(f1_score(y_val, y_val_pred, average="weighted"))
    (OUT_DIR / "val_report.json").write_text(json.dumps(
        {"val_accuracy": acc, "val_f1_weighted": f1w}, indent=2), encoding="utf-8"
    )
    log(f"[OK] Saved model to {ckpt_path}")
    log(f"[OK] Validation — accuracy={acc:.4f}, f1_weighted={f1w:.4f}")
    log(f"[DONE] Artifacts in: {OUT_DIR}")


if __name__ == "__main__":
    # Quiet TF logs a bit
    tf.get_logger().setLevel("ERROR")
    main()
