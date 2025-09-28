# models/train_xgb.py
"""
Train an XGBoost multiclass classifier for `category` using the saved featurizer.
Uses native xgboost.train(...) with early stopping (robust across versions).
Saves booster JSON and training/eval metrics.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

# Ensure the featurizer's custom transformer is importable when loading
from feature_store.build_features import TextConcatenator  # noqa: F401

FEAT_DIR = Path("data/artifacts/features")
PROC_DIR = Path("data/processed")
MODEL_DIR = Path("data/artifacts/models")

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
TARGET_CATEGORY = "category"

def ts(msg: str):
    sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n"); sys.stdout.flush()

def load_split(name: str) -> pd.DataFrame:
    path = PROC_DIR / f"{name}.json"
    df = pd.read_json(path, orient="records")
    for c in TEXT_COLS + CAT_COLS + BOOL_COLS + NUM_COLS + [TARGET_CATEGORY]:
        if c not in df.columns:
            df[c] = np.nan if c not in BOOL_COLS else False
    for c in TEXT_COLS:
        df[c] = df[c].fillna("")
    for c in BOOL_COLS:
        df[c] = df[c].astype("boolean").astype(float)
    return df

def compute_class_weights(y_int: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y_int, return_counts=True)
    inv = 1.0 / np.maximum(counts, 1)
    weights = inv / inv.mean()
    class_to_w = {c: w for c, w in zip(classes, weights)}
    return np.array([class_to_w[i] for i in y_int], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MODEL_DIR))
    ap.add_argument("--limit_train", type=int, default=None, help="Cap training rows for faster runs")
    ap.add_argument("--n_estimators", type=int, default=1200)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=0.08)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--early_stopping_rounds", type=int, default=80)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--eval_verbosity", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ts("[INFO] Loading artifacts...")
    featurizer = joblib.load(FEAT_DIR / "featurizer.joblib")
    le_cat     = joblib.load(FEAT_DIR / "label_encoder_category.joblib")

    ts("[INFO] Loading splits...")
    train_df = load_split("train")
    val_df   = load_split("val")

    if args.limit_train is not None and args.limit_train < len(train_df):
        train_df = train_df.sample(n=args.limit_train, random_state=args.random_state).reset_index(drop=True)
        ts(f"[INFO] Limiting training rows to {len(train_df)}")

    ts("[INFO] Preparing targets...")
    y_train = le_cat.transform(train_df[TARGET_CATEGORY].fillna("unknown").astype(str).values).astype(np.int32)
    y_val   = le_cat.transform(val_df[TARGET_CATEGORY].fillna("unknown").astype(str).values).astype(np.int32)
    num_class = len(le_cat.classes_)
    ts(f"[INFO] Classes: {num_class}")

    ts("[INFO] Transforming features (this can take a moment)...")
    X_train = featurizer.transform(train_df[TEXT_COLS + CAT_COLS + NUM_COLS + BOOL_COLS])
    X_val   = featurizer.transform(val_df[TEXT_COLS + CAT_COLS + NUM_COLS + BOOL_COLS])
    ts(f"[INFO] Shapes — X_train: {X_train.shape}, X_val: {X_val.shape}")

    sample_weights = compute_class_weights(y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    ts("[INFO] Training XGBoost (native booster with early stopping)...")
    params = {
        "objective": "multi:softprob",
        "num_class": int(num_class),
        "max_depth": int(args.max_depth),
        "eta": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": 1.5,
        "reg_alpha": 0.1,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "seed": int(args.random_state),
        "verbosity": 1,
    }
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=int(args.n_estimators),
        evals=[(dval, "validation")],
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=args.eval_verbosity if args.eval_verbosity > 0 else False,
    )

    best_iter = getattr(bst, "best_iteration", None)
    ts(f"[INFO] Best iteration: {best_iter}")

    ts("[INFO] Evaluating on validation...")
    if best_iter is not None:
        proba_val = bst.predict(dval, iteration_range=(0, best_iter + 1))
    else:
        proba_val = bst.predict(dval)
    y_val_pred = np.argmax(proba_val, axis=1)
    f1_w = float(f1_score(y_val, y_val_pred, average="weighted"))
    report = classification_report(y_val, y_val_pred, target_names=le_cat.classes_, output_dict=True)

    ts("[INFO] Saving model + metrics...")
    booster_path = out_dir / "xgb_category.booster.json"
    bst.save_model(booster_path.as_posix())

    (out_dir / "training_meta.json").write_text(json.dumps({
        "model_type": "xgboost_booster",
        "params": params,
        "label_classes": le_cat.classes_.tolist(),
        "best_iteration": best_iter
    }, indent=2), encoding="utf-8")
    (out_dir / "val_metrics.json").write_text(json.dumps({
        "f1_weighted": f1_w,
        "classification_report": report
    }, indent=2), encoding="utf-8")

    ts(f"[OK] Done. Val F1-weighted = {f1_w:.4f}")

if __name__ == "__main__":
    main()
