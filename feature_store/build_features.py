"""
Build features from processed JSON splits.
Saves featurizer, label encoders, and transformed datasets to artifacts.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Directories
PROC_DIR = Path("data/processed")
ARTIF_DIR = Path("data/artifacts/features")

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
TARGET_SUBCATEGORY = "subcategory"

class TextConcatenator:
    """Concatenate multiple text columns into one string per row."""
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X[self.cols].fillna("").agg(" ".join, axis=1)

def build_featurizer():
    # Richer text features: bigrams, more vocab, sublinear TF, min_df to denoise
    text_pipe = Pipeline([
        ("concat", TextConcatenator(TEXT_COLS)),
        ("tfidf", TfidfVectorizer(
            max_features=120_000,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
            dtype=np.float32
        ))
    ])
    transformers = [
        ("text", text_pipe, TEXT_COLS),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
        ("num", StandardScaler(with_mean=False), NUM_COLS),
    ]
    return ColumnTransformer(transformers, sparse_threshold=0.3)

def fit_and_save(train_df, val_df, test_df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    featurizer = build_featurizer()
    X_train = featurizer.fit_transform(train_df)
    X_val   = featurizer.transform(val_df)
    X_test  = featurizer.transform(test_df)

    joblib.dump(featurizer, out_dir / "featurizer.joblib", compress=3)

    # Labels
    le_cat = LabelEncoder()
    y_train_cat = le_cat.fit_transform(train_df[TARGET_CATEGORY].fillna("unknown"))
    y_val_cat   = le_cat.transform(val_df[TARGET_CATEGORY].fillna("unknown"))
    y_test_cat  = le_cat.transform(test_df[TARGET_CATEGORY].fillna("unknown"))
    joblib.dump(le_cat, out_dir / "label_encoder_category.joblib", compress=3)

    le_sub = LabelEncoder()
    y_train_sub = le_sub.fit_transform(train_df[TARGET_SUBCATEGORY].fillna("unknown"))
    y_val_sub   = le_sub.transform(val_df[TARGET_SUBCATEGORY].fillna("unknown"))
    y_test_sub  = le_sub.transform(test_df[TARGET_SUBCATEGORY].fillna("unknown"))
    joblib.dump(le_sub, out_dir / "label_encoder_subcategory.joblib", compress=3)

    # Save arrays (optional, we mainly load the featurizer later)
    np.savez_compressed(out_dir / "train_arrays.npz", X=X_train, y_cat=y_train_cat, y_sub=y_train_sub)
    np.savez_compressed(out_dir / "val_arrays.npz",   X=X_val,   y_cat=y_val_cat,   y_sub=y_val_sub)
    np.savez_compressed(out_dir / "test_arrays.npz",  X=X_test,  y_cat=y_test_cat,  y_sub=y_test_sub)

    meta = {
        "X_shape": {"train": X_train.shape, "val": X_val.shape, "test": X_test.shape},
        "num_classes": {"category": len(le_cat.classes_), "subcategory": len(le_sub.classes_)},
        "classes_category": le_cat.classes_.tolist(),
        "classes_subcategory": le_sub.classes_.tolist(),
        "tfidf": {"max_features": 120_000, "ngram_range": [1, 2], "min_df": 3, "sublinear_tf": True}
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Feature artifacts saved to {out_dir}")

def load_split(name: str, in_dir: Path) -> pd.DataFrame:
    path = in_dir / f"{name}.json"
    df = pd.read_json(path, orient="records")
    for c in TEXT_COLS + CAT_COLS + BOOL_COLS + NUM_COLS + [TARGET_CATEGORY, TARGET_SUBCATEGORY]:
        if c not in df.columns:
            df[c] = np.nan if c not in BOOL_COLS else False
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=str(PROC_DIR))
    ap.add_argument("--out", type=str, default=str(ARTIF_DIR))
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)

    train_df = load_split("train", in_dir)
    val_df   = load_split("val", in_dir)
    test_df  = load_split("test", in_dir)

    fit_and_save(train_df, val_df, test_df, out_dir)

if __name__ == "__main__":
    main()
