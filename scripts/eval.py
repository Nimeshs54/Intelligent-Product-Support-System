"""
Evaluate the trained XGBoost booster (JSON) on the test split.
Outputs:
  data/artifacts/models/
    - test_metrics.json
    - confusion_matrix.png
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Ensure featurizer's custom transformer class is importable for joblib
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

def load_split(name: str) -> pd.DataFrame:
    df = pd.read_json(PROC_DIR / f"{name}.json", orient="records")
    for c in TEXT_COLS + CAT_COLS + BOOL_COLS + NUM_COLS + [TARGET_CATEGORY]:
        if c not in df.columns:
            df[c] = np.nan if c not in BOOL_COLS else False
    for c in TEXT_COLS:
        df[c] = df[c].fillna("")
    for c in BOOL_COLS:
        df[c] = df[c].astype("boolean").astype(float)
    return df

def plot_confusion(cm: np.ndarray, labels: list[str], out_png: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.4), max(4, len(labels)*0.4)))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label", title="Confusion Matrix (Test)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    print("[INFO] Loading artifacts...")
    featurizer = joblib.load(FEAT_DIR / "featurizer.joblib")
    le_cat     = joblib.load(FEAT_DIR / "label_encoder_category.joblib")

    booster_path = MODEL_DIR / "xgb_category.booster.json"
    if not booster_path.exists():
        raise FileNotFoundError(f"Booster file not found: {booster_path}")
    booster = xgb.Booster()
    booster.load_model(booster_path.as_posix())

    print("[INFO] Loading TEST split...")
    test_df = load_split("test")
    X_test = featurizer.transform(test_df[TEXT_COLS + CAT_COLS + NUM_COLS + BOOL_COLS])
    y_test_str = test_df[TARGET_CATEGORY].fillna("unknown").astype(str).values
    y_test = le_cat.transform(y_test_str)

    print("[INFO] Running inference on TEST...")
    dtest = xgb.DMatrix(X_test)
    y_proba = booster.predict(dtest)
    y_pred = np.argmax(y_proba, axis=1)

    print("[INFO] Computing metrics...")
    f1_w = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(
        y_test, y_pred,
        target_names=le_cat.classes_,
        output_dict=True,
        zero_division=0  # avoid warnings for unseen classes
    )
    cm = confusion_matrix(y_test, y_pred)

    (MODEL_DIR / "test_metrics.json").write_text(json.dumps({
        "f1_weighted": f1_w,
        "classification_report": report
    }, indent=2), encoding="utf-8")
    plot_confusion(cm, le_cat.classes_.tolist(), MODEL_DIR / "confusion_matrix.png")

    print(f"[OK] Test F1-weighted: {f1_w:.4f}")
    print(f"[OK] Saved metrics to {MODEL_DIR}")

if __name__ == "__main__":
    main()
