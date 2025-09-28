# scripts/sanity_label_shuffle.py
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, joblib, json
import xgboost as xgb
from sklearn.metrics import f1_score
from feature_store.build_features import TextConcatenator  # noqa: F401

FEAT_DIR = Path("data/artifacts/features")
PROC_DIR = Path("data/processed")

TEXT_COLS = ["subject","description","error_logs","stack_trace"]
CAT_COLS  = ["product","product_module","language","region","priority","severity","customer_tier","channel","environment"]
BOOL_COLS = ["contains_error_code","contains_stack_trace","weekend_ticket","after_hours","known_issue","bug_report_filed","auto_suggestion_accepted","resolution_helpful","escalated"]
NUM_COLS  = ["previous_tickets","account_age_days","account_monthly_value","similar_issues_last_30_days","product_version_age_days","ticket_text_length","response_count","attachments_count","affected_users","resolution_time_hours","resolution_attempts","agent_experience_months","transferred_count","satisfaction_score"]
TARGET = "category"

def load_split(name:str)->pd.DataFrame:
    df = pd.read_json(PROC_DIR/f"{name}.json", orient="records")
    for c in TEXT_COLS+CAT_COLS+BOOL_COLS+NUM_COLS+[TARGET]:
        if c not in df.columns:
            df[c] = np.nan if c not in BOOL_COLS else False
    for c in TEXT_COLS: df[c] = df[c].fillna("")
    for c in BOOL_COLS: df[c] = df[c].astype("boolean").astype(float)
    return df

def main():
    print("[INFO] Loading artifacts…")
    featurizer = joblib.load(FEAT_DIR/"featurizer.joblib")
    le = joblib.load(FEAT_DIR/"label_encoder_category.joblib")

    print("[INFO] Loading train/val…")
    tr = load_split("train"); va = load_split("val")

    print("[INFO] Transforming features…")
    Xtr = featurizer.transform(tr[TEXT_COLS+CAT_COLS+NUM_COLS+BOOL_COLS])
    Xva = featurizer.transform(va[TEXT_COLS+CAT_COLS+NUM_COLS+BOOL_COLS])

    ytr = le.transform(tr[TARGET].fillna("unknown").astype(str).values).astype(np.int32)
    yva = le.transform(va[TARGET].fillna("unknown").astype(str).values).astype(np.int32)

    print("[INFO] Shuffling labels on train…")
    rng = np.random.default_rng(42)
    ytr_shuf = rng.permutation(ytr)

    dtr = xgb.DMatrix(Xtr, label=ytr_shuf)
    dva = xgb.DMatrix(Xva, label=yva)
    params = dict(objective="multi:softprob", num_class=len(le.classes_), max_depth=8, eta=0.08,
                  subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss", seed=42)
    print("[INFO] Training on SHUFFLED labels…")
    bst = xgb.train(params, dtr, num_boost_round=400, evals=[(dva,"validation")], verbose_eval=50)

    print("[INFO] Evaluating…")
    yhat = bst.predict(dva).argmax(axis=1)
    f1w = float(f1_score(yva, yhat, average="weighted"))
    print(f"[OK] Sanity F1 (should be ~0.20): {f1w:.4f}")

if __name__=="__main__":
    main()
