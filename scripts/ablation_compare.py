from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, joblib, xgboost as xgb
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

def train_eval(Xtr, ytr, Xva, yva, label):
    dtr, dva = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xva, label=yva)
    params = dict(objective="multi:softprob", num_class=len(np.unique(ytr)), max_depth=8, eta=0.1,
                  subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss", seed=42)
    bst = xgb.train(params, dtr, num_boost_round=400, evals=[(dva,"val")], verbose_eval=False)
    yhat = bst.predict(dva).argmax(axis=1)
    f1w = f1_score(yva, yhat, average="weighted")
    print(f"[OK] {label} F1: {f1w:.4f}")

def main():
    print("[INFO] Loading artifacts…")
    featurizer = joblib.load(FEAT_DIR/"featurizer.joblib")
    le = joblib.load(FEAT_DIR/"label_encoder_category.joblib")

    print("[INFO] Loading train/val…")
    tr = load_split("train"); va = load_split("val")

    # text only
    ct = featurizer
    text_pipe = dict(ct.named_transformers_)["text"]
    tfidf = text_pipe.named_steps["tfidf"]
    Xtr_text = tfidf.transform(text_pipe.named_steps["concat"].transform(tr))
    Xva_text = tfidf.transform(text_pipe.named_steps["concat"].transform(va))

    ytr = le.transform(tr[TARGET].fillna("unknown").astype(str).values)
    yva = le.transform(va[TARGET].fillna("unknown").astype(str).values)

    # non-text only
    Xtr_full = featurizer.transform(tr[TEXT_COLS+CAT_COLS+NUM_COLS+BOOL_COLS])
    Xva_full = featurizer.transform(va[TEXT_COLS+CAT_COLS+NUM_COLS+BOOL_COLS])
    # subtract text block from the full sparse matrix
    # the ColumnTransformer puts 'text' first, so:
    n_text = Xtr_text.shape[1]
    Xtr_ntext = Xtr_full[:, n_text:]
    Xva_ntext = Xva_full[:, n_text:]

    print("[INFO] Training ablations…")
    train_eval(Xtr_text, ytr, Xva_text, yva, "TEXT-ONLY")
    train_eval(Xtr_ntext, ytr, Xva_ntext, yva, "NON-TEXT-ONLY")

if __name__=="__main__":
    main()
