from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
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

    print("[INFO] Using TEXT ONLY to probe tokens…")
    # Rebuild a text-only matrix by asking the saved ColumnTransformer for the 'text' part
    # The saved featurizer has named transformers -> we can access the fitted tfidf vocab
    ct = featurizer
    text_pipe = dict(ct.named_transformers_)["text"]
    tfidf = text_pipe.named_steps["tfidf"]
    # Build text-only inputs:
    Xtr_text = tfidf.transform(text_pipe.named_steps["concat"].transform(tr))
    Xva_text = tfidf.transform(text_pipe.named_steps["concat"].transform(va))

    ytr = le.transform(tr[TARGET].fillna("unknown").astype(str).values)
    yva = le.transform(va[TARGET].fillna("unknown").astype(str).values)

    print("[INFO] Training a quick multinomial logistic regression on TEXT ONLY…")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, penalty="l2", C=1.0, solver="lbfgs", multi_class="multinomial")
    clf.fit(Xtr_text, ytr)
    yhat = clf.predict(Xva_text)
    f1w = f1_score(yva, yhat, average="weighted")
    print(f"[OK] Text-only baseline F1: {f1w:.4f}")

    print("[INFO] Top tokens per class:")
    inv_vocab = np.array([t for t,_ in sorted(tfidf.vocabulary_.items(), key=lambda kv: kv[1])])
    for i, cls in enumerate(le.classes_):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[-10:][::-1]
        print(f"  - {cls}: " + ", ".join(inv_vocab[top_idx]))

if __name__=="__main__":
    main()
