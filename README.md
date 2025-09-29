# 🧠 Intelligent Product Support System

An end-to-end AI-driven support assistant that learns from 100,000+ historical tickets to:

End-to-end system that:
- **Understands** tickets (category, priority, sentiment)
- **Retrieves** similar historical resolutions (Hybrid RAG + metadata)
- **Detects** emerging issues (daily anomaly scan)
- **Serves** results via a **FastAPI** backend + lightweight **HTML** demo
- Includes **reproducible training**, **mini MLOps tracking**, and **containerization**

---
## Project Structure

```plaintext
Intelligent_Product_Support_System/
├─ analytics/
│  └─ compute_metrics.py
├─ data/
│  ├─ artifacts/
│  │  ├─ analytics/                # parquet metrics, charts, data_checks.json
│  │  ├─ anomalies/                # anomaly report.json
│  │  ├─ experiments/              # MLOps tracker + registry
│  │  │  ├─ runs.jsonl
│  │  │  ├─ experiments_summary.csv
│  │  │  └─ registry.json
│  │  ├─ features/                 # featurizer.joblib, label encoders
│  │  ├─ models/                   # xgb_category.booster.json, keras_text/
│  │  │  └─ keras_text/
│  │  │     ├─ model.keras
│  │  │     ├─ tfidf.joblib
│  │  │     ├─ svd.joblib
│  │  │     └─ scaler.joblib
│  │  └─ retrieval/                # tfidf.npz, vectorizer.joblib, docs_meta.parquet
│  ├─ processed/                   # train.json, val.json, test.json
│  ├─ sample/
│  │  └─ mini_support_tickets.json
│  └─ .gitkeep
├─ Dockerfile
├─ docker-compose.yml
├─ environment.yml
├─ feature_store/
│  └─ build_features.py            # builds sklearn ColumnTransformer + encoders
├─ mlops/
│  ├─ registry.py                  # simple JSON registry helper
│  └─ tracker.py                   # JSONL run tracker (start/end, metrics, artifacts)
├─ models/
│  ├─ infer.py                     # load deployed booster + featurizer; predict
│  ├─ pickle_shims.py              # safe unpickling shims for custom transformers
│  ├─ train_keras_textonly.py      # Keras (TF-IDF+SVD) baseline
│  └─ train_xgb.py                 # XGBoost multiclass classifier
├─ monitoring/
│  └─ anomaly_rules.py             # rolling z-score based detectors
├─ pipelines/
│  └─ bootstrap.py                 # split -> features -> xgb -> eval artifacts
├─ retrieval/
│  ├─ build_kb.py                  # index construction + support graph
│  └─ query.py                     # category-aware TF-IDF retrieval
├─ scripts/
│  ├─ ablation_compare.py
│  ├─ build_all.ps1                # one-shot (Windows)
│  ├─ build_all.sh                 # one-shot (Bash)
│  ├─ eval.py
│  ├─ inspect_runs.py
│  ├─ inspect_tokens.py
│  ├─ remake_splits_grouped.py
│  ├─ run_anomaly_scan.py
│  ├─ run_data_checks.py
│  ├─ run_experiment.py
│  ├─ sample_ticket.py
│  ├─ sanity_label_shuffle.py
│  └─ smoke_infer.py
├─ service/
│  ├─ api.py                       # FastAPI endpoints + static mount
│  └─ static/
│     └─ demo/
│        └─ demo.html              # simple UI to post to /solve
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---
## 🧭 Quick Start (One-Shot Build)

If you just want to see it work with the provided `data/support_tickets.json`, run one of these:

### Option A — Docker (recommended)

```bash
# 1) Build image
docker build -t ips-system:latest .

# 2) Ensure dataset is present locally
#    Copy the provided JSON array to ./data/support_tickets.json
ls -lh data/support_tickets.json

# 3) Run container with volume mount (so artifacts persist to ./data)
docker run --rm -it -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  ips-system:latest

# FastAPI: http://127.0.0.1:8000
# Demo UI: http://127.0.0.1:8000/demo/demo.html

```


## 🚀 Manual Setup (Full Command Guide)

### Option B: Local Environment
```bash
# ── 1) Clone the repo ─────────────────────────────────────────────────────────
git clone https://github.com/Nimeshs54/Intelligent-Product-Support-System.git
cd Intelligent_Product_Support_System

# ── 2) Environment Setup ──────────────────────────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate              # (Windows PowerShell:  .\.venv\Scripts\Activate.ps1)
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow-cpu==2.15.0 xgboost==2.0.3

# ── 3) Dataset ────────────────────────────────────────────────────────────────
# Place the provided JSON array at:
# data/support_tickets.json
# (The pipeline expects a single JSON array with 100k objects.)

# ── 4) Data Checks ────────────────────────────────────────────────────────────
python -m scripts.run_data_checks
# -> data/artifacts/analytics/data_checks.json

# ── 5) Bootstrap (split + features + baseline train + test eval) ──────────────
python -m pipelines.bootstrap
# Writes:
#   - data/processed/train.json, val.json, test.json
#   - data/artifacts/features/*.joblib
#   - data/artifacts/models/xgb_category.booster.json
#   - data/artifacts/models/(metrics images/json)

# ── 6) (Optional) Re-train XGBoost with custom params ─────────────────────────
python -m scripts.run_experiment xgb --limit_train 20000 --n_estimators 300 --max_depth 6 --learning_rate 0.1
# Artifacts + registry promoted to Production on success.

# ── 7) (Optional) Train compact Keras text-only model ─────────────────────────
python -m scripts.run_experiment keras_text --limit_train 10000 --svd_dim 256 --epochs 5 --batch_size 128
# Writes to data/artifacts/models/keras_text

# ── 8) Build Retrieval Knowledge Base ─────────────────────────────────────────
python -m retrieval.build_kb
# -> data/artifacts/retrieval/{tfidf.npz, vectorizer.joblib, docs_meta.parquet, support_graph.graphml}

# ── 9) Analytics & Anomaly Scan ───────────────────────────────────────────────
python -m analytics.compute_metrics
python -m scripts.run_anomaly_scan --window 14 --z 3 --ret_fail_threshold 0.10
# -> data/artifacts/anomalies/report.json

# ── 10) Smoke Inference (sanity) ──────────────────────────────────────────────
python -m scripts.sample_ticket --split test --idx 0 --out tmp_ticket.json
python -m scripts.smoke_infer tmp_ticket.json
python -m retrieval.query --ticket tmp_ticket.json --k 5

# ── 11) Serve API + Demo UI ───────────────────────────────────────────────────
uvicorn service.api:app --reload --port 8000
# Open: http://127.0.0.1:8000/demo/demo.html

# ── 12) Inspect MLOps runs / registry (optional) ──────────────────────────────
python -m scripts.inspect_runs
# CSV summary -> data/artifacts/experiments/experiments_summary.csv
```

### Option C — Local (Windows PowerShell)
```powershell
# 0) Python 3.9+
python --version

# 1) Virtual environment
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow-cpu==2.15.0 xgboost==2.0.3

# 3) Dataset
#    Place the provided JSON array as: data\support_tickets.json

# 4) One-shot build (idempotent)
powershell -ExecutionPolicy Bypass -File .\scripts\build_all.ps1

# 5) Run API
uvicorn service.api:app --reload --port 8000
# API:  http://127.0.0.1:8000
# Demo: http://127.0.0.1:8000/demo/demo.html
```

---

### Option D — Local (macOS/Linux Bash)
```bash
# 0) Python 3.9+
python3 --version

# 1) Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow-cpu==2.15.0 xgboost==2.0.3

# 3) Dataset
#    Place the provided JSON array at: data/support_tickets.json

# 4) One-shot build
bash scripts/build_all.sh

# 5) Run API
uvicorn service.api:app --reload --port 8000
# API:  http://127.0.0.1:8000
# Demo: http://127.0.0.1:8000/demo/demo.html
```
## 🌐 API Reference

### POST `/classify`
Predicts **category, priority, sentiment**.

**Request**
```json
{
  "ticket_id": "TK-2024-000002",
  "subject": "License upgrade needed for DataSync Pro",
  "description": "We need more seats...",
  "error_logs": "",
  "stack_trace": "",
  "language": "en",
  "region": "NA",
  "priority": "medium",
  "severity": "P2",
  "contains_error_code": false,
  "contains_stack_trace": false
}
```

**Response**
```json
{
  "category": "Account Management",
  "subcategory": null,
  "priority": "high",
  "sentiment": "neutral",
  "confidence": 0.9999,
  "model": "xgb_category_booster",
  "top_features": []
}
```

---

### POST `/retrieve?k=5`
Category-aware TF-IDF + metadata filters; re-ranks by resolution quality.

**Response (truncated)**
```json
{
  "predicted_category": "Account Management",
  "predicted_confidence": 0.9999,
  "results": [
    {
      "doc_id": "TK-2024-019933",
      "product": "DataSync Pro",
      "category": "Account Management",
      "subcategory": "License",
      "resolution_code": "PATCH_APPLIED",
      "quality": 0.73,
      "score": 0.4481,
      "preview": "[TITLE] TK-2024-019933\n[PRODUCT] DataSync Pro\n[CATEGORY] Account Management::License\n[RES_CODE] PATCH_APPLIED"
    }
  ]
}
```

---

### POST `/solve?k=5`
Convenience endpoint that runs **/classify + /retrieve** and adds a short summary.

**PowerShell example**
```powershell
(Invoke-WebRequest -Uri "http://127.0.0.1:8000/solve?k=5" `
  -Method POST -Headers @{"Content-Type"="application/json"} `
  -Body (Get-Content -Raw -Path "tmp_ticket.json")).Content
```

**cURL example**
```bash
curl -X POST "http://127.0.0.1:8000/solve?k=5" \
  -H "Content-Type: application/json" \
  --data-binary @tmp_ticket.json
```

## 🖥️ Demo

The web interface provides an interactive way to test ticket classification and retrieval.  
Below are example runs using **XGBoost** and **Keras DL** models.

---

### 🔹 XGBoost Classification + Retrieval
This example shows a support ticket classified by the **XGBoost model** and matched against the knowledge base with top retrieved cases.
<img width="1898" height="917" alt="xgb" src="https://github.com/user-attachments/assets/a9e5b9e7-9426-405c-b8d7-4915d8bf81fe" alt="XGBoost Demo"/>

---

### 🔹 Keras DL Classification + Retrieval
This example demonstrates the same ticket processed with the **Keras DL model**, highlighting lightweight text-only classification and retrieval results.


<img width="1903" height="928" alt="keras DL" src="https://github.com/user-attachments/assets/ba603262-1b59-4650-b191-a4a82cf115fd" alt="Keras DL Demo"/>


---

## 🧪 Reproducibility & Evaluation
- Deterministic bootstrap creates splits & features with fixed seeds.  
- JSONL tracker records runs with params/metrics; a simple registry promotes the latest successful model to **Production**.  
- Artifacts are written to `data/artifacts/**` and reused by the service.  

**Common evaluation commands**
```bash
# Baseline end-to-end build
python -m pipelines.bootstrap

# Re-train with custom params
python -m models.train_xgb --limit_train 20000 --n_estimators 300 --max_depth 6 --learning_rate 0.1
python -m models.train_keras_textonly --limit_train 10000 --svd_dim 256 --epochs 5 --batch_size 128

# Evaluate on test (if script provided)
python -m scripts.eval
```

---

## 🧠 Design Decisions & Trade-offs

**Dual Models**
- **XGBoost (tabular + text):** strong baseline, CPU-fast, interpretable.  
- **Keras text-only:** TF-IDF → SVD → small DNN; portable and lightweight.  
  - *Trade-off:* ignores structured fields but simple and robust.  

**Hybrid Retrieval (RAG-lite)**  
- TF-IDF vectors filtered by predicted category & product metadata; re-rank by resolution success/quality.  
- *Trade-off:* no external embeddings; fully offline, transparent, deterministic.  

**Anomaly Detection**  
- Rolling z-scores on daily volume, category share, sentiment shifts, retrieval failure proxies.  
- *Trade-off:* prioritize explainability and easy ops tuning over sophistication.  

**MLOps Minimalism**  
- Filesystem-based JSON tracker + registry; good enough for demo; easily swappable for MLflow/DVC.  

**Containerization**  
- CPU-only Docker image; reviewers can run without OS/driver mismatch concerns.  

---

## 🧩 Requirements
- **Python 3.9+**  
- **pip** (or conda)  
- Optional **Docker** (recommended for quick review)  

**Key packages** (see `requirements.txt`):  
- fastapi, uvicorn, pandas, numpy, scikit-learn, scipy, networkx, pyarrow  
- loguru, joblib, jinja2  
- xgboost==2.0.3, tensorflow-cpu==2.15.0  

---

## 🛠 Troubleshooting

**No module named 'xgboost' / tensorflow**  
```bash
pip install xgboost==2.0.3 tensorflow-cpu==2.15.0
```

**Parquet errors**  
```bash
pip install pyarrow
```

**PowerShell JSON posting**  
```powershell
-Body (Get-Content -Raw -Path "file.json")
```

**Retrieval shape mismatch**  
```bash
python -m pipelines.bootstrap && python -m retrieval.build_kb
```

**API says 404 for demo**  
- Ensure file exists: `service/static/demo/demo.html`  
- Restart server:  
  ```bash
  uvicorn service.api:app --reload --port 8000
  ```

