#!/usr/bin/env bash
set -euo pipefail

echo "🐍 Using Python: $(command -v python)"

echo "[0/8] Installing requirements (if needed)..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

has_module () { python - <<PY
import importlib, sys
sys.exit(0 if importlib.util.find_spec("$1") else 1)
PY
}

echo "[1/8] Running data checks..."
python -m scripts.run_data_checks

echo "[2/8] Bootstrapping splits + features..."
python -m pipelines.bootstrap

echo "[3/8] Training XGBoost model..."
if has_module xgboost; then
  python -m scripts.run_experiment xgb --limit_train 20000 --n_estimators 300 --max_depth 6 --learning_rate 0.1
else
  echo "[WARN] xgboost not installed; skipping XGB training."
fi

echo "[4/8] Training Keras text-only model..."
if has_module tensorflow; then
  python -m scripts.run_experiment keras_text --limit_train 10000 --svd_dim 256 --epochs 5 --batch_size 128
else
  echo "[WARN] tensorflow not installed; skipping Keras text model."
fi

echo "[5/8] Building KB + support graph..."
python -m retrieval.build_kb

echo "[6/8] Computing analytics..."
python -m analytics.compute_metrics

echo "[7/8] Running anomaly scan..."
python -m scripts.run_anomaly_scan --window 14 --z 3 --ret_fail_threshold 0.10

echo "[8/8] Writing experiment summary CSV..."
python -m scripts.inspect_runs

echo "✅ One-shot build complete! Artifacts ready in data/artifacts"
