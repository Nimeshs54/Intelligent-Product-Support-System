# ==========================
# One-shot build script (Windows PowerShell)
# Robust + verbose + safe fallbacks for missing deps
# ==========================

$ErrorActionPreference = "Stop"
function Log($msg, $color="Gray") { Write-Host $msg -ForegroundColor $color }

# Pick the right python for this shell
$Python = $null
if ($env:CONDA_PREFIX -and (Test-Path "$env:CONDA_PREFIX\python.exe")) {
  $Python = "$env:CONDA_PREFIX\python.exe"
} else {
  $Python = (Get-Command python).Path
}
Log "🐍 Using Python: $Python" "Cyan"

# 0) Install requirements
Log "[0/8] Installing requirements (if needed)..." "Yellow"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

# Tiny helper to test if a module exists
function HasModule($mod) {
  $cmd = "$Python - << 'PY'\nimport importlib, sys\nsys.exit(0 if importlib.util.find_spec('$mod') else 1)\nPY"
  cmd /c $cmd | Out-Null
  return ($LASTEXITCODE -eq 0)
}

# 1) Data checks
Log "[1/8] Running data checks..." "Yellow"
& $Python -m scripts.run_data_checks

# 2) Bootstrap (splits + features)
Log "[2/8] Bootstrapping splits + features..." "Yellow"
& $Python -m pipelines.bootstrap

# 3) Train XGBoost (skip if xgboost missing)
Log "[3/8] Training XGBoost model..." "Yellow"
if (HasModule "xgboost") {
  & $Python -m scripts.run_experiment xgb --limit_train 20000 --n_estimators 300 --max_depth 6 --learning_rate 0.1
} else {
  Log "[WARN] xgboost not installed; skipping XGB training." "DarkYellow"
}

# 4) Train Keras text-only (skip if tensorflow missing)
Log "[4/8] Training Keras text-only model..." "Yellow"
if (HasModule "tensorflow") {
  & $Python -m scripts.run_experiment keras_text --limit_train 10000 --svd_dim 256 --epochs 5 --batch_size 128
} else {
  Log "[WARN] tensorflow not installed; skipping Keras text model." "DarkYellow"
}

# 5) Build Knowledge Base + Graph-RAG
Log "[5/8] Building KB + support graph..." "Yellow"
& $Python -m retrieval.build_kb

# 6) Compute analytics
Log "[6/8] Computing analytics..." "Yellow"
& $Python -m analytics.compute_metrics

# 7) Run anomaly detection
Log "[7/8] Running anomaly scan..." "Yellow"
& $Python -m scripts.run_anomaly_scan --window 14 --z 3 --ret_fail_threshold 0.10

# 8) Quick experiment registry summary (CSV)
Log "[8/8] Writing experiment summary CSV..." "Yellow"
& $Python -m scripts.inspect_runs

Log "✅ One-shot build complete! Artifacts ready in data/artifacts" "Green"
