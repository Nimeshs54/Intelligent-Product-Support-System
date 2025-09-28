"""
One-shot pipeline runner with live logs:

Steps
  1) Grouped split (prevents leakage across train/val/test)
  2) Build features (TF-IDF + OneHot + Standardize)
  3) Train XGBoost (native booster; early stopping handled in script)
  4) Evaluate on test

Usage
  python -m pipelines.bootstrap
  # Fast pass (env var) -> will pass --limit_train to trainer
  FAST=1 python -m pipelines.bootstrap
  # Or Windows PowerShell:
  $env:FAST="1"; python -m pipelines.bootstrap
"""
from __future__ import annotations
from pathlib import Path
from subprocess import Popen, PIPE
import os, sys
from loguru import logger

DATA_RAW = Path("data/support_tickets.json")
PROCESSED_DIR = Path("data/processed")
ARTIF_FEAT = Path("data/artifacts/features")
ARTIF_MODEL = Path("data/artifacts/models")

def sh_stream(cmd: list[str]):
    """
    Run a command and stream stdout/stderr live to the console.
    Adds -u for unbuffered Python.
    """
    if cmd and cmd[0] == "python":
        cmd.insert(1, "-u")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    logger.info("$ " + " ".join(cmd))
    with Popen(cmd, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True, env=env) as p:
        # Stream stdout
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        # Stream any stderr that’s left
        err = p.stderr.read()
        if err:
            sys.stderr.write(err)
            sys.stderr.flush()
        rc = p.wait()
        if rc != 0:
            raise SystemExit(f"Command failed ({rc}): {' '.join(cmd)}")

def ensure_paths():
    Path("data").mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIF_FEAT.mkdir(parents=True, exist_ok=True)
    ARTIF_MODEL.mkdir(parents=True, exist_ok=True)

def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    ensure_paths()
    if not DATA_RAW.exists():
        raise FileNotFoundError("Place your dataset at data/support_tickets.json")

    # Decide if we run a fast pass (limit training rows)
    fast = os.environ.get("FAST", "").strip() in {"1", "true", "True", "yes", "YES"}
    train_args = ["-m", "models.train_xgb"]
    if fast:
        # Smaller run to validate pipeline quickly
        train_args += ["--limit_train", "20000", "--n_estimators", "300", "--max_depth", "6", "--learning_rate", "0.1", "--eval_verbosity", "25"]
        logger.info("FAST=1 detected: training on a 20k sample with lighter settings.")
    else:
        # Full dataset but still reasonable defaults; trainer has early stopping
        train_args += ["--n_estimators", "700", "--max_depth", "8", "--learning_rate", "0.08", "--early_stopping_rounds", "50", "--eval_verbosity", "25"]

    logger.info("Step 1/4: Grouped split (deduplicated across splits)…")
    sh_stream(["python", "-m", "scripts.remake_splits_grouped", "--raw", str(DATA_RAW), "--outdir", str(PROCESSED_DIR)])

    logger.info("Step 2/4: Build features…")
    sh_stream(["python", "-m", "feature_store.build_features", "--in_dir", str(PROCESSED_DIR), "--out", str(ARTIF_FEAT)])

    logger.info("Step 3/4: Train model (XGBoost booster)…")
    sh_stream(["python", *train_args])

    logger.info("Step 4/4: Evaluate on TEST split…")
    sh_stream(["python", "-m", "scripts.eval"])

    logger.success("All steps complete. Artifacts:")
    logger.info(f"  Features: {ARTIF_FEAT}")
    logger.info(f"  Models  : {ARTIF_MODEL}")
    logger.info("Key files:")
    logger.info("  - data/artifacts/models/xgb_category.booster.json")
    logger.info("  - data/artifacts/models/val_metrics.json")
    logger.info("  - data/artifacts/models/test_metrics.json")
    logger.info("  - data/artifacts/models/confusion_matrix.png")

if __name__ == "__main__":
    main()
