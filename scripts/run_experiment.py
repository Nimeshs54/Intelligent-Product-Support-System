from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import strftime
import argparse

from mlops.tracker import Tracker

MODELS_DIR = Path("data/artifacts/models")


def _read_metric(path: Path, key: str, default=None):
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get(key, default)
    except Exception:
        return default


def _run(cmd: list[str]) -> None:
    print(f"[RUNEXP] RUN: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_xgb(passthrough_args: list[str]):
    _run([sys.executable, "-m", "models.train_xgb", *passthrough_args])

    val_f1 = _read_metric(MODELS_DIR / "val_metrics.json", "f1_weighted")
    test_f1 = _read_metric(MODELS_DIR / "test_metrics.json", "f1_weighted")
    version = f"v{strftime('%Y%m%d_%H%M%S')}"
    booster_path = MODELS_DIR / "xgb_category.booster.json"

    return {
        "metrics": {"val_f1_weighted": val_f1, "test_f1_weighted": test_f1},
        "artifacts": {
            "xgb_booster": booster_path.as_posix(),
            "val_metrics.json": (MODELS_DIR / "val_metrics.json").as_posix(),
            "test_metrics.json": (MODELS_DIR / "test_metrics.json").as_posix(),
            "confusion_matrix.png": (MODELS_DIR / "confusion_matrix.png").as_posix(),
        },
        "registry": {"name": "xgb_category", "version": version},
    }


def run_keras_text(passthrough_args: list[str]):
    _run([sys.executable, "-m", "models.train_keras_textonly", *passthrough_args])

    version = f"v{strftime('%Y%m%d_%H%M%S')}"
    return {
        "metrics": {},  # add if your trainer saves a metrics JSON
        "artifacts": {
            "keras_text_model": "data/artifacts/models/keras_text/model.keras",
            "tfidf.joblib": "data/artifacts/models/keras_text/tfidf.joblib",
            "svd.joblib": "data/artifacts/models/keras_text/svd.joblib",
            "scaler.joblib": "data/artifacts/models/keras_text/scaler.joblib",
        },
        "registry": {"name": "keras_text", "version": version},
    }


def main():
    print(f"[RUNEXP] cwd={Path.cwd().as_posix()}")

    # Only parse the required positional arg; everything else is passed through
    base = argparse.ArgumentParser(description="Run experiment and track it.")
    base.add_argument("which", choices=["xgb", "keras_text"], help="experiment to run")
    args, passthrough = base.parse_known_args()

    component = "train_xgb" if args.which == "xgb" else "train_keras_text"
    t = Tracker(component)

    try:
        if args.which == "xgb":
            out = run_xgb(passthrough)
        else:
            out = run_keras_text(passthrough)

        if passthrough:
            t.log_params(passthrough=" ".join(passthrough))

        if out.get("metrics"):
            t.log_metrics(**out["metrics"])

        for name, path in out.get("artifacts", {}).items():
            t.log_artifact(name, path)

        reg = out["registry"]
        t.register_model(reg["name"], reg["version"], stage="Staging")
        t.promote_model(reg["name"], reg["version"], stage="Production")
        t.end("finished")

    except subprocess.CalledProcessError as e:
        print(f"[RUNEXP] TRAINING FAILED with return code {e.returncode}", file=sys.stderr, flush=True)
        t.end("failed")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
