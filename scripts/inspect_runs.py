from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from mlops.tracker import Tracker

EXP_DIR = Path("data/artifacts/experiments")
RUNS_PATH = EXP_DIR / "runs.jsonl"
REG_PATH = EXP_DIR / "registry.json"
OUT_CSV = EXP_DIR / "experiments_summary.csv"


def _load_runs():
    print(f"[INSPECT] Reading runs log: {RUNS_PATH}")
    runs = []
    if not RUNS_PATH.exists():
        print("[INSPECT] No runs.jsonl found.")
        return runs
    with RUNS_PATH.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON at line {ln}: {e}")
    print(f"[INSPECT] Loaded {len(runs)} runs.")
    return runs


def _load_registry():
    print(f"[INSPECT] Reading registry: {REG_PATH}")
    if REG_PATH.exists():
        try:
            return json.loads(REG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to parse registry.json: {e}")
            return {}
    print("[INSPECT] No registry.json found.")
    return {}


def _to_frame(runs):
    rows = []
    for r in runs:
        m = r.get("metrics") or {}
        p = r.get("params") or {}
        rows.append(
            {
                "run_id": r.get("run_id") or r.get("id"),
                "when": r.get("when"),
                "component": r.get("component"),
                "status": r.get("status"),
                "val_f1_weighted": m.get("val_f1_weighted"),
                "test_f1_weighted": m.get("test_f1_weighted"),
                # common hyperparams if present
                "limit_train": p.get("limit_train"),
                "n_estimators": p.get("n_estimators"),
                "max_depth": p.get("max_depth"),
                "learning_rate": p.get("learning_rate"),
                "epochs": p.get("epochs"),
                "batch_size": p.get("batch_size"),
                "svd_dim": p.get("svd_dim"),
                "hidden": p.get("hidden"),
                "dropout": p.get("dropout"),
            }
        )
    return pd.DataFrame(rows)


def main():
    print(f"[MLOPS] start inspection; experiments dir = {EXP_DIR.resolve()}")
    # Initialize tracker only to ensure dirs exist (does not start a new run)
    _ = Tracker("inspect")

    runs = _load_runs()
    if not runs:
        print("\n[INSPECT] No runs found. Example to create one:")
        print("  python -m scripts.run_experiment xgb -- --limit_train 20000 --n_estimators 300")
        return

    runs_sorted = sorted(runs, key=lambda r: r.get("when", ""), reverse=True)

    print("\n=== Recent Runs (max 10) ===")
    for r in runs_sorted[:10]:
        m = r.get("metrics") or {}
        print(
            f"{r.get('when')} | {r.get('component')} "
            f"| val_f1={m.get('val_f1_weighted')} "
            f"| test_f1={m.get('test_f1_weighted')} "
            f"| id={r.get('run_id') or r.get('id')}"
        )

    registry = _load_registry()
    print("\n=== Model Registry ===")
    if registry:
        for name, ver in registry.items():
            print(f"{name}: {ver}")
    else:
        print("(empty)")

    print("\n[INSPECT] Building CSV summary…")
    df = _to_frame(runs_sorted)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INSPECT] Saved CSV -> {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
