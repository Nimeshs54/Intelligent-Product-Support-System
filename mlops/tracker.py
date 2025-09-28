from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional


EXP_DIR = Path("data/artifacts/experiments")
RUNS_PATH = EXP_DIR / "runs.jsonl"
REG_SIMPLE_PATH = EXP_DIR / "registry.json"         # simple: {model_name: version}
REG_FULL_PATH = EXP_DIR / "registry_full.json"      # detailed history


def _now_iso() -> str:
    # ISO-like timestamp without TZ (simple and portable)
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class RunRecord:
    run_id: str
    when: str
    component: str
    status: str = "running"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path


class Tracker:
    """
    Minimal, file-based experiment tracker.

    - Appends JSONL entries to runs.jsonl
    - Keeps a simple model registry in registry.json (name -> version)
      and a detailed registry in registry_full.json
    """
    def __init__(self, component: str) -> None:
        self.component = component
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        self.run = RunRecord(run_id=uuid.uuid4().hex, when=_now_iso(), component=component)

        # print a start banner
        print(f"[MLOPS] start run={self.run.run_id} component={self.component}")

    # -------- logging helpers --------
    def log_params(self, **kwargs: Any) -> None:
        self.run.params.update(kwargs)

    def log_metrics(self, **kwargs: Any) -> None:
        self.run.metrics.update(kwargs)

    def log_artifact(self, name: str, path: str) -> None:
        # store relative path for portability
        self.run.artifacts[name] = path.replace("\\", "/")

    def set_status(self, status: str) -> None:
        self.run.status = status

    def _append_run(self) -> None:
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        with RUNS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(self.run), ensure_ascii=False) + "\n")

    # -------- registry helpers --------
    def register_model(self, model_name: str, version: str, stage: str = "Staging", extra: Optional[Dict[str, Any]] = None) -> None:
        EXP_DIR.mkdir(parents=True, exist_ok=True)

        # simple mapping
        simple = {}
        if REG_SIMPLE_PATH.exists():
            try:
                simple = json.loads(REG_SIMPLE_PATH.read_text(encoding="utf-8"))
            except Exception:
                simple = {}

        # detailed
        detailed = {}
        if REG_FULL_PATH.exists():
            try:
                detailed = json.loads(REG_FULL_PATH.read_text(encoding="utf-8"))
            except Exception:
                detailed = {}

        # update detailed history
        det = detailed.setdefault(model_name, {"versions": []})
        det["versions"].append({
            "version": version,
            "stage": stage,
            "registered_at": _now_iso(),
            "by_run": self.run.run_id,
            "component": self.component,
            "extra": extra or {},
        })

        # simple file keeps the latest version pointer for convenience (still stage-aware in full file)
        simple[model_name] = version

        REG_SIMPLE_PATH.write_text(json.dumps(simple, indent=2), encoding="utf-8")
        REG_FULL_PATH.write_text(json.dumps(detailed, indent=2), encoding="utf-8")
        print(f"[REG] registered {model_name}:{version} stage={stage}")

    def promote_model(self, model_name: str, version: str, stage: str = "Production") -> None:
        # update detailed registry to mark a version as Production (and demote others)
        if not REG_FULL_PATH.exists():
            print(f"[REG] no full registry yet; cannot promote {model_name}:{version}")
            return

        detailed = json.loads(REG_FULL_PATH.read_text(encoding="utf-8"))
        if model_name not in detailed:
            print(f"[REG] model {model_name} not found in registry.")
            return

        for v in detailed[model_name]["versions"]:
            if v["version"] == version:
                v["stage"] = stage
            else:
                # demote others to Archived if they were Production
                if v.get("stage") == "Production":
                    v["stage"] = "Archived"

        REG_FULL_PATH.write_text(json.dumps(detailed, indent=2), encoding="utf-8")
        # keep simple pointer to the *promoted* version
        simple = {}
        if REG_SIMPLE_PATH.exists():
            simple = json.loads(REG_SIMPLE_PATH.read_text(encoding="utf-8"))
        simple[model_name] = version
        REG_SIMPLE_PATH.write_text(json.dumps(simple, indent=2), encoding="utf-8")

        print(f"[REG] promoted {model_name}:{version} -> {stage}")

    def end(self, status: str = "finished") -> None:
        self.set_status(status)
        self._append_run()
        print(f"[MLOPS] status: {status}")
        print(f"[MLOPS] end run={self.run.run_id} saved -> {RUNS_PATH.as_posix()}")
