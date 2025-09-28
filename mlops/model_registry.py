# mlops/model_registry.py
from __future__ import annotations
import json, shutil, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

REG_DIR = Path("data/artifacts/registry")
REG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = REG_DIR / "models.json"

@dataclass
class ModelEntry:
    name: str
    version: str
    created_at: float
    stage: str
    files: Dict[str, str]
    metrics: Dict[str, float]

def _load() -> list[dict]:
    if not MANIFEST.exists(): return []
    return json.loads(MANIFEST.read_text(encoding="utf-8"))

def _save(entries: list[dict]):
    MANIFEST.write_text(json.dumps(entries, indent=2), encoding="utf-8")

def register_model(name: str, files: Dict[str, str], metrics: Dict[str, float], stage: str = "None", version: Optional[str] = None) -> ModelEntry:
    ts = time.strftime("%Y%m%d_%H%M%S")
    version = version or f"v{ts}"
    entry = ModelEntry(
        name=name, version=version, created_at=time.time(), stage=stage,
        files=files, metrics=metrics
    )
    entries = _load()
    entries.append(asdict(entry))
    _save(entries)
    print(f"[REG] registered {name}:{version} stage={stage}")
    return entry

def promote(name: str, version: str, stage: str):
    entries = _load()
    changed = False
    for e in entries:
        if e["name"] == name and e["version"] == version:
            e["stage"] = stage
            changed = True
        elif e["name"] == name and stage == "Production" and e["stage"] == "Production":
            e["stage"] = "Archived"
    _save(entries)
    if changed:
        print(f"[REG] promoted {name}:{version} -> {stage}")
    else:
        print(f"[REG] WARN no entry for {name}:{version}")

def latest_production(name: str) -> Optional[ModelEntry]:
    entries = _load()
    prod = [e for e in entries if e["name"] == name and e["stage"] == "Production"]
    if not prod: return None
    e = max(prod, key=lambda x: x["created_at"])
    return ModelEntry(**e)

def copy_to_stage_dir(entry: ModelEntry):
    stage_dir = REG_DIR / entry.name / entry.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    for _, rel in entry.files.items():
        src = Path(rel)
        dst = stage_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"[REG] mirrored files to {stage_dir}")
