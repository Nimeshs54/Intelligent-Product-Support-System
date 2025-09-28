# scripts/smoke_infer.py
"""
Smoke test: load deployed model + featurizer and run one prediction on a ticket JSON.
Usage:
  python -m scripts.smoke_infer tmp_ticket.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from models.infer import load_deployed_model

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.smoke_infer <ticket.json>")
        sys.exit(1)

    ticket_path = Path(sys.argv[1])
    if not ticket_path.exists():
        print(f"[ERROR] File not found: {ticket_path}")
        sys.exit(2)

    print("[1/3] Loading deployed model (booster + featurizer + label encoder)…")
    model = load_deployed_model()
    print("[OK] Model loaded:", model.name)

    print(f"[2/3] Reading ticket: {ticket_path}")
    ticket = json.loads(ticket_path.read_text(encoding="utf-8"))

    print("[3/3] Predicting…")
    out = model.predict(ticket)
    print("[OK] Prediction:")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
