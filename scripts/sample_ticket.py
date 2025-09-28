# scripts/sample_ticket.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

SPLIT_DIR = Path("data/processed")

def load_split(split: str):
    p = SPLIT_DIR / f"{split}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing split file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--idx", type=int, default=None, help="index within filtered pool (after --category filter)")
    ap.add_argument("--category", type=str, default=None, help="filter by ground-truth category")
    ap.add_argument("--out", type=str, default="tmp_ticket.json")
    args = ap.parse_args()

    data = load_split(args.split)

    pool = data
    if args.category:
        want = args.category.strip().lower()
        pool = [r for r in data if (r.get("category") or "").strip().lower() == want]
        if not pool:
            print(f"[ERR] No tickets with category={args.category!r} in split={args.split}", file=sys.stderr)
            sys.exit(2)

    if args.idx is None:
        idx = random.randrange(len(pool))
    else:
        if args.idx < 0 or args.idx >= len(pool):
            print(f"[ERR] idx out of range 0..{len(pool)-1}", file=sys.stderr)
            sys.exit(2)
        idx = args.idx

    rec = pool[idx]
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {args.out} (split={args.split}, idx={idx}, ticket_id={rec.get('ticket_id')}, category={rec.get('category')})")

if __name__ == "__main__":
    main()
