
from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List

SPLIT_DIR = Path("data/processed")

def load_json(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def summarize_split(name: str) -> Counter:
    data = load_json(SPLIT_DIR / f"{name}.json")
    cat_counter = Counter((r.get("category") or "").strip() for r in data)
    # normalize empty keys
    if "" in cat_counter:
        cat_counter["<MISSING>"] = cat_counter.pop("")
    return cat_counter

def main():
    ap = argparse.ArgumentParser(description="List available categories in processed splits.")
    ap.add_argument("--split", choices=["train", "val", "test"], help="Only show one split")
    ap.add_argument("--top", type=int, default=None, help="Show top-N by frequency")
    ap.add_argument("--csv", action="store_true", help="Output CSV (split,category,count)")
    args = ap.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]

    # collect stats
    by_split = OrderedDict()
    all_counter = Counter()
    for s in splits:
        c = summarize_split(s)
        by_split[s] = c
        all_counter.update(c)

    if args.csv:
        print("split,category,count")
        for s, c in by_split.items():
            items = c.most_common(args.top) if args.top else c.items()
            for cat, cnt in items:
                print(f"{s},{cat},{cnt}")
        return

    # pretty text output
    print("=== Categories by split ===")
    for s, c in by_split.items():
        total = sum(c.values())
        print(f"\n[{s}] total={total}")
        items = c.most_common(args.top) if args.top else c.most_common()
        for cat, cnt in items:
            print(f"  - {cat}: {cnt}")

    print("\n=== Union (all splits) ===")
    items = all_counter.most_common(args.top) if args.top else all_counter.most_common()
    for cat, cnt in items:
        print(f"  - {cat}: {cnt}")

if __name__ == "__main__":
    main()
