"""Add a val split to a synthetic manifest by carving out source pieces.

Reads the manifest, picks ~5% of unique source_paths to be the val set, and
writes the manifest back with those entries marked split=val (others stay
split=train). All crops/systems derived from one source MusicXML stay in
the same split, which prevents leakage between fonts that share the same
source piece.

Usable against either the per-staff synthetic_v2 manifest (135,610 rows) or
the per-system synthetic_systems_v1 manifest (20,583 rows). The current Stage 3
training pipeline consumes the systems_v1 manifest, so the immediate
target is:

    venv-cu132\\Scripts\\python -m scripts.data.add_val_split_synthetic \\
        --manifest data/processed/synthetic_systems_v1/manifests/synthetic_token_manifest.jsonl \\
        --backup data/processed/synthetic_systems_v1/manifests/synthetic_token_manifest.no_val.jsonl

After running, re-run `scripts/build_stage3_combined_manifest.py` to propagate
the new split to `src/data/manifests/token_manifest_stage3.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to the synthetic manifest to modify in place.")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Fraction of unique source_paths to assign to val (default 5%%).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible source selection.")
    p.add_argument("--backup", type=Path, required=True,
                   help="Path to write an unmodified backup of the original manifest.")
    args = p.parse_args()

    shutil.copyfile(args.manifest, args.backup)
    print(f"Backed up original manifest to {args.backup}")

    entries: list[dict] = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            entries.append(json.loads(line))
    print(f"Read {len(entries)} entries")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_source[e.get("source_path") or ""].append(e)

    sources = sorted(s for s in by_source if s)
    if not sources:
        raise SystemExit("No entries have a non-empty source_path; cannot do by-source split.")
    print(f"{len(sources)} unique source_paths")

    rng = random.Random(args.seed)
    n_val = max(1, int(round(len(sources) * args.val_fraction)))
    val_sources = set(rng.sample(sources, n_val))
    print(f"Selecting {n_val} sources for val ({100 * n_val / len(sources):.2f}% of sources)")

    n_changed = 0
    for e in entries:
        sp = e.get("source_path")
        if sp in val_sources:
            if e.get("split") != "val":
                e["split"] = "val"
                n_changed += 1

    with args.manifest.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"Reassigned {n_changed} entries to split=val")

    counts = Counter(e.get("split") for e in entries)
    print(f"Final split counts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
