"""Stream 1: Inspect synthetic_systems training data for quality issues.

Samples 20 entries spread across all font styles and sub-corpora, dumps
their image paths and token sequences for manual review, and computes
aggregate stats. Output JSON includes per-sample data so a human can
spot-check a handful.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.inspect_synthetic_samples \\
        --manifest data\\processed\\synthetic_v2\\manifests\\synthetic_token_manifest.jsonl \\
        --out audit_results\\synthetic_inspection.json \\
        --n 20
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from collections import Counter

_REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    # Group entries by (style_id, dataset) so we can sample evenly
    by_group = {}
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            key = (e.get("style_id", "?"), e.get("dataset", "?"))
            by_group.setdefault(key, []).append(e)

    print(f"Groups in manifest:")
    for key, entries in sorted(by_group.items()):
        print(f"  {key}: {len(entries)} entries")

    # Pick samples per group, distribute args.n total
    per_group = max(1, args.n // len(by_group))
    sampled = []
    for key, entries in sorted(by_group.items()):
        chosen = rng.sample(entries, min(per_group, len(entries)))
        chosen.sort(key=lambda e: e["sample_id"])
        sampled.extend(chosen)

    print(f"Sampled {len(sampled)} entries total across {len(by_group)} groups")

    # Per-sample digest
    per_sample = []
    for e in sampled:
        toks = e.get("token_sequence", [])
        per_sample.append({
            "sample_id": e["sample_id"],
            "dataset": e.get("dataset"),
            "style_id": e.get("style_id"),
            "image_path": e.get("image_path"),
            "source_path": e.get("source_path"),
            "token_count": len(toks),
            "first_30": toks[:30],
            "last_10": toks[-10:],
            "n_note_tokens": sum(1 for t in toks if t.startswith("note-")),
            "n_rest_tokens": sum(1 for t in toks if t == "rest"),
            "n_staff_starts": sum(1 for t in toks if t == "<staff_start>"),
            "n_measure_starts": sum(1 for t in toks if t == "<measure_start>"),
        })

    # Aggregate stats
    total = len(per_sample) or 1
    stats = {
        "n_groups": len(by_group),
        "per_group_count": {f"{k[0]}|{k[1]}": len(v) for k, v in by_group.items()},
        "mean_token_count": sum(r["token_count"] for r in per_sample) / total,
        "mean_note_tokens": sum(r["n_note_tokens"] for r in per_sample) / total,
        "mean_staff_starts": sum(r["n_staff_starts"] for r in per_sample) / total,
        "mean_measure_starts": sum(r["n_measure_starts"] for r in per_sample) / total,
        "multi_staff_fraction": sum(1 for r in per_sample if r["n_staff_starts"] > 1) / total,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "synthetic_inspection",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "stats": stats,
        "per_sample": per_sample,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(f"\nStats:")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
