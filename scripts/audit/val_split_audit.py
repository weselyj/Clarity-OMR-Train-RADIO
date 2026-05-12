"""Stream 4: Verify val-split structure and check for source_path leakage
between train and val splits per corpus.

For synthetic_systems (which has zero val entries), proposes a split
methodology by counting source_path uniqueness.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.val_split_audit \\
        --manifest src\\data\\manifests\\token_manifest_stage3.jsonl \\
        --synthetic-manifest data\\processed\\synthetic_v2\\manifests\\synthetic_token_manifest.jsonl \\
        --out audit_results\\val_split_audit.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict


def _audit_corpus(rows: list) -> dict:
    """Check for source_path leakage between train and val."""
    by_split = defaultdict(set)
    for r in rows:
        sp = r.get("source_path", r.get("sample_id"))
        by_split[r.get("split", "?")].add(sp)
    train_set = by_split.get("train", set())
    val_set = by_split.get("val", set())
    overlap = train_set & val_set
    return {
        "n_train": sum(1 for r in rows if r.get("split") == "train"),
        "n_val": sum(1 for r in rows if r.get("split") == "val"),
        "n_test": sum(1 for r in rows if r.get("split") == "test"),
        "unique_source_paths_train": len(train_set),
        "unique_source_paths_val": len(val_set),
        "source_path_overlap_train_val": len(overlap),
        "overlap_sample": sorted(overlap)[:5],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="Combined Stage 3 training manifest")
    p.add_argument("--synthetic-manifest", type=Path, required=True,
                   help="Synthetic corpus's own manifest (for split-methodology recommendation)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # Audit combined manifest
    by_corpus = defaultdict(list)
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            by_corpus[e.get("dataset", "?")].append(e)

    combined_audit = {c: _audit_corpus(rows) for c, rows in sorted(by_corpus.items())}

    # Synthetic-only audit (since the combined manifest may have already
    # filtered out something)
    with args.synthetic_manifest.open(encoding="utf-8") as f:
        synthetic_rows = [json.loads(line) for line in f]
    synthetic_audit = _audit_corpus(synthetic_rows)

    # Split methodology: how many unique source_paths in synthetic? If we
    # split 5% by source, how many val pieces would that be?
    sources = {r.get("source_path") for r in synthetic_rows}
    sources.discard(None)
    n_sources = len(sources)
    crops_per_source = defaultdict(int)
    for r in synthetic_rows:
        crops_per_source[r.get("source_path")] += 1
    mean_crops = sum(crops_per_source.values()) / max(1, len(crops_per_source))
    target_val_fraction = 0.05
    target_val_sources = max(1, int(n_sources * target_val_fraction))
    target_val_entries = int(target_val_sources * mean_crops)

    split_recommendation = {
        "n_unique_synthetic_sources": n_sources,
        "mean_crops_per_source": mean_crops,
        "recommended_val_sources": target_val_sources,
        "estimated_val_entries": target_val_entries,
        "methodology": "by_source_path",
        "rationale": ("Split by source_path so all fonts/sub-corpora derived "
                       "from one source MusicXML file go to one split. "
                       "Prevents leakage when the same source is rendered in "
                       "multiple fonts."),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "val_split_audit",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "combined_manifest_audit": combined_audit,
        "synthetic_manifest_audit": synthetic_audit,
        "synthetic_split_recommendation": split_recommendation,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}\n")
    for c, s in combined_audit.items():
        print(f"=== {c}: train={s['n_train']} val={s['n_val']} test={s['n_test']} sources_overlap={s['source_path_overlap_train_val']} ===")
    print(f"\nSynthetic split recommendation: {split_recommendation['recommended_val_sources']} sources -> ~{split_recommendation['estimated_val_entries']} val entries")


if __name__ == "__main__":
    main()
