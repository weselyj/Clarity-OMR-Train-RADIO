"""Stream 3: Per-corpus sequence-length and complexity distributions.

Computes for each corpus the token-sequence length percentiles, note
density per system, and multi-staff ratio. Compares against the v2
config's max_sequence_length=512 to identify silent truncation.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.corpus_distributions \\
        --manifest src\\data\\manifests\\token_manifest_stage3.jsonl \\
        --max-sequence-length 512 \\
        --out audit_results\\corpus_distributions.json
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    by_corpus = defaultdict(list)
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            toks = e.get("token_sequence", [])
            n_notes = sum(1 for t in toks if t.startswith("note-"))
            n_measures = sum(1 for t in toks if t == "<measure_start>")
            n_staffs = sum(1 for t in toks if t == "<staff_start>")
            by_corpus[e.get("dataset", "?")].append({
                "length": len(toks),
                "n_notes": n_notes,
                "n_measures": n_measures,
                "n_staffs": n_staffs,
                "split": e.get("split", "?"),
            })

    per_corpus = {}
    for corpus, rows in sorted(by_corpus.items()):
        lengths = [r["length"] for r in rows]
        notes = [r["n_notes"] for r in rows]
        staffs = [r["n_staffs"] for r in rows]
        truncated = sum(1 for r in rows if r["length"] > args.max_sequence_length)
        per_corpus[corpus] = {
            "n_entries": len(rows),
            "length_p50": statistics.median(lengths),
            "length_p90": statistics.quantiles(lengths, n=10)[-1] if len(lengths) >= 10 else None,
            "length_p95": statistics.quantiles(lengths, n=20)[-1] if len(lengths) >= 20 else None,
            "length_p99": statistics.quantiles(lengths, n=100)[-1] if len(lengths) >= 100 else None,
            "length_max": max(lengths),
            "mean_notes_per_entry": statistics.mean(notes),
            "mean_staffs_per_entry": statistics.mean(staffs),
            "multi_staff_fraction": sum(1 for s in staffs if s > 1) / len(staffs),
            "n_truncated_at_max_seq_len": truncated,
            "truncation_rate": truncated / len(rows),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "corpus_distributions",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "per_corpus": per_corpus,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}\n")
    for c, s in per_corpus.items():
        print(f"=== {c} (n={s['n_entries']}) ===")
        print(f"  length p50/p95/p99/max: {s['length_p50']:.0f} / {s['length_p95']} / {s['length_p99']} / {s['length_max']}")
        print(f"  mean notes/staffs:      {s['mean_notes_per_entry']:.1f} / {s['mean_staffs_per_entry']:.2f}")
        print(f"  multi-staff fraction:   {s['multi_staff_fraction']:.2%}")
        print(f"  truncated at {args.max_sequence_length}: {s['n_truncated_at_max_seq_len']} ({s['truncation_rate']:.2%})")


if __name__ == "__main__":
    main()
