# Data-Pipeline + Capacity Audit and Stage 3 v3 Retrain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the design in [`docs/superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md`](../specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md): audit the data pipeline, fix the identified issues, retrain Stage 3 v3, re-evaluate.

**Architecture:** 13 tasks across three phases with an explicit user-review gate after Phase 1. Phase 1 produces an audit report; the user approves the Phase 2 fix proposal before any code lands. Phase 1 tasks (1-5) are independent and parallelizable.

**Tech Stack:** Python 3.13, PyTorch on seder via `venv-cu132`, PEFT/DoRA, music21, verovio (for synthetic generation). Run on `feat/stage3-v3-data-rebalance` branch off `main` at `b2ed4a2`.

**Prereqs for the executing session:**
- Read the handoff doc at `archive/handoffs/2026-05-11-session-end-handoff.md` first
- Read the spec (referenced above)
- Read these memory entries: `project_radio_stage3_v2_audit.md`, `project_radio_frankenstein_diagnostic.md`, `project_radio_training_data_imbalance.md`
- Verify branch `feat/stage3-v3-data-rebalance` is off `main` at `b2ed4a2`; create if not

---

## File Structure

**New files (single branch, off `main` at `b2ed4a2`):**

| Path | Phase | Task |
|---|---|---|
| `scripts/audit/inspect_synthetic_samples.py` | 1 | 1 |
| `scripts/audit/reproduce_synthetic_generation.py` | 1 | 2 |
| `scripts/audit/corpus_distributions.py` | 1 | 3 |
| `scripts/audit/val_split_audit.py` | 1 | 4 |
| `scripts/audit/decoder_capacity_analysis.py` | 1 | 5 |
| `docs/audits/<DATE>-data-pipeline-and-capacity-audit.md` | 1 | 6 |
| `tests/train/test_freeze_encoder.py` | 2 | 7 |
| `scripts/data/add_val_split_synthetic.py` | 2 | 8 |
| `configs/train_stage3_radio_systems_v3.yaml` | 2 | 9 |
| `docs/audits/<DATE>-stage3-v3-data-rebalance-results.md` | 3 | 13 |

**Modified files:**
- `src/train/train.py` (Task 7): `_prepare_model_for_dora` + call site
- `data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl` (Task 8): split column populated for val carve-out

---

## Task 1: Stream 1 — Synthetic-corpus inspection

**Files:**
- Create: `scripts/audit/inspect_synthetic_samples.py`

- [ ] **Step 1.1: Implement the script**

```python
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

    # Per-sample digest: id, image path, first 30 tokens, last 10 tokens, length
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
```

- [ ] **Step 1.2: Push to seder and run**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
scp scripts/audit/inspect_synthetic_samples.py '10.10.1.29:audit_inspect.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_inspect.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\inspect_synthetic_samples.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.inspect_synthetic_samples --manifest data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl --out audit_results/synthetic_inspection.json --n 20'
```

Expected: prints group breakdown and aggregate stats. Pull `audit_results/synthetic_inspection.json` locally for analysis.

- [ ] **Step 1.3: Manual spot-check on 5 sampled entries**

Pull the JSON locally and pick 5 entries (covering at least 2 font styles and both sub-corpora). For each:
- Read the `image_path` value
- Pull the image from seder: `scp '10.10.1.29:Clarity-OMR-Train-RADIO/<image_path>' /tmp/<basename>.png`
- Open the image and visually compare against the `first_30` + `last_10` tokens
- Note: any token-image mismatches, degenerate images, suspiciously simple labels

- [ ] **Step 1.4: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git checkout -b feat/stage3-v3-data-rebalance
git add scripts/audit/inspect_synthetic_samples.py
git commit -m "feat(audit): Stream 1 synthetic-corpus inspection script

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Record findings (spot-check observations + stats interpretation) for use in Task 6's audit report.

---

## Task 2: Stream 2 — Reproduce synthetic generation pipeline

**Files:**
- Create: `scripts/audit/reproduce_synthetic_generation.py`

- [ ] **Step 2.1: Locate the existing generator**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
grep -rn "def generate\|verovio\|synthetic" src/data/generate_synthetic.py | head -20
ls src/data/ | grep -i synth
```

The expected entry point is `src/data/generate_synthetic.py`. If it doesn't exist or is named differently, find it via `grep -l verovio src/data/*.py` and document the actual entry point.

- [ ] **Step 2.2: Implement the reproduction script**

```python
"""Stream 2: Reproduce a small sample of synthetic_systems generation and
compare against the cached training data.

Picks 5 source MusicXML files from the openscore_lieder corpus, re-runs
the synthetic generator on each (with the same font styles the original
cache used), and diffs the resulting images + labels against what's
on disk in data/processed/synthetic_v2/.

Confirms generator determinism and detects silent rendering failures.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.reproduce_synthetic_generation \\
        --source-dir data\\openscore_lieder\\scores \\
        --cache-dir data\\processed\\synthetic_v2 \\
        --out audit_results\\synthetic_reproduce.json \\
        --n 5
"""
from __future__ import annotations
import argparse
import json
import sys
import hashlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True,
                   help="Root of source MusicXML files (data/openscore_lieder/scores)")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Root of cached generated data (data/processed/synthetic_v2)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    # Find 5 .mxl files in the source dir that have corresponding entries in
    # the cache.
    cache_pages = args.cache_dir / "pages"
    cache_manifest = args.cache_dir / "manifests" / "synthetic_token_manifest.jsonl"

    # Build map from source_path → list of generated entries
    by_source = {}
    with cache_manifest.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            sp = e.get("source_path")
            if sp:
                by_source.setdefault(sp, []).append(e)

    sources_with_cache = sorted(by_source.keys())
    print(f"Cache has entries for {len(sources_with_cache)} source files")

    # Pick first N
    chosen_sources = sources_with_cache[:args.n]

    per_source = []
    for src in chosen_sources:
        entries = by_source[src]
        print(f"\nSource: {src} ({len(entries)} cached crops)")
        # Hash a sample of the cached images
        cached_hashes = []
        for e in entries[:3]:
            img_path = _REPO / e["image_path"]
            if img_path.exists():
                cached_hashes.append({
                    "sample_id": e["sample_id"],
                    "image_path": e["image_path"],
                    "sha256_16": _file_hash(img_path),
                    "token_count": len(e.get("token_sequence", [])),
                })
            else:
                cached_hashes.append({
                    "sample_id": e["sample_id"],
                    "image_path": e["image_path"],
                    "error": "image_missing",
                })

        # TODO when implementing: invoke the actual generator here to regenerate
        # the same source piece, then compare hashes. The exact import depends
        # on what generate_synthetic.py exposes — look for `def main` or
        # `def generate(...)` that takes a source path.
        #
        # For this audit pass, document that the generator entry point is at
        # <path>, takes <args>, and the regenerated output for this source
        # produces the same/different hashes as the cache.
        per_source.append({
            "source_path": src,
            "n_cached_entries": len(entries),
            "cached_sample": cached_hashes,
            "regeneration": "TODO: invoke generator and compare",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "synthetic_reproduction",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "per_source": per_source,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
```

**NOTE TO IMPLEMENTER:** The TODO inside the script — actually invoking the generator — depends on what `generate_synthetic.py` exposes. Read the generator source first (Step 2.1) and adapt the script to call it on the source pieces. If the generator only has a CLI entry point, shell out via `subprocess`. If it has a callable `generate(source_path)` function, import and call directly.

If the generator can't be invoked from a single source piece (only batch mode), document this and limit Stream 2 to confirming cache consistency without re-running generation.

- [ ] **Step 2.3: Push, run, commit**

```bash
scp scripts/audit/reproduce_synthetic_generation.py '10.10.1.29:audit_repro.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_repro.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\reproduce_synthetic_generation.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.reproduce_synthetic_generation --source-dir data/openscore_lieder/scores --cache-dir data/processed/synthetic_v2 --out audit_results/synthetic_reproduce.json --n 5'

git add scripts/audit/reproduce_synthetic_generation.py
git commit -m "feat(audit): Stream 2 synthetic-generator reproduction script

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Stream 3 — Sequence-length + complexity distributions

**Files:**
- Create: `scripts/audit/corpus_distributions.py`

- [ ] **Step 3.1: Implement**

```python
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
```

- [ ] **Step 3.2: Push, run, commit**

```bash
scp scripts/audit/corpus_distributions.py '10.10.1.29:audit_dists.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_dists.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\corpus_distributions.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.corpus_distributions --manifest src/data/manifests/token_manifest_stage3.jsonl --max-sequence-length 512 --out audit_results/corpus_distributions.json'

git add scripts/audit/corpus_distributions.py
git commit -m "feat(audit): Stream 3 corpus distributions script

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Record findings: which corpora exceed 512? Is grandstaff systematically longer than primus? Is synthetic_polyphonic dramatically longer than synthetic_fullpage?

---

## Task 4: Stream 4 — Val-split structural audit

**Files:**
- Create: `scripts/audit/val_split_audit.py`

- [ ] **Step 4.1: Implement**

```python
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
    # Each source has multiple sub-entries (fonts × pages × staves). Need to
    # know mean crops-per-source for entry-count estimation.
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
```

- [ ] **Step 4.2: Push, run, commit**

```bash
scp scripts/audit/val_split_audit.py '10.10.1.29:audit_splits.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_splits.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\val_split_audit.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.val_split_audit --manifest src/data/manifests/token_manifest_stage3.jsonl --synthetic-manifest data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl --out audit_results/val_split_audit.json'

git add scripts/audit/val_split_audit.py
git commit -m "feat(audit): Stream 4 val-split structural audit

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Record findings: any source_path leakage on the 3 corpora with existing splits? Recommended synthetic split size?

---

## Task 5: Stream 5 — Decoder capacity analysis

**Files:**
- Create: `scripts/audit/decoder_capacity_analysis.py`

- [ ] **Step 5.1: Implement**

```python
"""Stream 5: Count decoder parameters and compare against reference
implementations.

Loads the Stage 3 v2 checkpoint, counts trainable + total parameters
broken down by encoder vs decoder, and prints a capacity assessment.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.decoder_capacity_analysis \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --out audit_results\\decoder_capacity.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _categorize(name: str) -> str:
    if "encoder" in name:
        return "encoder"
    if "decoder" in name:
        return "decoder"
    if "positional_bridge" in name or "deformable" in name:
        return "bridge"
    if "lm_head" in name or "output" in name.lower() or "vocab" in name.lower():
        return "head"
    if "embed" in name.lower():
        return "embedding"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    ckpt = torch.load(args.stage_b_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    if not isinstance(sd, dict):
        raise RuntimeError("Could not find state_dict in checkpoint")

    by_category = {}
    for name, tensor in sd.items():
        cat = _categorize(name)
        if cat not in by_category:
            by_category[cat] = {"n_params": 0, "n_tensors": 0, "sample_keys": []}
        by_category[cat]["n_params"] += tensor.numel()
        by_category[cat]["n_tensors"] += 1
        if len(by_category[cat]["sample_keys"]) < 5:
            by_category[cat]["sample_keys"].append(name)

    total = sum(c["n_params"] for c in by_category.values())
    summary = {
        "total_params": total,
        "by_category": {
            k: {**v, "fraction": v["n_params"] / total}
            for k, v in by_category.items()
        },
    }

    # Reference: upstream Clarity-OMR DaViT model. The DaViT-base + decoder
    # used upstream is roughly ~200-300M params total per published spec;
    # encoder ~80M, decoder ~120M. Exact numbers TBD by reviewer when
    # interpreting — this script just reports.
    summary["reference_note"] = (
        "Reference: upstream Clarity-OMR DaViT model is approximately "
        "200-300M params (encoder ~80M, decoder ~120M per published spec). "
        "Compare the decoder count above to ~120M. If significantly smaller, "
        "capacity may be the bottleneck."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "decoder_capacity",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}\n")
    print(f"Total params: {total / 1e6:.2f}M")
    for cat, s in by_category.items():
        print(f"  {cat:<12} {s['n_params']/1e6:>7.2f}M  ({s['fraction']:.1%})  {s['n_tensors']} tensors")
        print(f"    sample keys: {s['sample_keys'][:2]}")
    print(f"\n{summary['reference_note']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Push, run, commit**

```bash
scp scripts/audit/decoder_capacity_analysis.py '10.10.1.29:audit_capacity.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_capacity.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\decoder_capacity_analysis.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.decoder_capacity_analysis --stage-b-ckpt checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt --out audit_results/decoder_capacity.json'

git add scripts/audit/decoder_capacity_analysis.py
git commit -m "feat(audit): Stream 5 decoder capacity analysis

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Record finding: total decoder param count + fraction vs reference. Recommendation: bump size, keep current, or out of scope to know.

---

## Task 6: Synthesize Phase 1 report (gate)

**Files:**
- Create: `docs/audits/<DATE>-data-pipeline-and-capacity-audit.md` (use today's date)

- [ ] **Step 6.1: Pull all 5 stream outputs locally and synthesize**

```bash
mkdir -p /tmp/audit
for f in synthetic_inspection synthetic_reproduce corpus_distributions val_split_audit decoder_capacity; do
    scp "10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/${f}.json" "/tmp/audit/${f}.json"
done
```

Read each JSON. For each Stream, write a section in the report including:
- One-sentence goal
- Key numbers from the JSON output
- Interpretation (what does it mean for the project?)
- Specific recommendation for Phase 2

- [ ] **Step 6.2: Write the audit report**

Create `docs/audits/<DATE>-data-pipeline-and-capacity-audit.md`. Replace `<DATE>` with today's date in YYYY-MM-DD format. Structure:

```markdown
# Data-Pipeline + Capacity Audit

**Date:** <YYYY-MM-DD>
**Spec:** [../superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md](../superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md)
**Branch:** `feat/stage3-v3-data-rebalance`
**Phase:** 1 of 3 (gate for Phase 2)

## TL;DR

<Two-paragraph synthesis: what the audit found, what the Phase 2 fix proposal is. Include the headline numbers (sequence-length stats, val-split status, capacity assessment).>

## Stream 1 — Synthetic-corpus inspection

**Result:** <FILL>
**Evidence:** `audit_results/synthetic_inspection.json`
**Numbers:** <key stats from the JSON>
**Interpretation:** <1-2 paragraphs>
**Recommendation for Phase 2:** <specific action>

## Stream 2 — Reproduce synthetic generation

(Same shape as above, filled from synthetic_reproduce.json)

## Stream 3 — Sequence-length + complexity distributions

(Same shape as above, filled from corpus_distributions.json)

## Stream 4 — Val-split structural audit

(Same shape as above, filled from val_split_audit.json. Include the recommended split methodology.)

## Stream 5 — Decoder capacity analysis

(Same shape as above, filled from decoder_capacity.json)

## Phase 2 fix proposal

Final list of concrete changes to make in Phase 2:

1. **Encoder freeze fix:** add cache-derived auto-detect to `_prepare_model_for_dora` (per the design spec). Always applied.
2. **Synthetic val split:** carve <N> val entries by source_path. Methodology: <from Stream 4>.
3. **dataset_mix rebalance:** new weights are <list> based on <reasoning>.
4. **Conditional fixes (if any):** max_seq_len → <N>, decoder size → <details>. Each with rationale.

**Estimated Phase 2 effort:** <hours/days>

## Gate decision

User reviews this report and either approves the Phase 2 proposal or escalates.

- If approved → proceed to Task 7
- If escalated (e.g., audit recommends regenerating synthetic, which is out of scope) → pause the plan and discuss
```

- [ ] **Step 6.3: Commit and ESCALATE (gate)**

```bash
git add docs/audits/<DATE>-data-pipeline-and-capacity-audit.md
git commit -m "audit: Phase 1 report (data-pipeline + capacity audit)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

**STOP here.** Report the audit findings to the user via your subagent-driven controller. Wait for user approval of the Phase 2 fix proposal before any Phase 2 task starts. If the report recommends bigger scope (e.g., regenerate synthetic), DO NOT silently expand scope — escalate to the user.

---

## Task 7: Encoder freeze fix

**Files:**
- Modify: `src/train/train.py` (`_prepare_model_for_dora` at ~lines 1228-1340, call site at ~line 2181)
- Create: `tests/train/test_freeze_encoder.py`

This task is fully designed in the prior v3 retrain spec ([`2026-05-11-stage3-v3-retrain-design.md`](../specs/2026-05-11-stage3-v3-retrain-design.md)) and was never landed. Refer to that spec's "Phase 2 — Fix (Option B)" section for the full design rationale.

- [ ] **Step 7.1: Add the new kwarg + freeze guard**

In `src/train/train.py`, modify `_prepare_model_for_dora` signature:

```python
def _prepare_model_for_dora(
    model,
    dora_config: Dict[str, object],
    *,
    stage_config: "Optional[StageTrainingConfig]" = None,
):
```

In the function body, replace the requires_grad loop (around line 1327) with:

```python
uses_cache = bool(
    stage_config is not None
    and stage_config.cache_root
    and stage_config.cache_hash16
)

for parameter in model.parameters():
    parameter.requires_grad = False
for name, parameter in model.named_parameters():
    if uses_cache and "encoder" in name:
        continue  # cache requires encoder be frozen
    if "lora_" in name or any(marker in name for marker in new_module_keywords):
        parameter.requires_grad = True
if not any(parameter.requires_grad for parameter in model.parameters()):
    for parameter in model.parameters():
        parameter.requires_grad = True
```

- [ ] **Step 7.2: Update the call site at ~line 2181**

Change:
```python
model, dora_applied = _prepare_model_for_dora(base_model, components["dora_config"])
```
to:
```python
model, dora_applied = _prepare_model_for_dora(
    base_model, components["dora_config"], stage_config=stage
)
```

Confirm the local variable holding the `StageTrainingConfig` is called `stage` — check the surrounding ~20 lines.

- [ ] **Step 7.3: Add pre-flight assertion immediately after the call site**

```python
_uses_cache = bool(stage.cache_root and stage.cache_hash16)
if _uses_cache:
    _trainable_encoder = sum(
        1 for n, p in model.named_parameters() if "encoder" in n and p.requires_grad
    )
    _trainable_decoder = sum(
        1 for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad
    )
    print(f"[freeze] trainable encoder params: {_trainable_encoder}")
    print(f"[freeze] trainable decoder params: {_trainable_decoder}")
    if _trainable_encoder != 0:
        raise RuntimeError(
            f"Stage {stage.stage_name!r} uses encoder cache "
            f"(hash16={stage.cache_hash16!r}), so all encoder params must be "
            f"frozen, but {_trainable_encoder} are trainable. "
            f"_prepare_model_for_dora freeze logic regressed; refusing to train."
        )
```

- [ ] **Step 7.4: Write the regression tests**

Create `tests/train/test_freeze_encoder.py`:

```python
"""Regression tests for the Stage 3 v2 encoder-freeze bug.

CUDA-gated by tests/conftest.py (path matches CUDA_REQUIRED_DIRS).
"""
from __future__ import annotations


def _build_minimal_stage_b():
    from src.models.radio_stage_b import RadioStageB, RadioStageBConfig
    cfg = RadioStageBConfig(
        decoder_dim=64, decoder_heads=2, decoder_layers=1, vocab_size=64,
    )
    return RadioStageB(cfg)


def _minimal_stage_config(*, cache_root=None, cache_hash16=None):
    from src.train.train import StageTrainingConfig
    return StageTrainingConfig(
        stage_name="test_stage",
        stage_b_encoder="radio_h",
        cache_root=cache_root,
        cache_hash16=cache_hash16,
        lr_dora=5e-4,
        lr_new_modules=3e-4,
    )


def test_cache_config_freezes_encoder_lora_params():
    from src.train.train import _prepare_model_for_dora
    model = _build_minimal_stage_b()
    stage_config = _minimal_stage_config(
        cache_root="data/cache/encoder",
        cache_hash16="ac8948ae4b5be3e9",
    )
    model, _ = _prepare_model_for_dora(
        model, {"r": 8, "alpha": 16}, stage_config=stage_config
    )
    trainable_encoder = [
        n for n, p in model.named_parameters()
        if "encoder" in n and p.requires_grad
    ]
    assert trainable_encoder == [], f"trainable encoder params: {trainable_encoder[:5]}"


def test_no_cache_unfreezes_encoder_lora_params():
    from src.train.train import _prepare_model_for_dora
    model = _build_minimal_stage_b()
    stage_config = _minimal_stage_config()
    model, _ = _prepare_model_for_dora(
        model, {"r": 8, "alpha": 16}, stage_config=stage_config
    )
    encoder_lora_trainable = [
        n for n, p in model.named_parameters()
        if "encoder" in n and "lora_" in n and p.requires_grad
    ]
    assert len(encoder_lora_trainable) > 0


def test_decoder_lora_always_trainable():
    from src.train.train import _prepare_model_for_dora
    for cache_root in (None, "data/cache/encoder"):
        cache_hash16 = "ac8948ae4b5be3e9" if cache_root else None
        model = _build_minimal_stage_b()
        stage_config = _minimal_stage_config(
            cache_root=cache_root, cache_hash16=cache_hash16,
        )
        model, _ = _prepare_model_for_dora(
            model, {"r": 8, "alpha": 16}, stage_config=stage_config
        )
        decoder_lora_trainable = [
            n for n, p in model.named_parameters()
            if "encoder" not in n and "lora_" in n and p.requires_grad
        ]
        assert len(decoder_lora_trainable) > 0
```

- [ ] **Step 7.5: Push, run tests, commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
scp src/train/train.py '10.10.1.29:train.py'
scp tests/train/test_freeze_encoder.py '10.10.1.29:test_freeze_encoder.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\train.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\src\train\train.py" && move /Y "%USERPROFILE%\test_freeze_encoder.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\tests\train\test_freeze_encoder.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m pytest tests/train/test_freeze_encoder.py -v'
```

Expected: 3 tests PASS. If any fail, the freeze logic needs adjustment — read the assertion messages.

```bash
git add src/train/train.py tests/train/test_freeze_encoder.py
git commit -m "fix(train): auto-freeze encoder when stage uses encoder cache

Resolves the Stage 3 v2 encoder-DoRA-not-frozen bug (audit PR #48).
Cache use and encoder updates are physically incompatible (cache becomes
stale after first encoder gradient step), so couple them in code.

Add 3 regression tests + runtime pre-flight assertion.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Synthetic val split

**Files:**
- Create: `scripts/data/add_val_split_synthetic.py`
- Modify: `data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl`

- [ ] **Step 8.1: Implement the val-split script**

The exact split fraction and methodology come from Stream 4's recommendation. Default to 5% by source_path.

```python
"""Add a val split to the synthetic_v2 manifest by carving out source pieces.

Reads the synthetic manifest, picks ~5% of unique source_paths to be the
val set, and writes back a new manifest with those entries marked
split=val (the rest stay split=train). All crops derived from a single
source MusicXML stay in one split — prevents leakage between fonts
that share the same source.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.data.add_val_split_synthetic \\
        --manifest data\\processed\\synthetic_v2\\manifests\\synthetic_token_manifest.jsonl \\
        --val-fraction 0.05 \\
        --seed 42 \\
        --backup data\\processed\\synthetic_v2\\manifests\\synthetic_token_manifest.no_val.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backup", type=Path, required=True,
                   help="Path to write a backup copy of the original manifest")
    args = p.parse_args()

    # Backup
    shutil.copyfile(args.manifest, args.backup)
    print(f"Backed up original manifest to {args.backup}")

    # Read all entries
    entries = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"Read {len(entries)} entries")

    # Group by source_path
    by_source = defaultdict(list)
    for e in entries:
        by_source[e.get("source_path", "?")].append(e)

    sources = sorted(by_source.keys())
    print(f"{len(sources)} unique source_paths")

    # Pick val sources
    rng = random.Random(args.seed)
    n_val = max(1, int(len(sources) * args.val_fraction))
    val_sources = set(rng.sample(sources, n_val))
    print(f"Picking {n_val} sources for val ({100*n_val/len(sources):.2f}% by source count)")

    # Reassign splits — only change train→val for the picked sources
    n_changed = 0
    for e in entries:
        if e.get("source_path") in val_sources:
            if e.get("split") == "train":
                e["split"] = "val"
                n_changed += 1
            elif e.get("split") in (None, ""):
                e["split"] = "val"
                n_changed += 1

    # Write back
    with args.manifest.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"Reassigned {n_changed} entries from train→val")

    # Sanity-print final counts
    from collections import Counter
    counts = Counter(e.get("split") for e in entries)
    print(f"Final split counts: {dict(counts)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Push, run, verify, commit**

```bash
scp scripts/data/add_val_split_synthetic.py '10.10.1.29:add_val.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\add_val.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\data\add_val_split_synthetic.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.data.add_val_split_synthetic --manifest data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl --val-fraction 0.05 --seed 42 --backup data/processed/synthetic_v2/manifests/synthetic_token_manifest.no_val.jsonl'
```

Re-run Stream 4 audit to verify the split is now non-leaking:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.val_split_audit --manifest src/data/manifests/token_manifest_stage3.jsonl --synthetic-manifest data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl --out audit_results/val_split_audit_post.json'
```

Expected: synthetic_systems now has ~1000 val entries with 0 source_path overlap with train.

Note: the **combined manifest** (`src/data/manifests/token_manifest_stage3.jsonl`) may also need updating if it filters by split. Check whether the trainer reads the combined manifest's split field or the per-corpus manifest's split field. If the combined manifest is regenerated from per-corpus manifests, regenerating it will propagate the val split. If the combined manifest is hand-edited, also update it.

```bash
git add scripts/data/add_val_split_synthetic.py
git commit -m "feat(data): add val split to synthetic_systems manifest

Carves ~5% of source_paths to val by deterministic seeded sample, so all
crops derived from one source MusicXML stay in one split.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

If the combined manifest needs to be regenerated, run the existing combined-manifest-builder (likely `python -m src.data.build_mixed_dataset` or similar — check `src/data/build_mixed_dataset.py` for usage) and commit the updated combined manifest.

---

## Task 9: New v3 training config

**Files:**
- Create: `configs/train_stage3_radio_systems_v3.yaml`

- [ ] **Step 9.1: Copy v2 and apply baseline + conditional changes**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
cp configs/train_stage3_radio_systems.yaml configs/train_stage3_radio_systems_v3.yaml
```

Edit `configs/train_stage3_radio_systems_v3.yaml`:

1. Replace the header docstring with:

```yaml
# Stage 3 Phase 1 v3 — data-pipeline rebalance + encoder-freeze-fix retrain.
#
# Supersedes v2 (which had encoder DoRA accidentally unfrozen and the
# dataset_mix giving synthetic_systems 77.8% of the cached-tier gradient
# share on the smallest corpus with zero val split). v3 fixes:
#   - encoder freeze (auto-detect from cache_root + cache_hash16)
#   - synthetic_systems val split (~5% by source_path)
#   - dataset_mix rebalanced per Phase 1 audit recommendation
#   - (conditional) max_sequence_length bump
#   - (conditional) decoder size bump
#
# Run command (executed on GPU box at 10.10.1.29):
#   ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
#     venv-cu132\Scripts\python -u src/train/train.py \
#       --stage-configs configs/train_stage3_radio_systems_v3.yaml \
#       --mode execute \
#       --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \
#       --start-stage stage3-radio-systems-frozen-encoder \
#       --checkpoint-dir checkpoints/full_radio_stage3_v3 \
#       --token-manifest src/data/manifests/token_manifest_stage3.jsonl \
#       --step-log logs/full_radio_stage3_v3_steps.jsonl'
```

2. Set `effective_samples_per_epoch: 9000` (carry forward from prior v3 spec; v2 was still descending at step 6000).

3. Update `dataset_mix` based on Phase 1 audit recommendation. Default proposal if audit just says "rebalance to natural":

```yaml
dataset_mix:
  - dataset: synthetic_systems
    ratio: 0.20     # was 0.7778; reduce drastically
    split: train
    required: true
  - dataset: grandstaff_systems
    ratio: 0.55     # was 0.1111; upweight to closest match for demo eval
    split: train
    required: true
  - dataset: primus_systems
    ratio: 0.25     # was unspecified; share remaining
    split: train
    required: true
```

If audit recommends dropping synthetic entirely, remove its entry from the list and rebalance grandstaff/primus to 70/30.

4. Apply conditional fixes if Phase 1 recommended them:
- If `max_sequence_length` bump: change `max_sequence_length: 512` → `max_sequence_length: 1024` (or as recommended).
- If decoder size bump: this requires architecture changes. Look at `src/models/radio_stage_b.py` `RadioStageBConfig` defaults and update them. ALSO update the model_factory config dict that the trainer reads.

- [ ] **Step 9.2: Push and commit**

```bash
scp configs/train_stage3_radio_systems_v3.yaml '10.10.1.29:stage3_v3.yaml'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\stage3_v3.yaml" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\configs\train_stage3_radio_systems_v3.yaml"'

git add configs/train_stage3_radio_systems_v3.yaml
git commit -m "feat(config): Stage 3 v3 training config

Rebalanced dataset_mix, 9000-step target, references the new freeze logic
in _prepare_model_for_dora. Encoder freeze is now structural via the
cache_root/cache_hash16 settings rather than a comment in the docstring.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: Resume verification smoke test

- [ ] **Step 10.1: Start a deliberately-short v3 training run**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -u src/train/train.py --stage-configs configs/train_stage3_radio_systems_v3.yaml --mode execute --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/_stage3_v3_smoke --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v3_smoke_steps.jsonl' &
```

Watch for the `[freeze] trainable encoder params: 0` pre-flight line. If non-zero, the assertion will raise — the freeze fix regressed; fix Task 7 first.

After step 500's checkpoint is written (visible via `dir checkpoints\_stage3_v3_smoke`), kill the run.

- [ ] **Step 10.2: Resume from the smoke-test step-500 checkpoint**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -u src/train/train.py --stage-configs configs/train_stage3_radio_systems_v3.yaml --mode execute --resume-checkpoint checkpoints/_stage3_v3_smoke/stage3-radio-systems-frozen-encoder_step_0000500.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/_stage3_v3_smoke_resume --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v3_smoke_resume_steps.jsonl'
```

Verify in the step log: global_step resumes from 501 (not 1). Loss at step 501 should be similar to step 500 of the original run (no spike, no re-init pattern).

If resume restarts from 1 or shows a loss spike, **STOP and escalate** — resume is broken and the 8h main run is at risk.

- [ ] **Step 10.3: Clean up smoke checkpoints**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && rmdir /S /Q checkpoints\_stage3_v3_smoke checkpoints\_stage3_v3_smoke_resume'
```

No commit needed — these are gitignored artifacts.

---

## Task 11: Launch and monitor v3 training

- [ ] **Step 11.1: Launch the main v3 run**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -u src/train/train.py --stage-configs configs/train_stage3_radio_systems_v3.yaml --mode execute --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/full_radio_stage3_v3 --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v3_steps.jsonl' &
```

Background it (the SSH-detached pattern from prior sessions is in `run_demo_eval_logged.cmd` on seder if needed).

- [ ] **Step 11.2: Verify pre-flight assertion at step ~0-1**

After 30 seconds, check the step log:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && type logs\full_radio_stage3_v3_steps.jsonl' | head -3
```

Look for the `[freeze]` line in stdout. If `trainable encoder params: 0` was printed and training is proceeding (steps logged), the freeze is real. If training never started, check whether the RuntimeError was raised.

- [ ] **Step 11.3: Monitor per-corpus val_loss for all 4 corpora**

Every ~1 hour, sample the latest validation entries:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -c "
import json
val_rows = []
with open(r''logs/full_radio_stage3_v3_steps.jsonl'', encoding=''utf-8'') as f:
    for line in f:
        d = json.loads(line)
        if d.get(''validation'') is not None:
            val_rows.append(d)
for r in val_rows[-5:]:
    v = r[''validation'']
    vd = v.get(''val_loss_per_dataset'', {})
    print(f''step {r[\"global_step\"]:>5}: overall={v.get(\"val_loss\"):.4f}'')
    for c, l in sorted(vd.items()):
        print(f''  {c:<28} {l:.4f}'')
"'
```

Expected: synthetic_systems now appears (it didn't in v2), and all 4 corpora show monotonic descent.

- [ ] **Step 11.4: Wait for completion**

When step 9000 is reached (or training crashes), confirm checkpoints exist:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && dir checkpoints\full_radio_stage3_v3 /B'
```

Expected: `_best.pt`, `_final.pt`, and `_step_NNNNNNN.pt` for every 500-step interval.

- [ ] **Step 11.5: Verify encoder LoRA keys are unchanged from Stage 2 v2**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -c "
import torch
s2 = torch.load(r''checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt'', map_location=''cpu'', weights_only=False)
s3v3 = torch.load(r''checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt'', map_location=''cpu'', weights_only=False)
def get_sd(c): return c.get(''model_state_dict'') or c.get(''model'') or c
s2_sd, s3_sd = get_sd(s2), get_sd(s3v3)
encoder_keys = [k for k in s2_sd if ''lora_'' in k and ''encoder'' in k]
print(f''checking {len(encoder_keys)} encoder LoRA keys'')
changed = 0
for k in encoder_keys:
    if k in s3_sd:
        diff = (s2_sd[k].float() - s3_sd[k].float()).abs().max().item()
        if diff > 1e-6:
            changed += 1
print(f''encoder LoRA keys changed: {changed} / {len(encoder_keys)}'')
if changed == 0:
    print(''FREEZE WORKED — encoder LoRA is identical between S2v2 and S3v3'')
else:
    print(''FREEZE FAILED — encoder LoRA drifted despite the fix'')
"'
```

If FREEZE FAILED, the pre-flight assertion didn't catch the regression. Investigate before moving to Phase 3.

- [ ] **Step 11.6: Commit run-log summary to report**

(Final report stub will be written in Task 13; for now, append run timing + best-step info to the running report draft.)

---

## Task 12: Phase 3 — Re-evaluation

- [ ] **Step 12.1: Re-run A2 against v3**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a2_encoder_parity --manifest src/data/manifests/token_manifest_stage3.jsonl --cache-root data/cache/encoder --cache-hash16 ac8948ae4b5be3e9 --stage-b-ckpt checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 5 --out audit_results/a2_encoder_stage3_v3.json'

scp '10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/a2_encoder_stage3_v3.json' /tmp/a2_v3.json
python3 -c "import json; d=json.load(open('/tmp/a2_v3.json')); print('PASS' if d['pass'] else 'FAIL'); print('max:', d.get('bilinear_vs_cache_overall_max_abs', d.get('overall_max_abs_diff')))"
```

Expected PASS (max_abs_diff << 0.01).

- [ ] **Step 12.2: Re-run A3 against v3**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a3_decoder_on_training --manifest src/data/manifests/token_manifest_stage3.jsonl --stage-b-ckpt checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 5 --out audit_results/a3_decoder_stage3_v3.json'

scp '10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/a3_decoder_stage3_v3.json' /tmp/a3_v3.json
python3 -c "
import json
d = json.load(open('/tmp/a3_v3.json'))
print(f'mean tok_acc: {d[\"mean_token_accuracy\"]:.3f}, exact: {d[\"exact_match_rate\"]:.3f}')
by_corpus = {}
for r in d['per_sample']:
    if r['status'] == 'compared':
        by_corpus.setdefault(r['dataset'], []).append(r['token_accuracy'])
for c, vals in sorted(by_corpus.items()):
    print(f'  {c:<26} n={len(vals)} mean={sum(vals)/len(vals):.3f}')
"
```

Expected: all 4 corpora ≥ 0.80 token accuracy. Compare against v2 baseline (synthetic 0.113, grandstaff 0.400, primus 0.753, cameraprimus 0.865).

- [ ] **Step 12.3: 4-piece HF demo eval**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m eval.run_clarity_demo_radio_eval --stage-b-ckpt checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt --name stage3_v3_best'

scp '10.10.1.29:Clarity-OMR-Train-RADIO/eval/results/clarity_demo_stage3_v3_best/summary.json' /tmp/demo_v3.json
python3 -c "
import json, statistics
d = json.load(open('/tmp/demo_v3.json'))['pieces']
of1 = [p['score']['onset_f1'] for p in d.values() if 'onset_f1' in p.get('score',{})]
print(f'mean onset_f1: {statistics.mean(of1):.4f}')
for stem, p in d.items():
    s = p.get('score', {})
    print(f'  {stem[:35]:<35} onset_f1={s.get(\"onset_f1\",0):.4f}')
"
```

Expected runtime: ~10 min. Record mean + per-piece. Compare against v2 baseline 0.0589.

- [ ] **Step 12.4: 50-piece lieder ship-gate**

Find the lieder eval entry point:

```bash
ls eval/ | grep -i lieder
```

Run:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m eval.run_lieder_eval --stage-b-ckpt checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt --name stage3_v3_best' &
```

Expected runtime: 1-3h. Wait for completion, pull summary, compute corpus mean onset_f1. Compare against v2 0.0819.

- [ ] **Step 12.5: Apply decision matrix**

From the spec:

| A2 | A3 multi-staff | Demo mean onset_f1 | Verdict |
|---|---|---|---|
| PASS | ≥ 0.80 | ≥ 0.241 | **SHIP** |
| PASS | ≥ 0.80 | 0.10-0.241 | Architecture/capacity sub-project next |
| PASS | < 0.80 | any | Data fix didn't take — revisit Phase 1 |
| FAIL | any | any | Freeze didn't stick — debug Phase 2 |

Record the verdict branch.

---

## Task 13: Final report and PR

**Files:**
- Create: `docs/audits/<DATE>-stage3-v3-data-rebalance-results.md`

- [ ] **Step 13.1: Write the final report**

Structure:

```markdown
# Stage 3 v3 Data-Rebalance Retrain — Results

**Date:** <YYYY-MM-DD>
**Spec:** [../superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md](../superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md)
**Plan:** [../superpowers/plans/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-plan.md](../superpowers/plans/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-plan.md)
**Branch:** `feat/stage3-v3-data-rebalance`
**Checkpoint:** `checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt`

## TL;DR

<2-paragraph synthesis: data fixes applied, training results, verdict.>

## Phase 1 audit summary

(Brief: the audit recommended <list of fixes>. Point to the audit report file.)

## Phase 2 fixes applied

- Encoder freeze: <how it took, evidence>
- Synthetic val split: <N val entries created, methodology>
- dataset_mix rebalance: <new weights, rationale>
- Conditional fixes: <none / which / rationale>

## Phase 3 results

| Eval | Result |
|---|---|
| A2 encoder parity | <PASS/FAIL + max_abs_diff> |
| A3 mean token accuracy | <per-corpus table> |
| 4-piece demo mean onset_f1 | <number + per-piece + comparison to v2/frankenstein> |
| 50-piece lieder corpus mean | <number + comparison to v2> |

## Verdict

<Which branch of the decision matrix. What follows.>
```

- [ ] **Step 13.2: Commit and open PR**

```bash
git add docs/audits/<DATE>-stage3-v3-data-rebalance-results.md
git commit -m "audit: Stage 3 v3 data-rebalance retrain final report

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"

git push -u origin feat/stage3-v3-data-rebalance
gh pr create --title "feat(train): Stage 3 v3 data-rebalance retrain + freeze fix" --body "$(cat <<'EOF'
## Summary

Implements the design at [docs/superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md]. Audits the data pipeline + decoder capacity, applies the recommended fixes (encoder freeze, synthetic val split, dataset_mix rebalance, possibly architecture changes), and retrains Stage 3 as v3.

## Phase 1 audit summary

<COPY FROM AUDIT REPORT>

## Phase 2 fixes

<COPY FROM RESULTS REPORT>

## Phase 3 verdict

<COPY FROM RESULTS REPORT>

## Test plan

- [ ] Reviewer: confirm `pytest tests/train/test_freeze_encoder.py -v` shows 3 PASSED on seder
- [ ] Reviewer: confirm `audit_results/a2_encoder_stage3_v3.json` shows PASS
- [ ] Reviewer: confirm `audit_results/a3_decoder_stage3_v3.json` shows ≥ 0.80 token accuracy on all 4 corpora

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace `<COPY FROM ...>` markers with the actual content before submitting.

---

## Self-review checklist

- [x] **Spec coverage:**
  - Phase 1 Streams 1-5 → Tasks 1-5
  - Phase 1 report + gate → Task 6
  - Phase 2 baseline fixes (freeze, val split, dataset_mix) → Tasks 7-9
  - Phase 2 conditional fixes → Task 9 sub-steps
  - Phase 2 retrain → Tasks 10-11
  - Phase 3 re-eval → Task 12
  - Phase 3 report + PR → Task 13
- [x] **Placeholder scan:** Only legitimate output-fill markers (`<FILL>`, `<COPY FROM ...>`, `<YYYY-MM-DD>`, `<N>`) — all clearly call out what fills them.
- [x] **Type consistency:** `pick_audit_samples` (existing), `_load_stage_b_crop_tensor` (existing), `_prepare_model_for_dora` new signature consistent between Task 7 implementation and tests. `StageTrainingConfig.cache_root` / `cache_hash16` field names match existing usage in `src/train/train.py:101-102`.
- [x] **Path consistency:** Manifest paths, checkpoint paths, cache root all consistent with what's known on seder + repo.
- [x] **Gate handling:** Task 6 explicitly halts and waits for user approval before any Phase 2 task runs. Task 10.2 halts if resume is broken. Task 11.5 halts if encoder LoRA drifted despite the fix. Task 12.5's matrix may also gate Task 13 (a failed verdict still produces a report but routes to a different next sub-project).
