# Stage 3 v4 Scan-Realistic Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain Stage 3 with a new scan-realistic corpus derived from `grandstaff_systems` so the decoder stops misreading visually-bass staves as treble on real-world piano scans.

**Architecture:** Cheap-cached retrain (option A from the design). Offline scan-style degradation applied to `grandstaff_systems` → new `scanned_grandstaff_systems` corpus → encoder cache rebuilt → v3-style cached retrain with rebalanced mix (clean grandstaff 0.30 / scanned grandstaff 0.30 + synth 0.15 + primus 0.125 + camera 0.125). Two cheap pre-flight gates: bottom-quartile lieder cluster analysis and 20-sample overfit smoke test.

**Tech Stack:** PyTorch, C-RADIOv4-H encoder (frozen via cache), custom transformer decoder. Python data pipeline (PIL + numpy + cv2 for offline degradation). YAML configs. GPU box `seder` (10.10.1.29, RTX 5090) for training/encoding/eval; CPU work runs locally.

**Spec:** [docs/superpowers/specs/2026-05-13-stage3-v4-scan-realistic-retrain-design.md](../specs/2026-05-13-stage3-v4-scan-realistic-retrain-design.md)

---

## File structure

**New files:**
- `src/data/scan_degradation.py` — degradation pipeline (pure Python, deterministic per seed)
- `tests/data/test_scan_degradation.py` — unit tests (CPU-only)
- `scripts/data/build_scanned_grandstaff_systems.py` — corpus build script
- `scripts/data/build_overfit_smoke_manifest.py` — generates 20-sample manifest deterministically
- `scripts/audit/measure_real_scan_distribution.py` — Phase 1a calibration script
- `scripts/audit/cluster_bottom_quartile_lieder.py` — Phase 0a cluster analysis
- `data/manifests/overfit_smoke_v4.jsonl` — generated, 20-sample static manifest (committed)
- `configs/train_stage3_v4_overfit_smoke.yaml` — overfit smoke config
- `configs/train_stage3_radio_systems_v4.yaml` — main retrain config
- `docs/audits/2026-05-13-real-scan-degradation-calibration.md` — Phase 1a output
- `docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md` — Phase 0a output
- `docs/audits/2026-05-13-stage3-v4-results.md` — Phase 3 results

**Modified files:**
- `scripts/build_stage3_combined_manifest.py` — add `--scanned_grandstaff_systems_manifest` arg + registry entry
- `src/data/manifests/token_manifest_stage3_audit.json` — regenerated post-Phase 1
- `src/data/manifests/token_manifest_stage3.jsonl` — regenerated post-Phase 1

---

## Pre-flight

### Task 0: Branch setup

**Files:** none yet — just branch creation.

- [ ] **Step 0.1: Create branch off main**

Run:
```bash
git fetch origin
git checkout -b feat/stage3-v4-scan-realistic origin/main
```
Expected: clean checkout, no conflicts.

- [ ] **Step 0.2: Verify worktree is clean**

Run: `git status`
Expected: nothing to commit.

---

## Phase 0a — Bottom-quartile lieder failure-mode analysis (CPU, half-day)

### Task 1: Cluster analysis script

**Files:**
- Create: `scripts/audit/cluster_bottom_quartile_lieder.py`
- Create: `tests/scripts/audit/test_cluster_bottom_quartile.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/scripts/audit/test_cluster_bottom_quartile.py`:

```python
"""Test bottom-quartile lieder cluster analysis script.

CPU-only. Asserts cluster-tagging logic on synthetic decoder-output fixtures.
"""
from pathlib import Path
import json
import pytest

from scripts.audit.cluster_bottom_quartile_lieder import classify_piece


def test_bass_clef_misread_detected():
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2", "median_octave_pred": 4},
                    {"clef_pred": "clef-G2", "median_octave_pred": 4},  # bottom predicted G2
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
    }
    tags = classify_piece(piece_tokens)
    assert "bass-clef-misread" in tags


def test_phantom_staff_residual_detected():
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2"}, {"clef_pred": "clef-G2"}, {"clef_pred": "clef-F4"}
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
    }
    tags = classify_piece(piece_tokens)
    assert "phantom-staff-residual" in tags


def test_clean_piece_has_other_tag_only():
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2", "median_octave_pred": 5},
                    {"clef_pred": "clef-F4", "median_octave_pred": 2},
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
    }
    tags = classify_piece(piece_tokens)
    assert tags == ["other"]


def test_key_time_signature_residual_detected_for_non_4_4():
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2", "median_octave_pred": 5},
                    {"clef_pred": "clef-F4", "median_octave_pred": 2},
                ],
                "time_sig_pred": "time-4/4",
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
        "ground_truth_time_sig": "time-3/4",
    }
    tags = classify_piece(piece_tokens)
    assert "key-time-sig-residual" in tags
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest tests/scripts/audit/test_cluster_bottom_quartile.py -v`
Expected: FAIL with "ModuleNotFoundError" or "function not defined".

- [ ] **Step 1.3: Implement the script**

Create `scripts/audit/cluster_bottom_quartile_lieder.py`:

```python
"""Cluster bottom-quartile lieder pieces by observable failure mode.

Reads:
  - eval/results/lieder_stage3_v3_best_scores.csv (139 pieces, with onset_f1)
  - eval/results/lieder_stage3_v3_best/<piece_id>.musicxml.diagnostics.json
  - eval/results/lieder_stage3_v3_best/<piece_id>.tokens.jsonl (if available; else re-run predict_pdf)
  - data/openscore_lieder/scores/.../<piece_id>.mxl (for ground-truth clef extraction)

Writes:
  - docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES_CSV = REPO_ROOT / "eval/results/lieder_stage3_v3_best_scores.csv"
DEFAULT_TOKENS_DIR = REPO_ROOT / "eval/results/lieder_stage3_v3_best"
DEFAULT_GT_MXL_ROOT = REPO_ROOT / "data/openscore_lieder/scores"
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md"
BOTTOM_QUARTILE_THRESHOLD = 0.10


def load_bottom_quartile(scores_csv: Path, threshold: float = BOTTOM_QUARTILE_THRESHOLD) -> List[str]:
    bottom = []
    with scores_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["onset_f1"]) < threshold:
                bottom.append(row["piece_id"])
    return bottom


def extract_ground_truth_clefs(mxl_path: Path) -> List[List[str]]:
    """Return per-system list of clef tokens from a MusicXML source."""
    import music21
    score = music21.converter.parse(str(mxl_path))
    # Extract from each part's first measure; group by system if layout info present.
    # Simplification: return per-part clef tokens for measure 1 (proxy for system 1).
    clefs = []
    for part in score.parts:
        part_clefs = []
        for clef in part.flatten().getElementsByClass(music21.clef.Clef):
            part_clefs.append(_music21_clef_to_token(clef))
            break  # first only
        clefs.append(part_clefs)
    # Return as single-system-equivalent (caller handles per-system pairing).
    return [[c for part in clefs for c in part]]


def _music21_clef_to_token(clef) -> str:
    if isinstance(clef, type(__import__("music21").clef.TrebleClef())):
        return "clef-G2"
    if isinstance(clef, type(__import__("music21").clef.BassClef())):
        return "clef-F4"
    return f"clef-other-{clef.sign}{clef.line}"


def classify_piece(piece_tokens: Dict) -> List[str]:
    """Given a piece's predicted tokens + ground truth, return list of failure-mode tags."""
    tags = []
    systems = piece_tokens.get("systems", [])
    gt_clefs = piece_tokens.get("ground_truth_clefs_by_system", [])

    bass_clef_misread = False
    phantom = False
    for sys_idx, system in enumerate(systems):
        staves = system.get("staves", [])
        # Phantom-staff residual: more than 2 staves
        if len(staves) > 2:
            phantom = True
        # Bass-clef-misread: bottom staff predicted G2, ground truth F4
        if sys_idx < len(gt_clefs) and len(gt_clefs[sys_idx]) >= 2 and len(staves) >= 2:
            bottom_pred = staves[-1].get("clef_pred")
            bottom_gt = gt_clefs[sys_idx][-1]
            if bottom_pred == "clef-G2" and bottom_gt == "clef-F4":
                bass_clef_misread = True

    if bass_clef_misread:
        tags.append("bass-clef-misread")
    if phantom:
        tags.append("phantom-staff-residual")

    # Key/time-signature residual
    gt_time = piece_tokens.get("ground_truth_time_sig")
    pred_time = systems[0].get("time_sig_pred") if systems else None
    if gt_time and pred_time and gt_time != pred_time:
        tags.append("key-time-sig-residual")

    if not tags:
        tags.append("other")
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", type=Path, default=DEFAULT_SCORES_CSV)
    ap.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    ap.add_argument("--gt-mxl-root", type=Path, default=DEFAULT_GT_MXL_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MD)
    args = ap.parse_args()

    pieces = load_bottom_quartile(args.scores_csv)
    print(f"Bottom-quartile pieces (onset_f1 < {BOTTOM_QUARTILE_THRESHOLD}): {len(pieces)}")

    cluster_counts: Dict[str, int] = {}
    cluster_examples: Dict[str, List[str]] = {}

    for piece_id in pieces:
        tokens_path = args.tokens_dir / f"{piece_id}.tokens.jsonl"
        if not tokens_path.exists():
            print(f"  SKIP {piece_id} — no tokens dump (re-run predict_pdf --dump-tokens)")
            continue
        # Build piece_tokens dict from dump + ground truth.
        piece_tokens = _build_piece_tokens(piece_id, tokens_path, args.gt_mxl_root)
        tags = classify_piece(piece_tokens)
        for tag in tags:
            cluster_counts[tag] = cluster_counts.get(tag, 0) + 1
            cluster_examples.setdefault(tag, []).append(piece_id)

    _write_report(args.output, len(pieces), cluster_counts, cluster_examples)
    return 0


def _build_piece_tokens(piece_id: str, tokens_path: Path, gt_root: Path) -> Dict:
    """Parse the --dump-tokens JSONL into the structure classify_piece expects."""
    systems = []
    with tokens_path.open() as f:
        for line in f:
            entry = json.loads(line)
            # Each line is one system; extract per-staff clef + median pitch.
            staves = _extract_staves_from_token_stream(entry["tokens"])
            systems.append({"staves": staves})
    # Locate matching .mxl in gt_root recursively
    mxl_candidates = list(gt_root.rglob(f"{piece_id}.mxl"))
    gt_clefs = extract_ground_truth_clefs(mxl_candidates[0]) if mxl_candidates else []
    return {"systems": systems, "ground_truth_clefs_by_system": gt_clefs}


def _extract_staves_from_token_stream(tokens: List[str]) -> List[Dict]:
    """Split tokens into staves at <staff_idx_N> markers; extract clef + median pitch per staff."""
    staves: List[Dict] = []
    current = None
    for tok in tokens:
        if tok.startswith("<staff_idx_"):
            if current is not None:
                staves.append(current)
            current = {"clef_pred": None, "pitches": []}
        elif current is not None:
            if tok.startswith("clef-"):
                current["clef_pred"] = tok
            elif tok.startswith("note-"):
                pitch = _pitch_to_octave(tok)
                if pitch is not None:
                    current["pitches"].append(pitch)
    if current is not None:
        staves.append(current)
    for s in staves:
        s["median_octave_pred"] = (
            sorted(s["pitches"])[len(s["pitches"]) // 2] if s["pitches"] else None
        )
        del s["pitches"]
    return staves


def _pitch_to_octave(note_token: str) -> int | None:
    """note-C4 -> 4. note-Eb5 -> 5. Return None if unparseable."""
    import re
    m = re.match(r"note-[A-G][b#]?(-?\d+)", note_token)
    return int(m.group(1)) if m else None


def _write_report(output: Path, total: int, counts: Dict[str, int], examples: Dict[str, List[str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bottom-Quartile Lieder Failure-Mode Cluster (Stage 3 v3 best.pt)",
        "",
        f"**Date:** 2026-05-13",
        f"**Input:** `eval/results/lieder_stage3_v3_best_scores.csv` — {total} pieces with onset_f1 < {BOTTOM_QUARTILE_THRESHOLD}",
        "",
        "## Cluster counts",
        "",
        "| Cluster | Count | % of bottom-quartile |",
        "|---|---:|---:|",
    ]
    for tag, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total else 0.0
        lines.append(f"| {tag} | {count} | {pct:.1f}% |")
    lines.append("")
    lines.append("## Examples per cluster")
    lines.append("")
    for tag, ids in examples.items():
        lines.append(f"### {tag}")
        for piece_id in ids[:5]:
            lines.append(f"- `{piece_id}`")
        lines.append("")
    lines.append("## Gate check")
    lines.append("")
    bcm = counts.get("bass-clef-misread", 0)
    gate_pct = (bcm / total * 100) if total else 0.0
    lines.append(
        f"`bass-clef-misread` = {bcm}/{total} = {gate_pct:.1f}% of bottom-quartile."
    )
    lines.append("")
    if gate_pct >= 30:
        lines.append("**GATE PASS:** ≥30% → proceed to Phase 1.")
    else:
        lines.append("**GATE FAIL:** <30% → pause and re-scope before Phase 1.")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest tests/scripts/audit/test_cluster_bottom_quartile.py -v`
Expected: 4 PASS.

- [ ] **Step 1.5: Commit**

```bash
git add scripts/audit/cluster_bottom_quartile_lieder.py tests/scripts/audit/test_cluster_bottom_quartile.py
git commit -m "feat(audit): bottom-quartile lieder cluster analysis script (Phase 0a)"
```

### Task 2: Run Phase 0a and check gate

**Files:**
- Create: `docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md` (generated by script)

- [ ] **Step 2.1: Verify v3 token dumps exist or regenerate**

Run on seder:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  dir eval\results\lieder_stage3_v3_best\*.tokens.jsonl 2>&1 | findstr /R "tokens.jsonl"'
```
If empty: re-run `eval.run_lieder_eval` with `--dump-tokens` flag (extend the eval runner if needed; document additional task). Otherwise: proceed.

- [ ] **Step 2.2: Run the cluster script locally**

Run:
```bash
python scripts/audit/cluster_bottom_quartile_lieder.py
```
Expected: writes `docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md`, prints cluster counts.

- [ ] **Step 2.3: Check gate**

Open `docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md`. Read the "Gate check" section. If GATE FAIL: STOP and revise spec before continuing — `bass-clef-misread` may not be the dominant cluster. Surface this to the user.

- [ ] **Step 2.4: Commit report**

```bash
git add docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md
git commit -m "audit: bottom-quartile lieder cluster (Phase 0a)"
```

---

## Phase 0b — Overfit smoke test (GPU, ~10 min)

### Task 3: Build the overfit-smoke manifest

**Files:**
- Create: `scripts/data/build_overfit_smoke_manifest.py`
- Create: `data/manifests/overfit_smoke_v4.jsonl`
- Create: `tests/scripts/data/test_build_overfit_smoke_manifest.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/scripts/data/test_build_overfit_smoke_manifest.py`:

```python
"""Smoke-manifest builder must produce 20 entries deterministically."""
from pathlib import Path
import json
import tempfile

from scripts.data.build_overfit_smoke_manifest import build_manifest


def test_build_manifest_deterministic(tmp_path):
    """Builder is deterministic and produces 20 entries."""
    # Setup: fake source manifest with 100 candidate entries
    src = tmp_path / "src.jsonl"
    with src.open("w") as f:
        for i in range(100):
            f.write(
                json.dumps({
                    "source_path": f"data/processed/grandstaff_systems/page_{i:03d}_sys000.png",
                    "image_path": f"data/processed/grandstaff_systems/images/page_{i:03d}_sys000.png",
                    "tokens": ["<staff_idx_0>", "clef-G2", "<staff_idx_1>", "clef-F4"],
                    "staff_count": 2,
                    "split": "train",
                    "token_count": 100,
                })
                + "\n"
            )

    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    build_manifest(src_manifest=src, output=out_a, n=20, max_token_count=256)
    build_manifest(src_manifest=src, output=out_b, n=20, max_token_count=256)

    a_lines = out_a.read_text().splitlines()
    b_lines = out_b.read_text().splitlines()
    assert len(a_lines) == 20
    assert a_lines == b_lines  # deterministic


def test_build_manifest_respects_max_tokens(tmp_path):
    src = tmp_path / "src.jsonl"
    with src.open("w") as f:
        for i in range(50):
            tok_count = 100 if i < 25 else 500
            f.write(
                json.dumps({
                    "source_path": f"src_{i}.png",
                    "image_path": f"img_{i}.png",
                    "tokens": ["<staff_idx_0>", "clef-G2"],
                    "staff_count": 2,
                    "split": "train",
                    "token_count": tok_count,
                })
                + "\n"
            )
    out = tmp_path / "out.jsonl"
    build_manifest(src_manifest=src, output=out, n=20, max_token_count=256)
    entries = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(e["token_count"] <= 256 for e in entries)


def test_build_manifest_two_staff_only(tmp_path):
    src = tmp_path / "src.jsonl"
    with src.open("w") as f:
        for i in range(50):
            staff = 2 if i % 2 == 0 else 3
            f.write(
                json.dumps({
                    "source_path": f"src_{i}.png",
                    "image_path": f"img_{i}.png",
                    "tokens": [],
                    "staff_count": staff,
                    "split": "train",
                    "token_count": 100,
                })
                + "\n"
            )
    out = tmp_path / "out.jsonl"
    build_manifest(src_manifest=src, output=out, n=20, max_token_count=256)
    entries = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(e["staff_count"] == 2 for e in entries)
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `pytest tests/scripts/data/test_build_overfit_smoke_manifest.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3.3: Implement the builder**

Create `scripts/data/build_overfit_smoke_manifest.py`:

```python
"""Build the 20-sample overfit-smoke manifest for Stage 3 v4 Phase 0b.

Selection rules:
  - 2-staff grand-staff systems only.
  - token_count <= max_token_count (default 256) — keeps loss tight.
  - From `grandstaff_systems` train split.
  - Deterministic: sort by source_path lexicographically, take first N.

The point of this manifest is to function as a pre-flight check that the trainer
can overfit a small clean set. It is NOT used in main training.

Usage:
  python scripts/data/build_overfit_smoke_manifest.py \\
      --src-manifest data/processed/grandstaff_systems/manifests/synthetic_token_manifest.jsonl \\
      --output data/manifests/overfit_smoke_v4.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data/processed/grandstaff_systems/manifests/synthetic_token_manifest.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/manifests/overfit_smoke_v4.jsonl"


def build_manifest(
    src_manifest: Path,
    output: Path,
    n: int = 20,
    max_token_count: int = 256,
) -> None:
    candidates = []
    with src_manifest.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("split") != "train":
                continue
            if entry.get("staff_count") != 2:
                continue
            if entry.get("token_count", 0) > max_token_count:
                continue
            candidates.append(entry)

    candidates.sort(key=lambda e: e["source_path"])
    selected = candidates[:n]

    if len(selected) < n:
        raise RuntimeError(
            f"Only {len(selected)} candidates after filtering (wanted {n}). "
            f"Check src_manifest and filter criteria."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for entry in selected:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(selected)} entries to {output}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-manifest", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-token-count", type=int, default=256)
    args = ap.parse_args()
    build_manifest(args.src_manifest, args.output, args.n, args.max_token_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `pytest tests/scripts/data/test_build_overfit_smoke_manifest.py -v`
Expected: 3 PASS.

- [ ] **Step 3.5: Generate the manifest from grandstaff source**

Run on seder (the grandstaff manifest lives there):
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe scripts\data\build_overfit_smoke_manifest.py'
```
Expected: `data/manifests/overfit_smoke_v4.jsonl` written with 20 entries. Pull file back to local repo:
```bash
scp 10.10.1.29:'C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\data\manifests\overfit_smoke_v4.jsonl' \
    data/manifests/overfit_smoke_v4.jsonl
```

- [ ] **Step 3.6: Commit**

```bash
git add scripts/data/build_overfit_smoke_manifest.py \
        tests/scripts/data/test_build_overfit_smoke_manifest.py \
        data/manifests/overfit_smoke_v4.jsonl
git commit -m "feat(data): overfit-smoke manifest + builder (Phase 0b)"
```

### Task 4: Overfit smoke training config

**Files:**
- Create: `configs/train_stage3_v4_overfit_smoke.yaml`

- [ ] **Step 4.1: Write the config**

Create `configs/train_stage3_v4_overfit_smoke.yaml`:

```yaml
# Stage 3 v4 Phase 0b — overfit smoke test.
#
# Purpose: 10-minute pre-flight check that the trainer can overfit a small
# clean grand-staff subset. If train CE loss does not reach <0.05 by step 500
# (val CE <0.10 on the same set), training infrastructure has a regression
# and the v4 retrain must NOT launch. Investigate first.

stage_name: stage3-v4-overfit-smoke
stage_b_encoder: radio_h

epochs: 125                          # 20 samples × 125 = 2500 sample passes
effective_samples_per_epoch: 20
batch_size: 4
max_sequence_length: 1024
grad_accumulation_steps: 1

# Disable tier-grouped sampling; small enough for naive batching.
tier_grouped_sampling: false

# No augmentation; we want clean overfitting.
lr_dora: 0.002
lr_new_modules: 0.001
loraplus_lr_ratio: 2.0
warmup_steps: 50
schedule: cosine
weight_decay: 0.0

label_smoothing: 0.0
contour_loss_weight: 0.0

checkpoint_every_steps: 100
validate_every_steps: 50

dataset_mix:
  - dataset: overfit_smoke_v4
    ratio: 1.0
    split: train
    required: true
    manifest_path: data/manifests/overfit_smoke_v4.jsonl
```

NOTE: the `manifest_path` field may not be supported by the existing dataset registry. Check `src/data/index.py` and the dataset-mix loader in `src/train/train.py`. If unsupported, two options: (a) extend the loader to accept a direct manifest path, or (b) register `overfit_smoke_v4` as a "corpus" with a single manifest file in `data/processed/overfit_smoke_v4/manifests/`. Pick whichever is less invasive — likely (b), by symlink to the manifest. Document the chosen approach in a follow-up commit.

- [ ] **Step 4.2: Commit**

```bash
git add configs/train_stage3_v4_overfit_smoke.yaml
git commit -m "feat(config): Stage 3 v4 overfit smoke config (Phase 0b)"
```

### Task 5: Run overfit smoke and check gate

**Files:**
- Output: `checkpoints/full_radio_stage3_v4_overfit_smoke/*.pt` (on seder)
- Output: `logs/full_radio_stage3_v4_overfit_smoke_steps.jsonl` (on seder)

- [ ] **Step 5.1: Run smoke training on seder**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -u src\train\train.py \
    --stage-configs configs\train_stage3_v4_overfit_smoke.yaml \
    --mode execute \
    --resume-checkpoint checkpoints\full_radio_stage2_systems_v2\stage2-radio-systems-polyphonic_best.pt \
    --start-stage stage3-v4-overfit-smoke \
    --checkpoint-dir checkpoints\full_radio_stage3_v4_overfit_smoke \
    --step-log logs\full_radio_stage3_v4_overfit_smoke_steps.jsonl'
```
Expected: training completes in ~10 min, logs train_loss + val_loss per step.

- [ ] **Step 5.2: Pull step log + check gate criteria**

Run locally:
```bash
scp 10.10.1.29:'C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\logs\full_radio_stage3_v4_overfit_smoke_steps.jsonl' \
    /tmp/overfit_smoke.jsonl
python -c "
import json
for line in open('/tmp/overfit_smoke.jsonl'):
    e = json.loads(line)
    if e.get('global_step') == 500:
        print(f\"step 500: train_loss={e.get('train_loss')}, val_loss={e.get('val_loss')}\")
"
```
Expected: train_loss < 0.05 AND val_loss < 0.10. If either fails: STOP. Investigate before Phase 1.

- [ ] **Step 5.3: Append result to a smoke log**

Create or append to `docs/audits/2026-05-13-stage3-v4-results.md` (will be the Phase 3 doc later — start it now):

```markdown
# Stage 3 v4 Results

## Phase 0b — Overfit smoke (2026-05-13)

| step | train_loss | val_loss | gate |
|---|---:|---:|---|
| 500 | <fill> | <fill> | <PASS/FAIL> |
```

- [ ] **Step 5.4: Commit results so far**

```bash
git add docs/audits/2026-05-13-stage3-v4-results.md
git commit -m "audit: Stage 3 v4 Phase 0b overfit smoke results"
```

---

## Phase 1a — Real-scan distribution calibration (CPU, ~2h)

### Task 6: Real-scan measurement script

**Files:**
- Create: `scripts/audit/measure_real_scan_distribution.py`
- Create: `tests/scripts/audit/test_measure_real_scan_distribution.py`

- [ ] **Step 6.1: Write the failing test**

Create `tests/scripts/audit/test_measure_real_scan_distribution.py`:

```python
"""Test real-scan distribution measurement script."""
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from scripts.audit.measure_real_scan_distribution import (
    estimate_rotation_degrees,
    estimate_noise_sigma,
    estimate_blur_laplacian_var,
)


def _synthetic_staff_image(rot_deg: float = 0.0, noise_sigma: float = 0.0) -> Image.Image:
    """Synthesize a 200x600 image with 5 horizontal staff lines, optionally rotated/noised."""
    arr = np.full((200, 600), 255, dtype=np.uint8)
    for y in [40, 70, 100, 130, 160]:
        arr[y, :] = 0
    if noise_sigma > 0:
        noise = np.random.default_rng(42).normal(0, noise_sigma * 255, arr.shape)
        arr = np.clip(arr.astype(float) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    if rot_deg != 0.0:
        img = img.rotate(rot_deg, resample=Image.BICUBIC, fillcolor=255)
    return img


def test_estimate_rotation_recovers_known_angle():
    img = _synthetic_staff_image(rot_deg=1.0)
    estimated = estimate_rotation_degrees(img)
    assert abs(estimated - 1.0) < 0.5  # within 0.5°


def test_estimate_noise_is_monotonic():
    clean = _synthetic_staff_image(noise_sigma=0.0)
    noisy = _synthetic_staff_image(noise_sigma=0.05)
    assert estimate_noise_sigma(noisy) > estimate_noise_sigma(clean)


def test_estimate_blur_is_monotonic():
    clean = _synthetic_staff_image()
    blurred = clean.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(2.0))
    assert estimate_blur_laplacian_var(clean) > estimate_blur_laplacian_var(blurred)
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `pytest tests/scripts/audit/test_measure_real_scan_distribution.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 6.3: Implement the script**

Create `scripts/audit/measure_real_scan_distribution.py`:

```python
"""Measure scan-quality statistics from real scanned piano scores.

Inputs: a directory of images (Bethlehem.jpg, TimeMachine page renders, plus
~5 IMSLP scans). Outputs per-image and aggregated statistics on rotation, noise,
brightness/contrast, blur, and JPEG quality estimate.

Writes:
  - docs/audits/2026-05-13-real-scan-degradation-calibration.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image, ImageFilter


def estimate_rotation_degrees(img: Image.Image) -> float:
    """Return estimated rotation in degrees via Hough transform on edges."""
    import cv2
    gray = np.asarray(img.convert("L"))
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return 0.0
    angles_deg = []
    for line in lines[:50]:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90.0  # 0 = horizontal
        if -10.0 < angle < 10.0:
            angles_deg.append(angle)
    if not angles_deg:
        return 0.0
    return float(np.median(angles_deg))


def estimate_noise_sigma(img: Image.Image) -> float:
    """Return high-frequency residual stddev — proxy for noise level (normalized 0-1)."""
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    blurred = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32) / 255.0
    residual = gray - blurred
    return float(residual.std())


def estimate_blur_laplacian_var(img: Image.Image) -> float:
    """Laplacian variance — higher = sharper, lower = blurrier."""
    import cv2
    gray = np.asarray(img.convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_jpeg_quality(img_path: Path) -> int | None:
    """Heuristic JPEG quality estimate from quantization tables. Returns None for non-JPEG."""
    if img_path.suffix.lower() not in {".jpg", ".jpeg"}:
        return None
    try:
        with Image.open(img_path) as img:
            qtables = getattr(img, "quantization", None)
            if not qtables:
                return None
            # Lower mean → higher quality. Calibrate roughly: q50 mean ≈ 30, q90 ≈ 6.
            mean_q = np.mean(qtables[0])
            return int(round(max(0.0, min(100.0, 100.0 - (mean_q - 6.0) * 100.0 / 24.0))))
    except Exception:
        return None


def measure(img_path: Path) -> Dict:
    img = Image.open(img_path)
    return {
        "path": str(img_path),
        "size": img.size,
        "rotation_deg": estimate_rotation_degrees(img),
        "noise_sigma": estimate_noise_sigma(img),
        "blur_laplacian_var": estimate_blur_laplacian_var(img),
        "jpeg_quality": estimate_jpeg_quality(img_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", type=Path, required=True,
                    help="Directory of real scan images")
    ap.add_argument("--output", type=Path,
                    default=Path("docs/audits/2026-05-13-real-scan-degradation-calibration.md"))
    args = ap.parse_args()

    results = []
    for img_path in sorted(args.scan_dir.glob("*.[jp][pn]g")):
        results.append(measure(img_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real-Scan Degradation Calibration",
        "",
        f"**Date:** 2026-05-13",
        f"**Inputs:** {len(results)} scans from `{args.scan_dir}`",
        "",
        "## Per-scan measurements",
        "",
        "| File | rotation (°) | noise σ | blur var | JPEG q |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{Path(r['path']).name}` | {r['rotation_deg']:.2f} | "
            f"{r['noise_sigma']:.4f} | {r['blur_laplacian_var']:.1f} | "
            f"{r['jpeg_quality'] or 'n/a'} |"
        )

    if results:
        rotations = [r["rotation_deg"] for r in results]
        noises = [r["noise_sigma"] for r in results]
        blurs = [r["blur_laplacian_var"] for r in results]
        jpegs = [r["jpeg_quality"] for r in results if r["jpeg_quality"] is not None]
        lines += [
            "",
            "## Aggregate",
            "",
            f"- Rotation range: [{min(rotations):.2f}, {max(rotations):.2f}], median {np.median(rotations):.2f}",
            f"- Noise σ range: [{min(noises):.4f}, {max(noises):.4f}], median {np.median(noises):.4f}",
            f"- Blur var range: [{min(blurs):.1f}, {max(blurs):.1f}], median {np.median(blurs):.1f}",
        ]
        if jpegs:
            lines.append(f"- JPEG quality range: [{min(jpegs)}, {max(jpegs)}], median {int(np.median(jpegs))}")

        lines += [
            "",
            "## Recommended degradation pipeline parameters",
            "",
            "Set `src/data/scan_degradation.py` defaults to span the observed distribution:",
            f"- Rotation: ±{max(abs(min(rotations)), abs(max(rotations))) + 0.3:.1f}° uniform",
            f"- Noise σ: uniform [{max(0.01, min(noises)):.3f}, {max(noises) + 0.02:.3f}]",
            f"- Blur kernel σ: scaled so output blur_laplacian_var matches [{min(blurs):.0f}, {max(blurs):.0f}]",
        ]
        if jpegs:
            lines.append(f"- JPEG quality: uniform [{max(50, min(jpegs) - 5)}, {min(95, max(jpegs) + 5)}]")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `pytest tests/scripts/audit/test_measure_real_scan_distribution.py -v`
Expected: 3 PASS.

- [ ] **Step 6.5: Commit**

```bash
git add scripts/audit/measure_real_scan_distribution.py \
        tests/scripts/audit/test_measure_real_scan_distribution.py
git commit -m "feat(audit): real-scan distribution measurement script (Phase 1a)"
```

### Task 7: Run Phase 1a — measure real scans

**Files:**
- Create: `docs/audits/2026-05-13-real-scan-degradation-calibration.md` (generated)

- [ ] **Step 7.1: Collect 5–10 reference scans**

Gather into a directory:
- `Scanned_20251208-0833.jpg` (Bethlehem; user has this locally)
- `Receipt - Restoration Hardware.pdf` rendered to page images (~4 pages; user has this locally; use `pdftoppm` to convert)
- 5 IMSLP scans (manual download — pick beginner piano scores at 300+ dpi)

Move into `/tmp/real_scans_calibration/`.

- [ ] **Step 7.2: Render TimeMachine PDF pages**

Run:
```bash
pdftoppm -r 200 "Receipt - Restoration Hardware.pdf" /tmp/real_scans_calibration/timemachine -png
```
Expected: 4 PNGs in target dir.

- [ ] **Step 7.3: Run measurement script**

Run:
```bash
python scripts/audit/measure_real_scan_distribution.py --scan-dir /tmp/real_scans_calibration
```
Expected: writes `docs/audits/2026-05-13-real-scan-degradation-calibration.md` with per-scan and aggregate stats.

- [ ] **Step 7.4: Commit calibration report**

```bash
git add docs/audits/2026-05-13-real-scan-degradation-calibration.md
git commit -m "audit: real-scan degradation calibration (Phase 1a)"
```

---

## Phase 1b — Scan-degradation module + tests (CPU, ~2h)

### Task 8: Scan-degradation module (TDD)

**Files:**
- Create: `src/data/scan_degradation.py`
- Create: `tests/data/test_scan_degradation.py`

- [ ] **Step 8.1: Write the failing determinism test**

Create `tests/data/test_scan_degradation.py`:

```python
"""Unit tests for scan_degradation pipeline.

CPU-only, run locally — no GPU dependency.
"""
import numpy as np
from PIL import Image
import pytest

from src.data.scan_degradation import apply_scan_degradation


def _make_test_image(size=(800, 200)) -> Image.Image:
    """Synthesize a grayscale image with simulated staff lines."""
    arr = np.full((size[1], size[0]), 255, dtype=np.uint8)
    for y in [40, 70, 100, 130, 160]:
        arr[y, :] = 0
    return Image.fromarray(arr, mode="L")


def test_apply_is_deterministic_for_fixed_seed():
    img = _make_test_image()
    out_a = apply_scan_degradation(img, seed=12345)
    out_b = apply_scan_degradation(img, seed=12345)
    assert np.array_equal(np.asarray(out_a), np.asarray(out_b))


def test_apply_differs_for_different_seeds():
    img = _make_test_image()
    out_a = apply_scan_degradation(img, seed=12345)
    out_b = apply_scan_degradation(img, seed=67890)
    assert not np.array_equal(np.asarray(out_a), np.asarray(out_b))


def test_output_dimensions_preserved():
    img = _make_test_image(size=(800, 200))
    out = apply_scan_degradation(img, seed=12345)
    assert out.size == (800, 200)


def test_output_is_grayscale_when_input_is_grayscale():
    img = _make_test_image()
    out = apply_scan_degradation(img, seed=12345)
    assert out.mode == "L"


def test_output_differs_from_input():
    img = _make_test_image()
    out = apply_scan_degradation(img, seed=12345)
    assert not np.array_equal(np.asarray(img), np.asarray(out)), \
        "Degradation must actually change the image"
```

- [ ] **Step 8.2: Run test to verify it fails**

Run: `pytest tests/data/test_scan_degradation.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 8.3: Implement the module**

Create `src/data/scan_degradation.py`:

```python
"""Offline scan-realistic degradation for engraved music score images.

Used to derive scanned_grandstaff_systems from grandstaff_systems. Deterministic
per source via seeded RNG so the corpus is reproducible across machines.

Pipeline (applied in order):
    1. Rotation: ±1.5° uniform (border replicate)
    2. Perspective warp: corners offset up to 3% of dimension
    3. Brightness ±0.10, contrast ±0.12
    4. Paper-texture overlay: low-amplitude noise field, σ=0.04
    5. Light Gaussian blur: kernel 3, σ uniform[0.3, 0.9]
    6. Salt-and-pepper noise: fraction 0.0015
    7. JPEG round-trip: quality uniform[75, 90]

Parameter ranges are starting points; revise per Phase 1a calibration report.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_scan_degradation(img: Image.Image, seed: int) -> Image.Image:
    """Apply the full degradation pipeline. Deterministic for fixed (img, seed)."""
    rng = np.random.default_rng(seed)
    out = img

    # 1. Rotation
    rot_deg = float(rng.uniform(-1.5, 1.5))
    out = out.rotate(rot_deg, resample=Image.BICUBIC, fillcolor=255)

    # 2. Perspective warp (mild)
    out = _apply_perspective(out, rng, max_offset_frac=0.03)

    # 3. Brightness + contrast
    out = ImageEnhance.Brightness(out).enhance(1.0 + float(rng.uniform(-0.10, 0.10)))
    out = ImageEnhance.Contrast(out).enhance(1.0 + float(rng.uniform(-0.12, 0.12)))

    # 4. Paper-texture overlay
    arr = np.asarray(out, dtype=np.float32) / 255.0
    noise = rng.normal(0.0, 0.04, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    out = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L")

    # 5. Light Gaussian blur
    sigma = float(rng.uniform(0.3, 0.9))
    out = out.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 6. Salt-and-pepper noise
    out = _apply_salt_pepper(out, rng, fraction=0.0015)

    # 7. JPEG round-trip
    quality = int(rng.integers(75, 91))
    buf = BytesIO()
    out.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("L").copy()

    return out


def _apply_perspective(img: Image.Image, rng: np.random.Generator, max_offset_frac: float) -> Image.Image:
    """Apply a mild perspective warp by jittering the four corners."""
    import cv2
    w, h = img.size
    max_off_x = max_offset_frac * w
    max_off_y = max_offset_frac * h
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + rng.uniform(-1.0, 1.0, src.shape).astype(np.float32) * np.array(
        [max_off_x, max_off_y], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(src, dst)
    arr = np.asarray(img, dtype=np.uint8)
    warped = cv2.warpPerspective(
        arr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(warped, mode=img.mode)


def _apply_salt_pepper(img: Image.Image, rng: np.random.Generator, fraction: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.uint8).copy()
    flat = arr.reshape(-1)
    n = max(1, int(round(flat.size * fraction)))
    indices = rng.choice(flat.size, size=n, replace=False)
    split = n // 2
    flat[indices[:split]] = 0
    flat[indices[split:]] = 255
    return Image.fromarray(arr, mode=img.mode)
```

- [ ] **Step 8.4: Run tests to verify they pass**

Run: `pytest tests/data/test_scan_degradation.py -v`
Expected: 5 PASS.

- [ ] **Step 8.5: Update degradation parameters per Phase 1a calibration**

Open `docs/audits/2026-05-13-real-scan-degradation-calibration.md`. If the recommended ranges differ from the starting defaults, update `src/data/scan_degradation.py`'s ranges to match. Re-run tests:

Run: `pytest tests/data/test_scan_degradation.py -v`
Expected: 5 PASS.

- [ ] **Step 8.6: Commit**

```bash
git add src/data/scan_degradation.py tests/data/test_scan_degradation.py
git commit -m "feat(data): scan_degradation module + tests (Phase 1b)"
```

---

## Phase 1c — Build scanned_grandstaff_systems corpus + cache rebuild

### Task 9: Corpus build script

**Files:**
- Create: `scripts/data/build_scanned_grandstaff_systems.py`
- Create: `tests/scripts/data/test_build_scanned_grandstaff_systems.py`

- [ ] **Step 9.1: Write the failing test**

Create `tests/scripts/data/test_build_scanned_grandstaff_systems.py`:

```python
"""Test scanned_grandstaff_systems corpus builder."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.data.build_scanned_grandstaff_systems import build_corpus


def test_build_corpus_preserves_token_labels(tmp_path):
    src_root = tmp_path / "src"
    src_images = src_root / "images"
    src_manifests = src_root / "manifests"
    src_images.mkdir(parents=True)
    src_manifests.mkdir(parents=True)
    # Make a test image
    img = Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L")
    img.save(src_images / "test.png")
    # Make a source manifest with one entry
    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        f.write(json.dumps({
            "source_path": "test.png",
            "image_path": str(src_images / "test.png"),
            "tokens": ["<staff_idx_0>", "clef-G2", "<staff_idx_1>", "clef-F4"],
            "split": "train",
            "staff_count": 2,
            "token_count": 4,
        }) + "\n")

    out_root = tmp_path / "out"
    build_corpus(src_root=src_root, out_root=out_root, limit=None)

    out_manifest = out_root / "manifests/synthetic_token_manifest.jsonl"
    assert out_manifest.exists()
    entries = [json.loads(line) for line in out_manifest.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["tokens"] == ["<staff_idx_0>", "clef-G2", "<staff_idx_1>", "clef-F4"]
    assert entries[0]["split"] == "train"


def test_build_corpus_writes_degraded_image(tmp_path):
    src_root = tmp_path / "src"
    src_images = src_root / "images"
    src_manifests = src_root / "manifests"
    src_images.mkdir(parents=True)
    src_manifests.mkdir(parents=True)
    img = Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L")
    img.save(src_images / "test.png")
    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        f.write(json.dumps({
            "source_path": "test.png",
            "image_path": str(src_images / "test.png"),
            "tokens": [],
            "split": "train",
            "staff_count": 2,
            "token_count": 0,
        }) + "\n")

    out_root = tmp_path / "out"
    build_corpus(src_root=src_root, out_root=out_root, limit=None)

    out_images = list((out_root / "images").glob("*.png"))
    assert len(out_images) == 1
    out_img = Image.open(out_images[0])
    assert out_img.size == img.size
    # Pixel content must differ from source
    assert not np.array_equal(np.asarray(img), np.asarray(out_img.convert("L")))


def test_build_corpus_idempotent(tmp_path):
    """Re-running with same inputs yields same outputs (deterministic seeds)."""
    src_root = tmp_path / "src"
    src_images = src_root / "images"
    src_manifests = src_root / "manifests"
    src_images.mkdir(parents=True)
    src_manifests.mkdir(parents=True)
    img = Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L")
    img.save(src_images / "test.png")
    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        f.write(json.dumps({
            "source_path": "test.png",
            "image_path": str(src_images / "test.png"),
            "tokens": [],
            "split": "train",
            "staff_count": 2,
            "token_count": 0,
        }) + "\n")

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    build_corpus(src_root=src_root, out_root=out_a, limit=None)
    build_corpus(src_root=src_root, out_root=out_b, limit=None)

    img_a = np.asarray(Image.open(next((out_a / "images").glob("*.png"))))
    img_b = np.asarray(Image.open(next((out_b / "images").glob("*.png"))))
    assert np.array_equal(img_a, img_b)
```

- [ ] **Step 9.2: Run test to verify it fails**

Run: `pytest tests/scripts/data/test_build_scanned_grandstaff_systems.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 9.3: Implement the build script**

Create `scripts/data/build_scanned_grandstaff_systems.py`:

```python
"""Build the scanned_grandstaff_systems corpus from grandstaff_systems.

For each source image in grandstaff_systems, apply src.data.scan_degradation.apply_scan_degradation
with a seed derived from source_path. Copy token labels verbatim — only the image changes.

Output manifest mirrors grandstaff's schema with updated image_path.

Usage (on seder, where grandstaff lives):
  python scripts\\data\\build_scanned_grandstaff_systems.py
  python scripts\\data\\build_scanned_grandstaff_systems.py --limit 100  # dry-run subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from src.data.scan_degradation import apply_scan_degradation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data/processed/grandstaff_systems"
DEFAULT_OUT = REPO_ROOT / "data/processed/scanned_grandstaff_systems"


def _seed_from_path(source_path: str) -> int:
    h = hashlib.sha256(source_path.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def build_corpus(src_root: Path, out_root: Path, limit: int | None = None) -> None:
    src_manifest = src_root / "manifests/synthetic_token_manifest.jsonl"
    out_images = out_root / "images"
    out_manifests = out_root / "manifests"
    out_images.mkdir(parents=True, exist_ok=True)
    out_manifests.mkdir(parents=True, exist_ok=True)

    out_manifest_path = out_manifests / "synthetic_token_manifest.jsonl"
    written = 0
    skipped = 0
    with src_manifest.open() as fin, out_manifest_path.open("w") as fout:
        for line in fin:
            if limit is not None and written >= limit:
                break
            entry = json.loads(line)
            src_img_path = Path(entry["image_path"])
            if not src_img_path.is_absolute():
                src_img_path = src_root / src_img_path.name  # fallback if relative
            stem = src_img_path.stem
            out_img_path = out_images / f"{stem}.png"

            if out_img_path.exists():
                skipped += 1
            else:
                with Image.open(src_img_path) as src_img:
                    degraded = apply_scan_degradation(src_img.convert("L"), seed=_seed_from_path(entry["source_path"]))
                degraded.save(out_img_path, format="PNG")

            new_entry = dict(entry)
            new_entry["image_path"] = str(out_img_path)
            new_entry["original_image_path"] = str(src_img_path)
            fout.write(json.dumps(new_entry) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"  ... {written} entries written")
    print(f"Wrote {written} entries to {out_manifest_path} ({skipped} images skipped because already present)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: limit number of entries (for dry-run testing)")
    args = ap.parse_args()
    build_corpus(args.src_root, args.out_root, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 9.4: Run tests to verify they pass**

Run: `pytest tests/scripts/data/test_build_scanned_grandstaff_systems.py -v`
Expected: 3 PASS.

- [ ] **Step 9.5: Commit**

```bash
git add scripts/data/build_scanned_grandstaff_systems.py \
        tests/scripts/data/test_build_scanned_grandstaff_systems.py
git commit -m "feat(data): scanned_grandstaff_systems corpus builder (Phase 1b)"
```

### Task 10: Build the corpus on seder

**Files:** none in repo; corpus output lives on seder.

- [ ] **Step 10.1: Dry-run on seder with --limit 100**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe scripts\data\build_scanned_grandstaff_systems.py --limit 100'
```
Expected: 100 images in `data/processed/scanned_grandstaff_systems/images/`, manifest with 100 lines.

- [ ] **Step 10.2: Spot-check 10 dry-run outputs visually**

Run:
```bash
scp 10.10.1.29:'C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\data\processed\scanned_grandstaff_systems\images\*.png' /tmp/scanned_spot/ 2>&1 | head -20
# Open in viewer manually; confirm rotation, noise, JPEG appearance look like real scans
```

If they look wrong (e.g., over-rotated, too noisy, or too clean): tune the degradation parameters in `src/data/scan_degradation.py`, re-run dry-run, re-inspect. Iterate.

- [ ] **Step 10.3: Full build on seder**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe scripts\data\build_scanned_grandstaff_systems.py'
```
Expected: ~107k images, ~30 min wall-clock on seder.

- [ ] **Step 10.4: Verify counts**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -c "from pathlib import Path; manifest = Path(\"data/processed/scanned_grandstaff_systems/manifests/synthetic_token_manifest.jsonl\"); n = sum(1 for _ in manifest.open()); imgs = len(list(Path(\"data/processed/scanned_grandstaff_systems/images\").glob(\"*.png\"))); print(f\"manifest entries: {n}, images: {imgs}\")"'
```
Expected: matches grandstaff system count (~107,724).

### Task 11: Update combined manifest builder to include scanned_grandstaff

**Files:**
- Modify: `scripts/build_stage3_combined_manifest.py`

- [ ] **Step 11.1: Add scanned_grandstaff_systems registry entry**

Open `scripts/build_stage3_combined_manifest.py`. Find the dataset registry around line 121 (the `("synthetic_systems", ...), ("grandstaff_systems", ...)` block). Add a new entry:

```python
("synthetic_systems", args.synthetic_systems_manifest),
("grandstaff_systems", args.grandstaff_systems_manifest),
("scanned_grandstaff_systems", args.scanned_grandstaff_systems_manifest),  # NEW
("primus_systems", args.primus_systems_manifest),
("cameraprimus_systems", args.cameraprimus_systems_manifest),
```

Add the corresponding CLI argument near the other manifest paths. Default to `data/processed/scanned_grandstaff_systems/manifests/synthetic_token_manifest.jsonl`.

Add provenance-extraction for `scanned_grandstaff_systems` if applicable — likely inherits grandstaff's provenance (since tokens are copied verbatim). If unsure, copy the grandstaff branch with the new corpus name.

- [ ] **Step 11.2: Regenerate the combined manifest**

Run on seder:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe scripts\build_stage3_combined_manifest.py'
```
Expected: writes `src/data/manifests/token_manifest_stage3.jsonl` with the new corpus included; updates `token_manifest_stage3_audit.json`.

- [ ] **Step 11.3: Verify audit counts**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  type src\data\manifests\token_manifest_stage3_audit.json'
```
Expected: shows scanned_grandstaff_systems with ~107k entries; total ~411k (previous 301,045 + ~107k new).

- [ ] **Step 11.4: Commit**

```bash
git add scripts/build_stage3_combined_manifest.py \
        src/data/manifests/token_manifest_stage3_audit.json \
        src/data/manifests/token_manifest_stage3.jsonl
git commit -m "feat(data): register scanned_grandstaff_systems in combined manifest"
```

### Task 12: Rebuild encoder cache on seder

**Files:** new cache in `data/cache/encoder/<new_hash16>/` on seder.

- [ ] **Step 12.1: Launch cache build**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe scripts\build_encoder_cache.py \
    --manifest src\data\manifests\token_manifest_stage3.jsonl \
    --datasets synthetic_systems grandstaff_systems scanned_grandstaff_systems primus_systems \
    --output data\cache\encoder \
    --encoder radio_h \
    --encoder-checkpoint checkpoints\full_radio_stage2_systems_v2\stage2-radio-systems-polyphonic_best.pt'
```
Use the appropriate flags for `build_encoder_cache.py` — check `--help` for current argument names. Adjust accordingly.

Expected: ~3-4h wall-clock, writes new cache directory keyed by a new `cache_hash16`. Capture the new hash from the build log.

- [ ] **Step 12.2: Record new cache_hash16**

Note the new cache hash (e.g., `b1234567c89abcde`). Will be referenced in the v4 training config.

- [ ] **Step 12.3: A2 parity verification on new cache**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe scripts\validate_cache_correctness.py \
    --cache-hash16 <NEW_HASH> --num-samples 10 --interp BILINEAR'
```
Expected: `max_abs_diff < 0.01`. If still fails (like v3 did), it's the known pre-existing cache-vs-live preprocessing issue — document in the Phase 3 results doc and proceed (v3 shipped despite this).

---

## Phase 2 — Stage 3 v4 retrain (~1h)

### Task 13: Stage 3 v4 training config

**Files:**
- Create: `configs/train_stage3_radio_systems_v4.yaml`

- [ ] **Step 13.1: Write the config**

Create `configs/train_stage3_radio_systems_v4.yaml`:

```yaml
# Stage 3 Phase 1 v4 — scan-realistic retrain.
#
# Supersedes v3 by adding scanned_grandstaff_systems (offline scan-degraded
# grandstaff). Dataset_mix rebalances grandstaff's share half-and-half between
# clean and scanned variants:
#   synth 0.15 / grand 0.30 / scanned_grand 0.30 / primus 0.125 / camera 0.125
#
# Phase 0a (bottom-quartile lieder cluster) and Phase 0b (overfit smoke) gates
# must pass before launching this run. See:
#   docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md
#   docs/audits/2026-05-13-stage3-v4-results.md (Phase 0b section)
#
# Run command (on seder):
#   ssh 10.10.1.29 'cd "C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO" && \
#     venv-cu132\\Scripts\\python -u src/train/train.py \
#       --stage-configs configs/train_stage3_radio_systems_v4.yaml \
#       --mode execute \
#       --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \
#       --start-stage stage3-radio-systems-frozen-encoder-v4 \
#       --checkpoint-dir checkpoints/full_radio_stage3_v4 \
#       --token-manifest src/data/manifests/token_manifest_stage3.jsonl \
#       --step-log logs/full_radio_stage3_v4_steps.jsonl'

stage_name: stage3-radio-systems-frozen-encoder-v4
stage_b_encoder: radio_h

epochs: 1
effective_samples_per_epoch: 9000
batch_size: 1                        # SENTINEL — unused with tier_grouped_sampling
max_sequence_length: 1024
grad_accumulation_steps: 1           # SENTINEL — unused with tier_grouped_sampling

tier_grouped_sampling: true
b_cached: 16
b_live: 2
grad_accumulation_steps_cached: 1
grad_accumulation_steps_live: 8
cached_data_ratio: 0.875
cache_root: data/cache/encoder
cache_hash16: <NEW_HASH_FROM_TASK_12>

lr_dora: 0.0005
lr_new_modules: 0.0003
loraplus_lr_ratio: 2.0
warmup_steps: 500
schedule: cosine
weight_decay: 0.01

label_smoothing: 0.01
contour_loss_weight: 0.01

checkpoint_every_steps: 500
validate_every_steps: 500

# Within-cached weights (sum 1.0); total share = 0.875 × ratio.
#   synth:        6/35 = 0.171428 → 0.150
#   grand:       12/35 = 0.342857 → 0.300
#   scanned:     12/35 = 0.342857 → 0.300
#   primus:       5/35 = 0.142857 → 0.125
# cameraprimus: 0.125 via live tier.
dataset_mix:
  - dataset: synthetic_systems
    ratio: 0.171428
    split: train
    required: true
  - dataset: grandstaff_systems
    ratio: 0.342857
    split: train
    required: true
  - dataset: scanned_grandstaff_systems
    ratio: 0.342857
    split: train
    required: true
  - dataset: primus_systems
    ratio: 0.142857
    split: train
    required: true
  - dataset: cameraprimus_systems
    ratio: 0.0
    split: train
    required: true
```

- [ ] **Step 13.2: Commit**

```bash
git add configs/train_stage3_radio_systems_v4.yaml
git commit -m "feat(config): Stage 3 v4 training config (scan-realistic mix)"
```

### Task 14: Smoke 500-step verification

**Files:** `checkpoints/full_radio_stage3_v4/_step_0000500.pt` on seder.

- [ ] **Step 14.1: Launch 500-step verification run**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -u src\train\train.py \
    --stage-configs configs\train_stage3_radio_systems_v4.yaml \
    --mode execute \
    --resume-checkpoint checkpoints\full_radio_stage2_systems_v2\stage2-radio-systems-polyphonic_best.pt \
    --start-stage stage3-radio-systems-frozen-encoder-v4 \
    --checkpoint-dir checkpoints\full_radio_stage3_v4 \
    --token-manifest src\data\manifests\token_manifest_stage3.jsonl \
    --step-log logs\full_radio_stage3_v4_steps.jsonl \
    --max-steps 500'
```
Expected: completes in ~3-4 min. Logs show `[freeze] trainable encoder params: 0`. Checkpoint written.

- [ ] **Step 14.2: Verify pre-flight assertion fired**

Inspect the run output. Confirm: `[freeze] trainable encoder params: 0` AND `[freeze] trainable decoder params: 312` (or current decoder-param count). If trainable encoder params > 0: the auto-freeze regressed; debug before continuing.

### Task 15: Full retrain

- [ ] **Step 15.1: Launch full retrain (resume from step 500)**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -u src\train\train.py \
    --stage-configs configs\train_stage3_radio_systems_v4.yaml \
    --mode execute \
    --resume-checkpoint checkpoints\full_radio_stage3_v4\_step_0000500.pt \
    --checkpoint-dir checkpoints\full_radio_stage3_v4 \
    --token-manifest src\data\manifests\token_manifest_stage3.jsonl \
    --step-log logs\full_radio_stage3_v4_steps.jsonl'
```
NOTE: `--start-stage` is OMITTED on resume (it would reset global_step). Expected: training continues from step 501, completes 9000 steps in ~1h.

- [ ] **Step 15.2: Verify best checkpoint exists**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  dir checkpoints\full_radio_stage3_v4\*best*.pt'
```
Expected: `stage3-radio-systems-frozen-encoder-v4_best.pt` present.

- [ ] **Step 15.3: Pull step log and inspect loss curve**

Run:
```bash
scp 10.10.1.29:'C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\logs\full_radio_stage3_v4_steps.jsonl' \
    logs/full_radio_stage3_v4_steps.jsonl
python -c "
import json
best = None
for line in open('logs/full_radio_stage3_v4_steps.jsonl'):
    e = json.loads(line)
    if 'val_loss' in e:
        if best is None or e['val_loss'] < best[1]:
            best = (e['global_step'], e['val_loss'])
print(f'Best val_loss: {best[1]:.4f} at step {best[0]}')
"
```
Expected: best val_loss reported. Note step for later reference.

---

## Phase 3 — Evaluation

### Task 16: Primary gate — Bethlehem + TimeMachine

**Files:**
- Output: `/tmp/bethlehem_v4.musicxml`, `/tmp/timemachine_v4.musicxml` (on seder)

- [ ] **Step 16.1: Run Bethlehem through predict_pdf with v4**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf \
    Scanned_20251208-0833.jpg /tmp/bethlehem_v4.musicxml \
    --stage-b-weights checkpoints/full_radio_stage3_v4/stage3-radio-systems-frozen-encoder-v4_best.pt \
    --dump-tokens /tmp/bethlehem_v4.tokens.jsonl'
```
Expected: ≥3 of 4 systems detected (Issue 1b out of scope), MXL written.

- [ ] **Step 16.2: Run TimeMachine through predict_pdf with v4**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf \
    "Receipt - Restoration Hardware.pdf" /tmp/timemachine_v4.musicxml \
    --stage-b-weights checkpoints/full_radio_stage3_v4/stage3-radio-systems-frozen-encoder-v4_best.pt \
    --dump-tokens /tmp/timemachine_v4.tokens.jsonl'
```
Expected: 4 systems detected, MXL written.

- [ ] **Step 16.3: Verify bass clef + pitches**

Pull MXLs locally and verify via music21:
```bash
scp 10.10.1.29:/tmp/bethlehem_v4.musicxml /tmp/
scp 10.10.1.29:/tmp/timemachine_v4.musicxml /tmp/

python << 'EOF'
import music21
from collections import defaultdict

for tag, path in [("Bethlehem", "/tmp/bethlehem_v4.musicxml"),
                   ("TimeMachine", "/tmp/timemachine_v4.musicxml")]:
    score = music21.converter.parse(path)
    print(f"\n=== {tag} ===")
    for part_idx, part in enumerate(score.parts):
        clefs = [c.sign + str(c.line) for c in part.flatten().getElementsByClass(music21.clef.Clef)]
        pitches = [p.midi for p in part.flatten().pitches]
        median_octave = (sorted(pitches)[len(pitches) // 2] // 12) - 1 if pitches else None
        print(f"  Part {part_idx}: clefs={clefs}, n_pitches={len(pitches)}, median_octave={median_octave}")
EOF
```

Manual verification:
- Bethlehem Part 1 (treble): clefs all `G2`, median octave ≥ 4.
- Bethlehem Part 2 (bass): clefs all `F4`, median octave ≤ 3.
- TimeMachine same pattern.

Pass = both pieces meet the per-part criteria. Single failure (e.g., one Part 2 system shows G2 or median octave ≥ 4) = primary gate FAIL.

- [ ] **Step 16.4: Record result**

Append to `docs/audits/2026-05-13-stage3-v4-results.md`:

```markdown
## Phase 3a — Primary gate (Bethlehem + TimeMachine, 2026-05-13)

| Piece | Stage A systems | Treble clefs | Bass clefs | Bass median octave | Verdict |
|---|---:|---|---|---|---|
| Bethlehem | <fill> | <fill> | <fill> | <fill> | PASS/FAIL |
| TimeMachine | <fill> | <fill> | <fill> | <fill> | PASS/FAIL |

**Primary gate: PASS/FAIL.**
```

### Task 17: Secondary — lieder ship-gate

- [ ] **Step 17.1: Run 139-piece lieder eval on v4**

Run:
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m eval.run_lieder_eval \
    --stage-b-weights checkpoints/full_radio_stage3_v4/stage3-radio-systems-frozen-encoder-v4_best.pt \
    --run-name stage3_v4_best \
    --run-scoring'
```
Expected: ~3h wall-clock, writes `eval/results/lieder_stage3_v4_best_scores.csv`.

- [ ] **Step 17.2: Compute summary statistics**

Run:
```bash
scp 10.10.1.29:'C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\eval\results\lieder_stage3_v4_best_scores.csv' \
    eval/results/lieder_stage3_v4_best_scores.csv

python << 'EOF'
import csv
import statistics

with open("eval/results/lieder_stage3_v4_best_scores.csv") as f:
    rows = list(csv.DictReader(f))
v4_scores = [float(r["onset_f1"]) for r in rows]
print(f"N={len(v4_scores)}  mean={statistics.mean(v4_scores):.4f}  median={statistics.median(v4_scores):.4f}")
print(f"  bottom-quartile (<0.10): {sum(1 for s in v4_scores if s < 0.10)}")
print(f"  >=0.50: {sum(1 for s in v4_scores if s >= 0.50)}")

# Compare to v3 bottom-quartile
with open("eval/results/lieder_stage3_v3_best_scores.csv") as f:
    v3_rows = {r["piece_id"]: float(r["onset_f1"]) for r in csv.DictReader(f)}
v3_bottom = {pid: s for pid, s in v3_rows.items() if s < 0.10}
v4_by_pid = {r["piece_id"]: float(r["onset_f1"]) for r in rows}
moved = sum(1 for pid in v3_bottom if v4_by_pid.get(pid, 0.0) >= 0.10)
print(f"v3-bottom-quartile pieces moved to v4 >= 0.10: {moved}/{len(v3_bottom)}")
EOF
```

Pass criteria:
- Corpus mean ≥ 0.2398 (no regression vs v3).
- Bottom-quartile movement: ≥17 of 33 v3-bottom pieces now have v4 onset_f1 ≥ 0.10.

- [ ] **Step 17.3: Record result**

Append to `docs/audits/2026-05-13-stage3-v4-results.md`:

```markdown
## Phase 3b — Lieder ship-gate (2026-05-13)

| Metric | v3 | v4 | Δ |
|---|---:|---:|---:|
| Corpus mean onset_f1 | 0.2398 | <fill> | <fill> |
| Median | 0.1830 | <fill> | <fill> |
| Bottom-quartile (<0.10) count | 33 | <fill> | <fill> |
| v3-bottom moved to v4 ≥ 0.10 | — | <fill>/33 | — |
```

### Task 18: A3 per-corpus token-accuracy spot check

- [ ] **Step 18.1: Run A3 audit on v4**

There should be an existing A3 audit script from v2/v3. Re-run it with v4 best.pt, adding the new corpus:

```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe scripts\audit\stage3_a3_token_accuracy.py \
    --weights checkpoints/full_radio_stage3_v4/stage3-radio-systems-frozen-encoder-v4_best.pt \
    --corpora synthetic_systems grandstaff_systems scanned_grandstaff_systems primus_systems cameraprimus_systems \
    --n 5 --out /tmp/v4_a3.json'
```
NOTE: confirm the A3 audit script path (check `scripts/audit/` for the v3 equivalent; the name may differ). If it doesn't exist as a single script, document the manual procedure used in v3 and reuse it.

- [ ] **Step 18.2: Record A3 results**

Append to `docs/audits/2026-05-13-stage3-v4-results.md`:

```markdown
## Phase 3d — A3 token accuracy (2026-05-13)

| Corpus | v3 | v4 | Δ |
|---|---:|---:|---:|
| synthetic_systems | 0.257 | <fill> | <fill> |
| grandstaff_systems | 0.904 | <fill> | <fill> |
| scanned_grandstaff_systems | (new) | <fill> | — |
| primus_systems | 0.810 | <fill> | <fill> |
| cameraprimus_systems | 0.869 | <fill> | <fill> |

Pass criteria: grandstaff_systems ≥ 0.80 (was 0.904). scanned_grandstaff_systems ≥ 0.50.
```

### Task 19: Final results doc + verdict

- [ ] **Step 19.1: Write verdict in results doc**

Update `docs/audits/2026-05-13-stage3-v4-results.md` with TL;DR + overall verdict:

```markdown
## TL;DR

<2-3 sentences summarizing the v4 retrain outcome, citing the primary gate
(Bethlehem + TimeMachine) and lieder corpus mean.>

## Verdict

**SHIP / NEAR-SHIP / FAIL** — <one-line reason>.

If FAIL: escalate to Approach (B) multi-variant cache rebuild — see escalation
section of the spec.
```

- [ ] **Step 19.2: Commit final results**

```bash
git add docs/audits/2026-05-13-stage3-v4-results.md \
        eval/results/lieder_stage3_v4_best_scores.csv \
        logs/full_radio_stage3_v4_steps.jsonl
git commit -m "audit: Stage 3 v4 retrain results"
```

### Task 20: PR

- [ ] **Step 20.1: Open PR**

Run:
```bash
git push -u origin feat/stage3-v4-scan-realistic
gh pr create --title "feat(train): Stage 3 v4 scan-realistic retrain" --body "$(cat <<'EOF'
## Summary
- Adds scan-realistic corpus `scanned_grandstaff_systems` derived from grandstaff via offline degradation pipeline.
- Stage 3 v4 retrain with rebalanced mix (synth 0.15 / grand 0.30 / scanned_grand 0.30 / primus 0.125 / camera 0.125).
- Pre-flight gates: bottom-quartile lieder cluster (Phase 0a) + overfit smoke (Phase 0b).

## Test plan
- [ ] Phase 0a gate passes (≥30% bass-clef-misread cluster)
- [ ] Phase 0b gate passes (train CE < 0.05 by step 500)
- [ ] Phase 3a primary gate: Bethlehem + TimeMachine bass clefs + pitches correct
- [ ] Phase 3b lieder corpus mean ≥ 0.2398 (no regression)
- [ ] Phase 3d A3 grandstaff ≥ 0.80 (no engraving regression)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Escalation (if Phase 3a fails)

Do not implement here — captured in the [spec's Section 7](../specs/2026-05-13-stage3-v4-scan-realistic-retrain-design.md#escalation-paths-documented-not-part-of-v4-plan-of-record).

Next sub-projects, in order:
1. **v4.1** — multi-variant cache rebuild (3 degraded variants per scanned_grandstaff source).
2. **v4.2** — no-cache live-tier fine-tune with online augmentation + encoder unfreeze.

Each gets its own spec + plan if v4 doesn't pass primary gate.

---

## Self-review notes

- All spec sections have corresponding tasks: Phase 0a (Tasks 1-2), Phase 0b (Tasks 3-5), Phase 1a (Tasks 6-7), Phase 1b (Task 8), Phase 1c (Tasks 9-12), Phase 2 (Tasks 13-15), Phase 3 (Tasks 16-19), PR (Task 20).
- Placeholders are intentional: `<NEW_HASH_FROM_TASK_12>` (cache hash determined at runtime), `<fill>` in result tables (filled when results arrive).
- The Task 11 step ("update combined manifest builder") has the most implementation risk because it depends on the exact CLI shape of `build_stage3_combined_manifest.py`. The implementer should run `--help` first to confirm argument names. Same for `build_encoder_cache.py` in Task 12.
- Task 4 (overfit smoke config's `manifest_path` direct reference) may need the dataset loader to be extended, OR the manifest can be placed at `data/processed/overfit_smoke_v4/manifests/synthetic_token_manifest.jsonl` to fit the existing pattern. Choice deferred to implementer; flag if it turns out to be invasive.
