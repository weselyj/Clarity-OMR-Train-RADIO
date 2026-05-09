# Stage 3 Phase 2 — Evaluation + Decision Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the three Phase 2 eval surfaces (lieder onset_f1, per-dataset quality regression-check, MusicXML validity rate) against Stage 3 v2 `_best.pt@step5500`, mechanize the decision gate from spec §"Decision flow", and produce a final handoff that says Ship / Investigate / Pivot / Diagnose with full evidence.

**Architecture:** Phase 2 is *operational* — the eval infrastructure already exists (`eval/run_lieder_eval.py` + `eval/score_lieder_eval.py` for the lieder gate; `eval/run_clarity_demo_eval.py` + `eval/score_demo_eval.py` for per-dataset quality; `src/eval/metrics.py:musicxml_validity_from_tokens` for the validity rate, enabled in commit `354d25b`). Stage 2 v2 used these same drivers to produce the 96.8 / 93.4 / 83.1 / 75.2 baselines. Plan D wires the existing drivers to the v2 ship artifact, plus adds two small TDD tools — a stratified-onset_f1 analyzer and a decision-gate evaluator — that turn raw CSV output into a verdict.

**Tech Stack:** Python 3.13 (GPU box) / 3.14 (local), pytest, existing eval infra (music21, zss, etc.), SSH + scp to GPU box at `10.10.1.29` (Windows, `venv-cu132\Scripts\python`). All inference runs on the GPU box; scoring + decision-gate analysis runs locally.

---

## Decisions locked at plan time

1. **Per-dataset quality eval driver = `src/eval/evaluate_stage_b_checkpoint.py`.** REVISED 2026-05-09 after discovery: the original draft assumed `eval/run_clarity_demo_eval.py` would handle all 5 datasets; it doesn't (it's hardcoded to 4 HF demo pieces). The actual Stage 2 v2 pipeline was `evaluate_stage_b_checkpoint.py` — it runs Stage B inference from a token manifest, calls `run_eval.evaluate_rows`, auto-groups by `dataset` field, and produces per-dataset composite quality scores via `src/eval/metrics.py:quality_score`. **Lock: this single driver replaces the original Tasks 2 and 3.**

2. **The per-dataset metric is composite `quality_score` (0-100), not pure onset_f1.** Per `src/eval/metrics.py:quality_score`, this is a weighted blend of SER, note_event_f1, pitch accuracy, rhythm accuracy, and onset_f1. The spec's historical 96.8 / 93.4 / 83.1 / 75.2 were almost certainly produced by this composite (no other eval driver in the repo aggregates 0-100 numbers per dataset). **Lock: floors compare composite quality_score, not onset_f1.** The spec's verbal floor descriptions ("≥ 95", "≥ 80", etc.) translate directly.

3. **Re-baseline Stage 2 v2 with the same driver as Stage 3 v2.** Don't trust the spec's historical 96.8/93.4/83.1/75.2 — we can't confirm what metric/aggregation produced them. Run `evaluate_stage_b_checkpoint.py` against Stage 2 v2 `_best.pt` on the same manifests as Stage 3 v2; the resulting numbers are guaranteed comparable. **Lock: 6 eval invocations total (3 manifests × 2 checkpoints).** Removes the metric-mystery and the parallel-verification-track distinction simultaneously.

4. **Per-dataset eval covers 7 datasets via 3 manifests** (post-discovery — token_manifest_stage3.jsonl has only `_systems` versions; token_manifest_full.jsonl has the older single-staff variants; synthetic_systems has 0 test rows in either manifest):
   - **`token_manifest_stage3.jsonl --split test`** → `grandstaff_systems` (5432), `primus_systems` (8835), `cameraprimus_systems` (8835)
   - **`token_manifest_full.jsonl --split test`** → `grandstaff` (10638), `primus` (8835), `cameraprimus` (8835)
   - **Fresh synthetic eval manifest** → `synthetic_systems` (200 newly-generated samples; train manifest has no test rows for synthetic)

5. **Synthetic eval generates fresh samples via `scripts/build_synthetic_systems_v1.py`.** synthetic_systems was train-only in the manifest; spec's ≥ 90 floor needs a measurement surface. New script `scripts/generate_synthetic_eval_samples.py` wraps the existing builder with eval-specific output (different seed, 200 samples, output to a small jsonl manifest fragment that `evaluate_stage_b_checkpoint.py` consumes). **Lock: 200 fresh samples is enough for the regression check; the spec's floor at 90 is not so tight that 200 vs 1000 changes the verdict.**

6. **Beam width = 5, max_decode_steps = 512 for production lieder eval.** Greedy (the default) is for the Task 1 smoke test only. Stage 2 v2's lieder numbers were collected at beam=5; comparing v2 at greedy would be an apples-to-oranges read on the architectural gate. **Lock: smoke = greedy 1 piece; production = beam=5, max_decode_steps=512, all pieces.**

7. **Decision-gate evaluator is a single Python tool with TDD tests.** Inputs: paths to the lieder per-piece CSV, the per-manifest evaluate_stage_b_checkpoint.py JSON outputs (with per-dataset breakdowns), the spec's threshold table embedded as constants. Output: a markdown verdict report + exit code (0 = Ship/Mixed, 1 = Flat/Pivot/Diagnose). **Why a single tool:** the four decision branches in spec §"Decision flow" are not independent (regression-check FAIL short-circuits the lieder branch); centralizing the logic in one place keeps the verdict reproducible.

8. **Stratified-by-staves analysis is post-hoc on the per-piece CSV.** `eval/score_lieder_eval.py` writes `eval/results/lieder_<name>.csv` with per-piece rows. The stratified analyzer is a new tool that reads that CSV, joins on a `staves_in_system` column (which the lieder split already records), and emits per-staves-bucket onset_f1 means + the `lc6548281` sanity-check row. Keeping this out of `score_lieder_eval.py` preserves the existing driver's stability (it ran clean for Stage 2 v2; don't perturb it).

9. **v1 head-to-head is conditional, not scheduled.** Spec doesn't require it. The Phase 1 handoff says "evaluate v2 as primary; if v2 fails the onset_f1 gate, evaluate v1 as well." **Lock: only run v1 lieder eval if Task 7's verdict is Mixed or Flat; otherwise it's needless compute.** Captured as Task 9 (conditional).

10. **All eval inference runs on the GPU box; scoring + decision-gate run locally.** Mirrors Plan C's split (training on GPU, test verification local). Scoring uses subprocess-isolated music21 (per `score_lieder_eval.py:6` rationale — Stage 2 v2 hit 43 GB committed memory at piece 6/20 in-process). Local box doesn't need GPU for scoring + decision-gate.

11. **Strong threshold = mean lieder onset_f1 ≥ 0.30.** Mixed = 0.241 ≤ x < 0.30. Flat = < 0.241. Drawn directly from spec §"Phase 2 §1" line 235-237. Threshold values embedded as constants in `eval/decision_gate.py`; tests pin them down so a future spec revision is a single-file change.

12. **Per-dataset Stage 3 floors (revised — 7 entries):**
    - `synthetic_systems` ≥ 90 (spec; architectural-bet floor; measured against fresh synthetic samples)
    - `grandstaff_systems` ≥ 95 (spec; system-level; primary verdict)
    - `grandstaff` ≥ 90 (spec; single-staff confirming)
    - `primus_systems` ≥ 80 (spec analog of `primus`; system-level)
    - `primus` ≥ 80 (spec; single-staff)
    - `cameraprimus_systems` ≥ max(75, fresh_baseline_quality - 5) (spec analog; dynamic per the "raise floor" rule, using the Stage 2 v2 cameraprimus_systems re-baseline result)
    - `cameraprimus` ≥ max(75, fresh_baseline_quality - 5) (spec; same dynamic rule, using Stage 2 v2 cameraprimus single-staff re-baseline)

13. **MusicXML validity rate is a "corroborating signal" not a hard gate.** Spec line 266: "Useful corroborating signal even though not gated." The decision-gate report includes the number but doesn't fail on it. Rationale: MusicXML validity correlates with onset_f1; if onset_f1 passes, validity is structurally informative but not load-bearing.

14. **Final verdict is one of four** — Ship, Investigate, Pivot, Diagnose. Logic from spec §"Decision flow":
    - Any per-dataset floor FAIL → **Diagnose** (regardless of lieder)
    - All floors PASS + lieder Strong → **Ship**
    - All floors PASS + lieder Mixed → **Investigate**
    - All floors PASS + lieder Flat → **Pivot**

---

## Files to create or modify

**New files:**
- `scripts/generate_synthetic_eval_samples.py` — wraps `scripts/build_synthetic_systems_v1.py` to produce 200 fresh synthetic_systems samples with eval-specific output dir + a small token-manifest fragment (Task 2)
- `eval/stratified_lieder_analysis.py` — onset_f1 stratified by `staves_in_system` + `lc6548281` sanity check (Task 6)
- `eval/decision_gate.py` — aggregates all eval outputs into a Ship/Investigate/Pivot/Diagnose verdict (Task 7)
- `tests/eval/test_stratified_lieder_analysis.py` — TDD for the stratified analyzer (Task 6)
- `tests/eval/test_decision_gate.py` — TDD for the verdict logic (Task 7)
- `docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase2-complete-handoff.md` — final handoff (Task 8, date filled at completion)

**Modified files:** none — the existing eval drivers are used as-is.

**Generated artifacts (under `eval/results/` on the GPU box, mirrored locally):**
- `lieder_stage3_v2_smoke/` + `lieder_stage3_v2_smoke.csv` (Task 1; ✓ committed at `d438f5b`)
- `per_dataset_stage3_v2_systems.json` + `per_dataset_stage2_v2_systems.json` (Task 2: token_manifest_stage3.jsonl evals)
- `per_dataset_stage3_v2_singlestaff.json` + `per_dataset_stage2_v2_singlestaff.json` (Task 2: token_manifest_full.jsonl evals)
- `per_dataset_stage3_v2_synthetic.json` + `per_dataset_stage2_v2_synthetic.json` (Task 2: fresh-synthetic evals)
- `lieder_stage3_v2/` + `lieder_stage3_v2.csv` (Task 4, 5)
- `eval/results/decision_gate_stage3_v2.md` (Task 7 output)

---

## Phase 2 Exit Criteria

Phase 2 produces one of four verdicts; "exit criteria" = the verdict is decided with all required evidence:

1. **Lieder onset_f1 evaluated** with beam=5 against v2 `_best.pt@step5500`. Mean onset_f1 measured across all pieces in `data/openscore_lieder/scores/`. Stratified analysis produced.
2. **Per-dataset composite quality measured** for all 7 datasets on Stage 3 v2 _best.pt: synthetic_systems (fresh samples), grandstaff_systems / primus_systems / cameraprimus_systems (system-level test split), grandstaff / primus / cameraprimus (single-staff test split). Each compared to its floor.
3. **Stage 2 v2 fresh baseline produced** for the same 7 datasets. Replaces spec's historical 96.8/93.4/83.1/75.2 numbers — the metric/aggregation has not been confirmed reproducible from the historical artifacts alone.
4. **MusicXML validity rate aggregated** from the lieder eval (corroborating signal).
5. **Decision-gate report written** at `eval/results/decision_gate_stage3_v2.md` with explicit verdict and per-surface evidence.
6. **Final handoff doc written** at `docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase2-complete-handoff.md` summarizing the verdict and next steps (PR for Ship, follow-up plan for Investigate, pivot decision for Flat, diagnostic plan for Diagnose).

If any of 1-5 cannot be produced (e.g. eval driver crashes, scoring OOM), Phase 2 is *blocked*, not *failed* — return to user with diagnostics before declaring a verdict.

---

## Tasks

### Task 0: Branch + Phase 2 launch handoff doc skeleton

**Files:**
- Create: `docs/superpowers/handoffs/2026-05-09-radio-stage3-phase2-launch-handoff.md`

- [ ] **Step 1: Create branch off updated main**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git checkout main && git pull origin main
git checkout -b feat/stage3-phase2-evaluation
```

- [ ] **Step 2: Pull v2 best.pt to local mirror for inspection (no edit)**

The artifact stays on the GPU box for inference. Mirror locally only if needed for spot-check.

```bash
ssh 10.10.1.29 'dir Clarity-OMR-Train-RADIO\checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt'
```

Expected output: file listing showing the file exists (~3.21 GB).

- [ ] **Step 3: Write launch-handoff doc**

```markdown
# RADIO Stage 3 Phase 2 — Launch Handoff (2026-05-09)

## TL;DR
Phase 1 produced ship artifact `_best.pt@step5500` (val_loss 0.164, cameraprimus
val_loss 0.136 below SV2 anchor). Phase 2 runs three eval surfaces and a
decision gate to decide: Ship / Investigate / Pivot / Diagnose.

## Inputs
- v2 ship artifact: `10.10.1.29:checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt`
- v2 step-log: `logs/full_radio_stage3_v2_steps.jsonl`
- Stage 3 trainer config: `configs/train_stage3_radio_systems.yaml`
- Lieder reference: `data/openscore_lieder/scores/`
- Per-dataset reference: `data/clarity_demo/mxl/` (grandstaff, primus, cameraprimus subsets)

## Plan
`docs/superpowers/plans/2026-05-09-radio-stage3-phase2-evaluation.md`

## Compute budget
- Smoke test: ~1 min (1 piece greedy)
- Lieder full eval (beam=5): ~1-2 h
- Per-dataset quality (5 datasets): ~30-60 min
- Cameraprimus 200-sample baseline (Stage 2 v2): ~30 min (parallel)
- Total wall: ~2-3 h (some parallelism on multi-GPU box)
```

- [ ] **Step 4: Commit + push**

```bash
git add docs/superpowers/handoffs/2026-05-09-radio-stage3-phase2-launch-handoff.md
git commit -m "docs(handoff): Phase 2 launch handoff"
git push -u origin feat/stage3-phase2-evaluation
```

---

### Task 1: Smoke test — single-piece lieder eval against v2 best.pt

**Goal:** confirm the inference + scoring pipeline works end-to-end against the Stage 3 v2 checkpoint and produces a non-NaN onset_f1 number for one piece. If this passes, the full eval (Task 4) is mechanical.

**Files:**
- No code changes. Operational task.
- Generated: `eval/results/lieder_stage3_v2_smoke/` + `eval/results/lieder_stage3_v2_smoke.csv`

- [ ] **Step 1: SSH to GPU box and pull latest main**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && git fetch origin && git checkout main && git pull origin main'
```

- [ ] **Step 2: Run lieder inference on 1 piece (greedy, smoke)**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m eval.run_lieder_eval --checkpoint checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt --config configs\train_stage3_radio_systems.yaml --name stage3_v2_smoke --max-pieces 1 --beam-width 1 --max-decode-steps 256'
```

Expected: a `eval/results/lieder_stage3_v2_smoke/<piece>.musicxml` file written. No exceptions. Inference status JSONL row says `success`.

- [ ] **Step 3: Score the 1-piece run**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m eval.score_lieder_eval --predictions-dir eval\results\lieder_stage3_v2_smoke --reference-dir data\openscore_lieder\scores --name stage3_v2_smoke --cheap-jobs 2 --tedn-jobs 1'
```

Expected: `eval/results/lieder_stage3_v2_smoke.csv` written with one row, `onset_f1` column populated with a float (any value, even 0.0, is success — NaN/missing is failure).

- [ ] **Step 4: Verify the smoke output**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -c "import csv; f=open(\"eval/results/lieder_stage3_v2_smoke.csv\"); r=list(csv.DictReader(f)); print(\"rows:\",len(r),\"first onset_f1:\",r[0].get(\"onset_f1\"))"'
```

Expected output: `rows: 1 first onset_f1: <float string>` (e.g. `0.234` or `0.401`). If `None` or missing column → STOP, debug before Task 4.

- [ ] **Step 5: Pull smoke CSV locally for record-keeping**

```bash
scp 10.10.1.29:Clarity-OMR-Train-RADIO/eval/results/lieder_stage3_v2_smoke.csv eval/results/
git add eval/results/lieder_stage3_v2_smoke.csv
git commit -m "feat(eval): Phase 2 smoke test — 1 piece v2 lieder eval ($(date +%Y-%m-%d))"
```

---

### Task 2: Per-dataset quality master eval (REVISED 2026-05-09)

**Goal:** produce per-dataset composite quality numbers for all 7 datasets on both Stage 3 v2 `_best.pt@step5500` AND Stage 2 v2 `_best.pt`, using `src/eval/evaluate_stage_b_checkpoint.py` (the same driver Stage 2 v2 used). Six eval invocations across 3 manifests × 2 checkpoints, plus a fresh-synthetic generation step.

**Why this replaces the original Tasks 2 + 3:** the original draft assumed `eval/run_clarity_demo_eval.py` would handle all 5 datasets. Discovery (2026-05-09) showed it's hardcoded to 4 HF demo pieces and has no per-dataset filtering. The actual Stage 2 v2 pipeline was `evaluate_stage_b_checkpoint.py`, which reads a token manifest, runs Stage B inference, calls `run_eval.evaluate_rows`, and auto-groups by `dataset` field — producing per-dataset composite quality scores in one pass.

**Files:**
- Create: `scripts/generate_synthetic_eval_samples.py` (Step 1)
- Operational: 6 invocations of `evaluate_stage_b_checkpoint.py`
- Generated artifacts:
  - `eval/results/per_dataset_stage3_v2_systems.json` (Stage 3 v2 on token_manifest_stage3.jsonl)
  - `eval/results/per_dataset_stage2_v2_systems.json` (Stage 2 v2 on token_manifest_stage3.jsonl)
  - `eval/results/per_dataset_stage3_v2_singlestaff.json` (Stage 3 v2 on token_manifest_full.jsonl)
  - `eval/results/per_dataset_stage2_v2_singlestaff.json` (Stage 2 v2 on token_manifest_full.jsonl)
  - `src/data/manifests/synthetic_systems_eval_fresh.jsonl` (fresh synthetic eval manifest fragment)
  - `eval/results/per_dataset_stage3_v2_synthetic.json`
  - `eval/results/per_dataset_stage2_v2_synthetic.json`

#### Sub-task 2A: Verify checkpoints + manifests on GPU box

- [ ] **Step 1: Identify the Stage 2 v2 best.pt path**

```bash
ssh 10.10.1.29 'dir Clarity-OMR-Train-RADIO\checkpoints\full_radio_stage2_systems_v2\*.pt'
```

Per discovery: actual filename is `stage2-radio-systems-polyphonic_best.pt` (memory said `stage2-radio-systems_best.pt`; memory drift). Use whichever filename the listing shows that ends in `_best.pt`.

- [ ] **Step 2: Verify both manifests have test splits as expected**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -c "
import json
from collections import Counter
for path in [\"src/data/manifests/token_manifest_stage3.jsonl\", \"src/data/manifests/token_manifest_full.jsonl\"]:
    c = Counter()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get(\"split\") == \"test\":
                c[r.get(\"dataset\",\"?\")] += 1
    print(path); print(\"  test rows by dataset:\", dict(c))
"'
```

Expected output:
```
src/data/manifests/token_manifest_stage3.jsonl
  test rows by dataset: {'cameraprimus_systems': 8835, 'grandstaff_systems': 5432, 'primus_systems': 8835}
src/data/manifests/token_manifest_full.jsonl
  test rows by dataset: {'cameraprimus': 8835, 'grandstaff': 10638, 'primus': 8835}
```

If the counts don't match: STOP and report — the manifest layout has shifted.

#### Sub-task 2B: Generate fresh synthetic eval samples

- [ ] **Step 3: Inspect `scripts/build_synthetic_systems_v1.py` to understand its CLI**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe scripts\build_synthetic_systems_v1.py --help' 2>&1 | head -40
```

Read its arguments. Key fields needed: output directory (where rendered images + ground-truth go), seed, count.

- [ ] **Step 4: Write `scripts/generate_synthetic_eval_samples.py`**

This script wraps `build_synthetic_systems_v1.py` to produce 200 fresh synthetic_systems samples for eval. Output: a small jsonl manifest fragment in the format `evaluate_stage_b_checkpoint.py` consumes (one row per sample with `dataset`, `split`, `image_path`, `tokens`, etc.).

The exact field schema MUST match what `evaluate_stage_b_checkpoint.py` reads. Inspect 1-2 rows of `token_manifest_stage3.jsonl` first to confirm field names:

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -c "
import json
with open(\"src/data/manifests/token_manifest_stage3.jsonl\") as f:
    for i, line in enumerate(f):
        if i >= 2: break
        r = json.loads(line)
        print(json.dumps(r, indent=2)[:400])
        print(\"---\")
"'
```

Then write the wrapper. Pseudo-skeleton (adapt to whatever `build_synthetic_systems_v1.py` actually outputs):

```python
"""Generate 200 fresh synthetic_systems samples for Phase 2 eval.

Wraps scripts/build_synthetic_systems_v1.py to produce eval-isolated samples
(different seed than training; output directory under data/eval/synthetic_fresh).
Emits a manifest fragment that evaluate_stage_b_checkpoint.py consumes.
"""
import argparse, json, subprocess
from pathlib import Path

DEFAULT_SEED = 0xEEAA0000  # distinct from training-time seed
DEFAULT_COUNT = 200
DEFAULT_OUTPUT_DIR = Path("data/eval/synthetic_fresh")
DEFAULT_MANIFEST = Path("src/data/manifests/synthetic_systems_eval_fresh.jsonl")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--count", type=int, default=DEFAULT_COUNT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = p.parse_args()

    # 1. Run the existing builder with eval-isolated output dir + seed.
    #    The exact CLI signature depends on what build_synthetic_systems_v1.py
    #    accepts (Step 3 above).
    # 2. After the builder writes its sample files, build a manifest fragment:
    #    one jsonl row per sample with dataset="synthetic_systems",
    #    split="test", and whatever path/tokens fields evaluate_stage_b_checkpoint.py reads.
    # 3. Write the manifest fragment to args.manifest.
    ...

if __name__ == "__main__":
    raise SystemExit(main())
```

If `build_synthetic_systems_v1.py` doesn't accept a clean seed/count/output-dir interface, treat this as a real implementation task — patch the builder to accept them, OR write a minimal renderer based on the same primitives the builder uses. **Do not generate samples that overlap with training data** (use a high seed value distinct from training-time seeds).

- [ ] **Step 5: Generate the 200 fresh samples + manifest fragment**

```bash
scp scripts/generate_synthetic_eval_samples.py '10.10.1.29:Clarity-OMR-Train-RADIO/scripts/'
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe scripts\generate_synthetic_eval_samples.py'
```

Expected: `data/eval/synthetic_fresh/` populated with 200 rendered samples; `src/data/manifests/synthetic_systems_eval_fresh.jsonl` has 200 rows, each with `dataset="synthetic_systems"`, `split="test"`.

#### Sub-task 2C: Run all 6 evals

- [ ] **Step 6: Stage 3 v2 on token_manifest_stage3.jsonl (system-level)**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m src.eval.evaluate_stage_b_checkpoint --checkpoint checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt --token-manifest src/data/manifests/token_manifest_stage3.jsonl --split test --beam-width 5 --max-decode-steps 512 --output-summary eval/results/per_dataset_stage3_v2_systems.json'
```

Expected: ~30-60 min wall. Output JSON has `{overall, by_dataset: {grandstaff_systems: {...}, primus_systems: {...}, cameraprimus_systems: {...}}, musicxml_validity_rate}`.

- [ ] **Step 7: Stage 2 v2 on token_manifest_stage3.jsonl (system-level baseline)**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m src.eval.evaluate_stage_b_checkpoint --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --token-manifest src/data/manifests/token_manifest_stage3.jsonl --split test --beam-width 5 --max-decode-steps 512 --output-summary eval/results/per_dataset_stage2_v2_systems.json'
```

(Note: substitute the actual `_best.pt` filename from Step 1 if different.)

- [ ] **Step 8: Stage 3 v2 on token_manifest_full.jsonl (single-staff)**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m src.eval.evaluate_stage_b_checkpoint --checkpoint checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt --token-manifest src/data/manifests/token_manifest_full.jsonl --split test --beam-width 5 --max-decode-steps 512 --output-summary eval/results/per_dataset_stage3_v2_singlestaff.json'
```

- [ ] **Step 9: Stage 2 v2 on token_manifest_full.jsonl (single-staff baseline)**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m src.eval.evaluate_stage_b_checkpoint --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --token-manifest src/data/manifests/token_manifest_full.jsonl --split test --beam-width 5 --max-decode-steps 512 --output-summary eval/results/per_dataset_stage2_v2_singlestaff.json'
```

- [ ] **Step 10: Stage 3 v2 on synthetic eval manifest**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m src.eval.evaluate_stage_b_checkpoint --checkpoint checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt --token-manifest src/data/manifests/synthetic_systems_eval_fresh.jsonl --split test --beam-width 5 --max-decode-steps 512 --output-summary eval/results/per_dataset_stage3_v2_synthetic.json'
```

- [ ] **Step 11: Stage 2 v2 on synthetic eval manifest (baseline)**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m src.eval.evaluate_stage_b_checkpoint --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --token-manifest src/data/manifests/synthetic_systems_eval_fresh.jsonl --split test --beam-width 5 --max-decode-steps 512 --output-summary eval/results/per_dataset_stage2_v2_synthetic.json'
```

#### Sub-task 2D: Pull results locally + commit

- [ ] **Step 12: scp all 6 JSON outputs locally**

```bash
for f in per_dataset_stage3_v2_systems per_dataset_stage2_v2_systems \
         per_dataset_stage3_v2_singlestaff per_dataset_stage2_v2_singlestaff \
         per_dataset_stage3_v2_synthetic per_dataset_stage2_v2_synthetic; do
  scp 10.10.1.29:Clarity-OMR-Train-RADIO/eval/results/${f}.json eval/results/
done
```

- [ ] **Step 13: Print a comparison table for sanity check**

```bash
python3 -c "
import json, glob
results = {}
for path in glob.glob('eval/results/per_dataset_*.json'):
    with open(path) as f:
        d = json.load(f)
    name = path.split('/')[-1].replace('per_dataset_', '').replace('.json', '')
    results[name] = d.get('by_dataset', {})

print('| Dataset | Stage 2 v2 | Stage 3 v2 |')
print('|---|---|---|')
all_datasets = set()
for r in results.values():
    all_datasets.update(r.keys())
for ds in sorted(all_datasets):
    s2_q = next((r.get(ds, {}).get('quality_score') for n,r in results.items() if 'stage2' in n and ds in r), None)
    s3_q = next((r.get(ds, {}).get('quality_score') for n,r in results.items() if 'stage3' in n and ds in r), None)
    print(f'| {ds} | {s2_q if s2_q is not None else \"-\":.2f} | {s3_q if s3_q is not None else \"-\":.2f} |')
"
```

Expected output: a 7-row table comparing Stage 2 v2 vs Stage 3 v2 composite quality_score per dataset. **The exact JSON key for the metric** (`quality_score` vs `quality` vs `composite`) depends on what `evaluate_stage_b_checkpoint.py` writes — adjust the print statement if the key is different.

- [ ] **Step 14: Commit + push**

```bash
git add scripts/generate_synthetic_eval_samples.py eval/results/per_dataset_*.json
git commit -m "feat(eval): Phase 2 per-dataset quality matrix (Stage 2 v2 + Stage 3 v2 × 3 manifests)"
git push origin feat/stage3-phase2-evaluation
```

---

### Task 4: Full lieder onset_f1 eval against v2 best.pt (production)

**Goal:** run inference on all pieces in `data/openscore_lieder/scores/` with beam=5, max_decode_steps=512.

**Files:**
- No code changes. Operational task.
- Generated: `eval/results/lieder_stage3_v2/` (per-piece predictions + status JSONL)

- [ ] **Step 1: Run lieder inference, all pieces, beam=5**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m eval.run_lieder_eval --checkpoint checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt --config configs\train_stage3_radio_systems.yaml --name stage3_v2 --beam-width 5 --max-decode-steps 512'
```

Expected: ~1-2 h wall (depends on lieder corpus size and beam timing). Per-piece status rows in `eval/results/lieder_stage3_v2_inference_status.jsonl`. Stage-D diagnostics sidecars under `eval/results/lieder_stage3_v2/` for any pieces that hit edge cases.

- [ ] **Step 2: Verify all pieces succeeded (or document failures)**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -c "
import json
with open(\"eval/results/lieder_stage3_v2_inference_status.jsonl\") as f:
    rows=[json.loads(line) for line in f]
n_ok=sum(1 for r in rows if r.get(\"status\")==\"success\")
print(f\"total={len(rows)} success={n_ok} failed={len(rows)-n_ok}\")
for r in rows:
    if r.get(\"status\")!=\"success\":
        print(\"FAILURE:\", r.get(\"piece\"), r.get(\"error\"))
"'
```

Expected output: `total=N success=N failed=0` (or document failures + their causes; small N of timeouts are tolerable but should be in the handoff).

- [ ] **Step 3: Commit (skip pulling — full set is too big to put in git; reference by path)**

No commit at this step — predictions stay on GPU box. The CSV from Task 5 is what gets committed.

---

### Task 5: Score the lieder eval

**Files:**
- No code changes. Operational task.
- Generated: `eval/results/lieder_stage3_v2.csv`

- [ ] **Step 1: Score**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m eval.score_lieder_eval --predictions-dir eval\results\lieder_stage3_v2 --reference-dir data\openscore_lieder\scores --name stage3_v2 --cheap-jobs 8 --tedn-jobs 4 --max-active-pieces 8'
```

Expected: `eval/results/lieder_stage3_v2.csv` with per-piece rows. Wall: ~30-60 min (TEDN is slow per piece).

- [ ] **Step 2: Compute mean onset_f1 + MusicXML validity rate**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -c "
import csv, statistics
f=open(\"eval/results/lieder_stage3_v2.csv\")
rows=list(csv.DictReader(f))
onset_vals=[float(r[\"onset_f1\"]) for r in rows if r.get(\"onset_f1\") not in (None,\"\")]
mxl_valid=sum(1 for r in rows if r.get(\"musicxml_valid\") in (\"true\",\"True\",\"1\",1))
print(f\"n={len(rows)} mean_onset_f1={statistics.mean(onset_vals):.4f} musicxml_validity_rate={mxl_valid/len(rows):.3f}\")
"'
```

Expected output: `n=N mean_onset_f1=0.XXXX musicxml_validity_rate=0.YYY`. **The mean_onset_f1 number is the architectural ship gate.** Record it.

- [ ] **Step 3: Pull CSV locally and commit**

```bash
scp 10.10.1.29:Clarity-OMR-Train-RADIO/eval/results/lieder_stage3_v2.csv eval/results/
git add eval/results/lieder_stage3_v2.csv
git commit -m "feat(eval): Phase 2 full lieder eval on v2 best.pt ($(date +%Y-%m-%d))"
```

---

### Task 6: Stratified-onset_f1 analyzer (TDD)

**Goal:** read `eval/results/lieder_stage3_v2.csv`, group rows by `staves_in_system` (the lieder split's stratification dimension; spec line 240), emit per-bucket onset_f1 means, and run the explicit `lc6548281` sanity check (≥ 0.10 vs DaViT's 0.05; spec line 241).

**Files:**
- Create: `eval/stratified_lieder_analysis.py`
- Create: `tests/eval/test_stratified_lieder_analysis.py`

- [ ] **Step 1: Verify the lieder CSV has a staves_in_system column**

```bash
head -2 eval/results/lieder_stage3_v2.csv 2>/dev/null
```

Expected: column header includes `staves_in_system` (or equivalent — `n_staves`, `staves_per_system`, etc.). If not present in the CSV, look at `eval/lieder_split.py` for the canonical name and grep the score driver to add it. Do NOT proceed to Step 2 until you know the column name.

- [ ] **Step 2: Write the failing test**

```python
# tests/eval/test_stratified_lieder_analysis.py
"""Stratified onset_f1 analysis: group lieder eval CSV by staves_in_system,
emit per-bucket means, run lc6548281 sanity check.
"""
from __future__ import annotations
import csv
from pathlib import Path
import pytest


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_stratified_groups_by_staves_in_system(tmp_path):
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "p1", "staves_in_system": "1", "onset_f1": "0.5"},
        {"piece": "p2", "staves_in_system": "1", "onset_f1": "0.6"},
        {"piece": "p3", "staves_in_system": "2", "onset_f1": "0.3"},
        {"piece": "p4", "staves_in_system": "2", "onset_f1": "0.4"},
        {"piece": "p5", "staves_in_system": "3", "onset_f1": "0.2"},
    ])

    result = analyze(csv_path)

    assert result.per_bucket == {
        "1": pytest.approx(0.55),
        "2": pytest.approx(0.35),
        "3": pytest.approx(0.20),
    }
    assert result.bucket_counts == {"1": 2, "2": 2, "3": 1}
    assert result.overall_mean == pytest.approx(0.4)


def test_stratified_lc6548281_sanity_check(tmp_path):
    """The architectural sanity check from spec line 241: lc6548281 should
    improve from DaViT's 0.05 to >= 0.10."""
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "lc6548281", "staves_in_system": "2", "onset_f1": "0.18"},
        {"piece": "other", "staves_in_system": "1", "onset_f1": "0.5"},
    ])

    result = analyze(csv_path)

    assert result.lc6548281_onset_f1 == pytest.approx(0.18)
    assert result.lc6548281_passes_sanity is True


def test_stratified_lc6548281_fails_sanity_when_below_threshold(tmp_path):
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "lc6548281", "staves_in_system": "2", "onset_f1": "0.08"},
    ])

    result = analyze(csv_path)

    assert result.lc6548281_onset_f1 == pytest.approx(0.08)
    assert result.lc6548281_passes_sanity is False


def test_stratified_lc6548281_missing_returns_none(tmp_path):
    """If lc6548281 isn't in the eval set (e.g. a smoke run), the field is
    None, not raising — the decision-gate consumer treats this as 'not
    evaluated' rather than a failure."""
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "p1", "staves_in_system": "1", "onset_f1": "0.5"},
    ])

    result = analyze(csv_path)

    assert result.lc6548281_onset_f1 is None
    assert result.lc6548281_passes_sanity is None


def test_stratified_skips_rows_with_missing_onset_f1(tmp_path):
    """Pieces that errored during scoring have onset_f1 empty/None.
    They're excluded from the buckets so the means aren't poisoned."""
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "p1", "staves_in_system": "1", "onset_f1": "0.5"},
        {"piece": "p2", "staves_in_system": "1", "onset_f1": ""},
        {"piece": "p3", "staves_in_system": "1", "onset_f1": "0.7"},
    ])

    result = analyze(csv_path)

    assert result.per_bucket["1"] == pytest.approx(0.6)
    assert result.bucket_counts["1"] == 2
```

- [ ] **Step 3: Run tests to verify RED**

```bash
python3 -m pytest tests/eval/test_stratified_lieder_analysis.py -v
```

Expected: 5 tests, all fail with `ModuleNotFoundError: No module named 'eval.stratified_lieder_analysis'`.

- [ ] **Step 4: Implement the analyzer**

```python
# eval/stratified_lieder_analysis.py
"""Stratified onset_f1 analysis on a lieder eval CSV.

Reads the per-piece CSV produced by eval.score_lieder_eval and emits:
- Per-bucket mean onset_f1 grouped by staves_in_system
- The lc6548281 sanity check (architectural sanity from spec §"Phase 2 §1"
  line 241: should improve from DaViT's 0.05 to >= 0.10)
- Overall mean onset_f1 across all valid rows

Used by eval.decision_gate to attach the stratified breakdown to the
verdict report. The threshold for lc6548281 is the spec's, locked here
as a constant; a future spec revision is a single-line change.
"""
from __future__ import annotations
import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# Architectural sanity check (spec §"Phase 2 §1" line 241): lc6548281 should
# improve from DaViT's 0.05 baseline to >= 0.10 on Stage 3 v2.
LC6548281_SANITY_THRESHOLD = 0.10
SANITY_PIECE_ID = "lc6548281"


@dataclass(frozen=True)
class StratifiedResult:
    per_bucket: Dict[str, float]
    bucket_counts: Dict[str, int]
    overall_mean: float
    lc6548281_onset_f1: Optional[float]
    lc6548281_passes_sanity: Optional[bool]


def analyze(csv_path: Path) -> StratifiedResult:
    """Read a lieder eval CSV and return the stratified breakdown.

    Rows with missing/empty onset_f1 are excluded (scoring errors). The
    staves_in_system column is read as a string key to preserve any 1/2/3+
    or 'multi' semantics the lieder split chooses to encode.
    """
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            f1 = r.get("onset_f1")
            if f1 in (None, ""):
                continue
            try:
                f1_val = float(f1)
            except (TypeError, ValueError):
                continue
            rows.append({"piece": r.get("piece", ""), "staves_in_system": str(r.get("staves_in_system", "")), "onset_f1": f1_val})

    by_bucket: Dict[str, list] = {}
    all_vals: list = []
    lc_val: Optional[float] = None
    for r in rows:
        by_bucket.setdefault(r["staves_in_system"], []).append(r["onset_f1"])
        all_vals.append(r["onset_f1"])
        if r["piece"] == SANITY_PIECE_ID:
            lc_val = r["onset_f1"]

    per_bucket = {k: statistics.mean(v) for k, v in by_bucket.items()}
    counts = {k: len(v) for k, v in by_bucket.items()}
    overall = statistics.mean(all_vals) if all_vals else 0.0
    sanity = (lc_val >= LC6548281_SANITY_THRESHOLD) if lc_val is not None else None

    return StratifiedResult(
        per_bucket=per_bucket,
        bucket_counts=counts,
        overall_mean=overall,
        lc6548281_onset_f1=lc_val,
        lc6548281_passes_sanity=sanity,
    )


def _format_report(result: StratifiedResult) -> str:
    lines = ["# Stratified lieder onset_f1 analysis", ""]
    lines.append(f"Overall mean onset_f1 (n={sum(result.bucket_counts.values())}): **{result.overall_mean:.4f}**")
    lines.append("")
    lines.append("## By staves_in_system")
    lines.append("")
    lines.append("| Bucket | n | mean onset_f1 |")
    lines.append("|---|---|---|")
    for k in sorted(result.per_bucket.keys()):
        lines.append(f"| {k} | {result.bucket_counts[k]} | {result.per_bucket[k]:.4f} |")
    lines.append("")
    lines.append("## lc6548281 sanity check (spec §Phase 2 §1)")
    lines.append("")
    if result.lc6548281_onset_f1 is None:
        lines.append("- **NOT EVALUATED** (lc6548281 not present in eval set)")
    else:
        verdict = "PASS" if result.lc6548281_passes_sanity else "FAIL"
        lines.append(f"- onset_f1 = {result.lc6548281_onset_f1:.4f}")
        lines.append(f"- threshold = {LC6548281_SANITY_THRESHOLD:.2f} (DaViT baseline: 0.05)")
        lines.append(f"- **{verdict}**")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Stratified onset_f1 analysis on a lieder eval CSV.")
    p.add_argument("--csv", type=Path, required=True, help="Path to eval/results/lieder_<name>.csv")
    p.add_argument("--output", type=Path, default=None, help="Optional path to write the markdown report")
    args = p.parse_args()

    result = analyze(args.csv)
    report = _format_report(result)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify GREEN**

```bash
python3 -m pytest tests/eval/test_stratified_lieder_analysis.py -v
```

Expected: 5 tests, all PASS.

- [ ] **Step 6: Run on the real lieder CSV from Task 5**

```bash
python3 -m eval.stratified_lieder_analysis --csv eval/results/lieder_stage3_v2.csv
```

Expected: markdown report with overall mean, per-bucket table, lc6548281 PASS/FAIL/NOT EVALUATED. Sanity-check by eye that the buckets are populated.

- [ ] **Step 7: Commit**

```bash
git add eval/stratified_lieder_analysis.py tests/eval/test_stratified_lieder_analysis.py
git commit -m "feat(eval): stratified-onset_f1 analyzer + lc6548281 sanity check"
```

---

### Task 7: Decision-gate evaluator (TDD) (REVISED 2026-05-09)

**Goal:** aggregate the lieder onset_f1 result, the 7 per-dataset quality results (from Task 2's per_dataset_*.json artifacts), and the MusicXML validity rate into one of four verdicts (Ship / Investigate / Pivot / Diagnose) per spec §"Decision flow".

**Per-dataset floors handled (revised 2026-05-09 — 7 entries):**
- 5 static: synthetic_systems ≥ 90, grandstaff_systems ≥ 95, grandstaff ≥ 90, primus_systems ≥ 80, primus ≥ 80
- 2 dynamic (cameraprimus + cameraprimus_systems): each `≥ max(75, fresh_baseline_quality - 5)` using its own Stage 2 v2 re-baseline number from Task 2 (the cameraprimus_systems baseline comes from token_manifest_stage3.jsonl eval; the cameraprimus baseline from token_manifest_full.jsonl eval — they're different numbers).

**Files:**
- Create: `eval/decision_gate.py`
- Create: `tests/eval/test_decision_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_decision_gate.py
"""Decision gate: aggregates lieder + per-dataset + validity into a verdict."""
from __future__ import annotations
import pytest

from dataclasses import asdict


def _mk_inputs(**overrides):
    """Build a complete, all-passing input dict; override specific fields per test."""
    base = {
        "lieder_mean_onset_f1": 0.32,
        "musicxml_validity_rate": 0.95,
        "per_dataset": {
            "synthetic_systems": 92.0,
            "grandstaff_systems": 96.0,
            "grandstaff": 92.0,
            "primus_systems": 82.0,
            "primus": 82.0,
            "cameraprimus_systems": 76.0,
            "cameraprimus": 76.0,
        },
        "cameraprimus_systems_baseline": 75.2,
        "cameraprimus_baseline": 75.2,
        "lc6548281_onset_f1": 0.15,
    }
    base.update(overrides)
    return base


def test_all_pass_strong_lieder_yields_ship():
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs())

    assert result.verdict == Verdict.SHIP
    assert result.lieder_outcome == "Strong"
    assert all(v.passed for v in result.per_dataset_results.values())


def test_all_floors_pass_mixed_lieder_yields_investigate():
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.27))

    assert result.verdict == Verdict.INVESTIGATE
    assert result.lieder_outcome == "Mixed"


def test_all_floors_pass_flat_lieder_yields_pivot():
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.20))

    assert result.verdict == Verdict.PIVOT
    assert result.lieder_outcome == "Flat"


def test_lieder_at_strong_threshold_is_strong():
    """Boundary check: 0.30 exactly is Strong (>= 0.30 per spec line 235)."""
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.30))

    assert result.lieder_outcome == "Strong"
    assert result.verdict == Verdict.SHIP


def test_lieder_at_mixed_threshold_is_mixed():
    """Boundary: 0.241 is Mixed (>= 0.241 per spec line 236)."""
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.2410))

    assert result.lieder_outcome == "Mixed"


def test_lieder_below_mixed_is_flat():
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.2409))

    assert result.lieder_outcome == "Flat"


def test_per_dataset_floor_fail_yields_diagnose_regardless_of_lieder():
    """A regression on grandstaff_systems means Stage 3 broke something
    Stage 2 v2 had. Don't draw lieder conclusions over a broken baseline.
    Spec §"Decision flow" line 285."""
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(
        lieder_mean_onset_f1=0.40,                          # Strong, but...
        per_dataset={
            "synthetic_systems": 92.0,
            "grandstaff_systems": 80.0,                     # FAILS floor 95
            "grandstaff": 92.0,
            "primus_systems": 82.0,
            "primus": 82.0,
            "cameraprimus_systems": 76.0,
            "cameraprimus": 76.0,
        },
    ))

    assert result.verdict == Verdict.DIAGNOSE
    assert result.per_dataset_results["grandstaff_systems"].passed is False
    assert result.per_dataset_results["grandstaff_systems"].floor == 95.0


def test_cameraprimus_floor_lifts_when_baseline_re_eval_higher():
    """Spec §"Phase 2 §2" line 261: 'cameraprimus ≥ 75 if 200-sample eval
    confirms baseline at 75.2; raise floor accordingly if eval shows higher.'
    Implementation: floor = max(75, baseline - 5). Same dynamic for cameraprimus_systems."""
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(
        per_dataset={
            "synthetic_systems": 92.0,
            "grandstaff_systems": 96.0,
            "grandstaff": 92.0,
            "primus_systems": 82.0,
            "primus": 82.0,
            "cameraprimus_systems": 79.0,
            "cameraprimus": 79.0,
        },
        cameraprimus_systems_baseline=85.0,
        cameraprimus_baseline=85.0,
    ))

    # Floor = max(75, 85.0 - 5) = 80; both cameraprimus variants at 79 fail.
    assert result.per_dataset_results["cameraprimus"].floor == 80.0
    assert result.per_dataset_results["cameraprimus"].passed is False
    assert result.per_dataset_results["cameraprimus_systems"].floor == 80.0
    assert result.per_dataset_results["cameraprimus_systems"].passed is False


def test_cameraprimus_floor_holds_at_75_when_baseline_at_or_below_baseline():
    """When the re-eval comes in at or below 80 (75 + 5), floor stays at 75."""
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(
        cameraprimus_systems_baseline=78.0,
        cameraprimus_baseline=78.0,
    ))

    assert result.per_dataset_results["cameraprimus"].floor == 75.0
    assert result.per_dataset_results["cameraprimus_systems"].floor == 75.0


def test_cameraprimus_variants_can_have_different_dynamic_floors():
    """The two cameraprimus variants are evaluated on different manifests
    (token_manifest_full.jsonl single-staff vs token_manifest_stage3.jsonl
    _systems). Stage 2 v2 baselines may differ — each variant's floor uses
    its own baseline."""
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(
        cameraprimus_systems_baseline=82.0,   # higher → floor 77
        cameraprimus_baseline=78.0,           # lower → floor 75
    ))

    assert result.per_dataset_results["cameraprimus_systems"].floor == 77.0
    assert result.per_dataset_results["cameraprimus"].floor == 75.0


def test_musicxml_validity_is_recorded_but_not_gated():
    """Spec line 266: corroborating signal, not gated. A low validity rate
    annotates the report but doesn't change the verdict."""
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(musicxml_validity_rate=0.20))  # very low

    assert result.musicxml_validity_rate == 0.20
    # All other checks pass + lieder Strong → Ship, despite low validity.
    assert result.verdict == Verdict.SHIP


def test_render_report_contains_verdict_and_evidence():
    from eval.decision_gate import evaluate, render_report

    result = evaluate(**_mk_inputs())
    report = render_report(result)

    assert "SHIP" in report
    assert "0.32" in report or "0.3200" in report
    assert "synthetic_systems" in report
    assert "grandstaff_systems" in report
```

- [ ] **Step 2: Run tests to verify RED**

```bash
python3 -m pytest tests/eval/test_decision_gate.py -v
```

Expected: 12 tests, all fail with `ModuleNotFoundError: No module named 'eval.decision_gate'`.

- [ ] **Step 3: Implement the gate**

```python
# eval/decision_gate.py
"""Phase 2 decision gate for RADIO Stage 3.

Aggregates the three eval surfaces from spec §"Phase 2" — lieder onset_f1,
per-dataset quality regression-check, MusicXML validity rate (corroborating)
— plus the cameraprimus baseline re-eval, into one of four verdicts:

    Ship       — all per-dataset floors PASS + lieder Strong (≥ 0.30)
    Investigate — all floors PASS + lieder Mixed (0.241 ≤ x < 0.30)
    Pivot      — all floors PASS + lieder Flat (< 0.241)
    Diagnose   — any floor FAILS (regardless of lieder)

Thresholds are spec constants (locked here; a spec revision is a one-file
edit). Inputs come from Tasks 1-6 outputs; the script aggregates and writes
a markdown report at eval/results/decision_gate_<name>.md.
"""
from __future__ import annotations
import argparse
import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# Spec §"Phase 2 §1" line 235-237.
STRONG_THRESHOLD = 0.30
MIXED_THRESHOLD = 0.2410

# Spec §"Phase 2 §2" line 254-262. Per-dataset Stage 3 quality floors (revised
# 2026-05-09 — 7 entries via _systems analogs from token_manifest_stage3.jsonl
# + single-staff originals from token_manifest_full.jsonl).
PER_DATASET_FLOORS = {
    "synthetic_systems": 90.0,
    "grandstaff_systems": 95.0,
    "grandstaff": 90.0,
    "primus_systems": 80.0,
    "primus": 80.0,
    # cameraprimus + cameraprimus_systems are dynamic — see _resolve_cameraprimus_floor().
}
CAMERAPRIMUS_FLOOR_BASE = 75.0
CAMERAPRIMUS_REGRESSION_TOLERANCE = 5.0


class Verdict(str, enum.Enum):
    SHIP = "Ship"
    INVESTIGATE = "Investigate"
    PIVOT = "Pivot"
    DIAGNOSE = "Diagnose"


@dataclass(frozen=True)
class FloorResult:
    dataset: str
    measured: float
    floor: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    verdict: Verdict
    lieder_mean_onset_f1: float
    lieder_outcome: str  # "Strong" | "Mixed" | "Flat"
    musicxml_validity_rate: float
    per_dataset_results: Dict[str, FloorResult]
    cameraprimus_systems_baseline: float
    cameraprimus_baseline: float
    lc6548281_onset_f1: Optional[float]


def _resolve_cameraprimus_floor(baseline_re_eval: float) -> float:
    """Spec line 261: floor = 75 if re-eval confirms 75.2; raise to
    re_eval - 5 if re-eval is higher. Used for both cameraprimus
    (single-staff) and cameraprimus_systems variants — each gets its
    own baseline + dynamic floor."""
    return max(CAMERAPRIMUS_FLOOR_BASE, baseline_re_eval - CAMERAPRIMUS_REGRESSION_TOLERANCE)


def _classify_lieder(mean_onset_f1: float) -> str:
    if mean_onset_f1 >= STRONG_THRESHOLD:
        return "Strong"
    if mean_onset_f1 >= MIXED_THRESHOLD:
        return "Mixed"
    return "Flat"


def evaluate(
    *,
    lieder_mean_onset_f1: float,
    musicxml_validity_rate: float,
    per_dataset: Dict[str, float],
    cameraprimus_systems_baseline: float,
    cameraprimus_baseline: float,
    lc6548281_onset_f1: Optional[float] = None,
) -> GateResult:
    """Aggregate the inputs into a verdict.

    Inputs:
        lieder_mean_onset_f1: from eval/results/lieder_<name>.csv
        musicxml_validity_rate: from same CSV (corroborating only)
        per_dataset: dict {dataset_name: composite quality_score, 0-100 scale};
            7 keys expected (synthetic_systems, grandstaff_systems, grandstaff,
            primus_systems, primus, cameraprimus_systems, cameraprimus)
        cameraprimus_systems_baseline: Stage 2 v2 quality on token_manifest_stage3.jsonl
        cameraprimus_baseline: Stage 2 v2 quality on token_manifest_full.jsonl
        lc6548281_onset_f1: optional sanity-check value (None if not evaluated)
    """
    floors = dict(PER_DATASET_FLOORS)
    floors["cameraprimus_systems"] = _resolve_cameraprimus_floor(cameraprimus_systems_baseline)
    floors["cameraprimus"] = _resolve_cameraprimus_floor(cameraprimus_baseline)

    per_dataset_results = {}
    any_floor_fail = False
    for dataset, floor in floors.items():
        measured = per_dataset.get(dataset)
        if measured is None:
            # Treat missing-input as floor-fail (decision-gate is invoked
            # only after Task 2 produces all 7; missing means upstream broke).
            per_dataset_results[dataset] = FloorResult(dataset=dataset, measured=float("nan"), floor=floor, passed=False)
            any_floor_fail = True
            continue
        passed = measured >= floor
        per_dataset_results[dataset] = FloorResult(dataset=dataset, measured=measured, floor=floor, passed=passed)
        if not passed:
            any_floor_fail = True

    lieder_outcome = _classify_lieder(lieder_mean_onset_f1)

    if any_floor_fail:
        verdict = Verdict.DIAGNOSE
    elif lieder_outcome == "Strong":
        verdict = Verdict.SHIP
    elif lieder_outcome == "Mixed":
        verdict = Verdict.INVESTIGATE
    else:
        verdict = Verdict.PIVOT

    return GateResult(
        verdict=verdict,
        lieder_mean_onset_f1=lieder_mean_onset_f1,
        lieder_outcome=lieder_outcome,
        musicxml_validity_rate=musicxml_validity_rate,
        per_dataset_results=per_dataset_results,
        cameraprimus_systems_baseline=cameraprimus_systems_baseline,
        cameraprimus_baseline=cameraprimus_baseline,
        lc6548281_onset_f1=lc6548281_onset_f1,
    )


def render_report(result: GateResult, *, name: str = "stage3_v2") -> str:
    lines = []
    lines.append(f"# Decision Gate Report — {name}")
    lines.append("")
    lines.append(f"## Verdict: **{result.verdict.value.upper()}**")
    lines.append("")
    lines.append(f"- Lieder mean onset_f1: **{result.lieder_mean_onset_f1:.4f}** ({result.lieder_outcome})")
    if result.lc6548281_onset_f1 is not None:
        lines.append(f"- lc6548281 sanity-check onset_f1: {result.lc6548281_onset_f1:.4f} (threshold ≥ 0.10)")
    lines.append(f"- MusicXML validity rate (corroborating): {result.musicxml_validity_rate:.3f}")
    lines.append("")
    lines.append("## Per-dataset quality regression-check")
    lines.append("")
    lines.append("| Dataset | Measured | Floor | Status |")
    lines.append("|---|---|---|---|")
    for ds in ["synthetic_systems", "grandstaff_systems", "grandstaff", "primus_systems", "primus", "cameraprimus_systems", "cameraprimus"]:
        r = result.per_dataset_results.get(ds)
        if r is None:
            lines.append(f"| {ds} | — | — | MISSING |")
            continue
        status = "✅ PASS" if r.passed else "❌ FAIL"
        lines.append(f"| {ds} | {r.measured:.2f} | {r.floor:.2f} | {status} |")
    lines.append("")
    cps_floor = result.per_dataset_results["cameraprimus_systems"].floor
    cp_floor = result.per_dataset_results["cameraprimus"].floor
    lines.append(f"_cameraprimus_systems floor = max(75, {result.cameraprimus_systems_baseline:.2f} - 5) = {cps_floor:.2f}; cameraprimus floor = max(75, {result.cameraprimus_baseline:.2f} - 5) = {cp_floor:.2f}_")
    lines.append("")
    lines.append("## Decision flow (spec §Phase 2)")
    lines.append("")
    if result.verdict == Verdict.SHIP:
        lines.append("All regression-checks PASS + lieder Strong → **Ship**: open PR, set up follow-ups.")
    elif result.verdict == Verdict.INVESTIGATE:
        lines.append("All regression-checks PASS + lieder Mixed → **Investigate**: residual error mode analysis.")
    elif result.verdict == Verdict.PIVOT:
        lines.append("All regression-checks PASS + lieder Flat → **Pivot**: Phase 0 / Audiveris alternative.")
    else:
        lines.append("One or more per-dataset floors FAIL → **Diagnose** before any pivot decision.")
        lines.append("Don't draw lieder conclusions over a broken baseline.")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2 decision gate for Stage 3.")
    p.add_argument("--lieder-onset-f1", type=float, required=True, help="Mean onset_f1 from eval/results/lieder_<name>.csv")
    p.add_argument("--musicxml-validity", type=float, required=True, help="Aggregate musicxml_validity_rate from the same CSV")
    p.add_argument("--per-dataset-json", type=str, required=True, help='JSON: {"dataset": composite_quality, ...} for 7 datasets')
    p.add_argument("--cameraprimus-systems-baseline", type=float, required=True, help="Stage 2 v2 cameraprimus_systems quality from token_manifest_stage3.jsonl eval")
    p.add_argument("--cameraprimus-baseline", type=float, required=True, help="Stage 2 v2 cameraprimus quality from token_manifest_full.jsonl eval")
    p.add_argument("--lc6548281-onset-f1", type=float, default=None, help="Optional sanity-check value")
    p.add_argument("--name", type=str, default="stage3_v2", help="Report identifier")
    p.add_argument("--output", type=Path, default=None, help="Optional path to write the markdown report")
    args = p.parse_args()

    per_dataset = json.loads(args.per_dataset_json)
    result = evaluate(
        lieder_mean_onset_f1=args.lieder_onset_f1,
        musicxml_validity_rate=args.musicxml_validity,
        per_dataset=per_dataset,
        cameraprimus_systems_baseline=args.cameraprimus_systems_baseline,
        cameraprimus_baseline=args.cameraprimus_baseline,
        lc6548281_onset_f1=args.lc6548281_onset_f1,
    )
    report = render_report(result, name=args.name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0 if result.verdict in (Verdict.SHIP, Verdict.INVESTIGATE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
python3 -m pytest tests/eval/test_decision_gate.py -v
```

Expected: 12 tests, all PASS.

- [ ] **Step 5: Run on real Phase 2 evidence**

Pull together the numbers from Tasks 2, 5, 6 and run:

```bash
# Extract per-dataset measurements from Stage 3 v2 JSON outputs.
PER_DS_JSON=$(python3 -c "
import json, sys
ds = {}
for path in [
    'eval/results/per_dataset_stage3_v2_systems.json',
    'eval/results/per_dataset_stage3_v2_singlestaff.json',
    'eval/results/per_dataset_stage3_v2_synthetic.json',
]:
    with open(path) as f: d = json.load(f)
    for name, m in d.get('by_dataset', {}).items():
        ds[name] = m.get('quality_score')   # adjust key name if different
print(json.dumps(ds))
")

# Extract Stage 2 v2 cameraprimus baselines (one per manifest).
CPS_BASELINE=$(python3 -c "
import json
print(json.load(open('eval/results/per_dataset_stage2_v2_systems.json')).get('by_dataset', {}).get('cameraprimus_systems', {}).get('quality_score'))
")
CP_BASELINE=$(python3 -c "
import json
print(json.load(open('eval/results/per_dataset_stage2_v2_singlestaff.json')).get('by_dataset', {}).get('cameraprimus', {}).get('quality_score'))
")

python3 -m eval.decision_gate \
  --lieder-onset-f1 <Task 5 mean> \
  --musicxml-validity <Task 5 rate> \
  --per-dataset-json "$PER_DS_JSON" \
  --cameraprimus-systems-baseline "$CPS_BASELINE" \
  --cameraprimus-baseline "$CP_BASELINE" \
  --lc6548281-onset-f1 <Task 6 lc6548281 if present> \
  --name stage3_v2 \
  --output eval/results/decision_gate_stage3_v2.md
```

Expected: report written; exit code 0 if Ship/Investigate, 1 if Pivot/Diagnose. Adjust the `quality_score` JSON key in the bash extractor if `evaluate_stage_b_checkpoint.py` writes a different field name (Task 2 Step 13 already requires you to figure that out).

- [ ] **Step 6: Commit**

```bash
git add eval/decision_gate.py tests/eval/test_decision_gate.py eval/results/decision_gate_stage3_v2.md
git commit -m "feat(eval): Phase 2 decision-gate evaluator + verdict report"
```

---

### Task 8: Final handoff doc + decide on next step

**Files:**
- Create: `docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase2-complete-handoff.md` (date filled at completion)

- [ ] **Step 1: Draft the handoff based on the verdict**

Use this skeleton; fill the verdict-specific section based on `eval/results/decision_gate_stage3_v2.md`:

```markdown
# RADIO Stage 3 Phase 2 — Complete Handoff (YYYY-MM-DD)

> Verdict: **<Ship | Investigate | Pivot | Diagnose>**

## TL;DR

- Lieder mean onset_f1: **<value>** (<Strong/Mixed/Flat>)
- Per-dataset regression-check: <PASS for all | FAIL on <which datasets>>
- MusicXML validity rate: <value>
- lc6548281 sanity check: <value> / <PASS/FAIL/NOT EVALUATED>

## The architectural question, answered

<One paragraph: did the frozen-encoder rebuild's val_loss-level success
translate into the user-visible quality gate? Reference the val_loss
numbers from Phase 1 and the onset_f1 number from Phase 2.>

## Per-dataset quality table

<Paste the markdown table from decision_gate_stage3_v2.md>

## Stratified by staves_in_system

<Paste from stratified_lieder_analysis output>

## Inputs (artifacts preserved)

- v2 ship artifact: `10.10.1.29:checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt`
- Lieder eval CSV: `eval/results/lieder_stage3_v2.csv`
- Per-dataset CSVs: `eval/results/stage3_v2_<dataset>.csv` × 5
- Cameraprimus baseline: `eval/results/cameraprimus_stage2_v2_200_baseline.csv`
- Decision-gate report: `eval/results/decision_gate_stage3_v2.md`

## Next step

<Pick one based on verdict:>

### Ship branch
- Open PR off feat/stage3-phase2-evaluation with the eval results.
- File follow-up issues for: Task 9 (v1 head-to-head, optional confirmation), Stage 2 v2 cameraprimus baseline now verified at <value>.

### Investigate branch
- Per-piece bottom-N analysis from lieder_stage3_v2.csv (which pieces dragged the mean below 0.30?)
- Cross-reference with stratified bucket counts (is one staves_in_system bucket dominating the failure mode?)
- Decide: targeted retrain (Stage 3 v3 with fix) or accept Mixed and ship.

### Pivot branch
- Phase 0 / Audiveris alternative is now the architectural recommendation.
- Spec rewrite or restart needed; this is a major decision — return to user with full evidence before drafting next plan.

### Diagnose branch
- Identify which per-dataset floor failed and why.
- Stage 3 broke something Stage 2 v2 had — root-cause before any pivot decision.
- Likely culprits: training mix, hyperparameters, frozen-encoder limits, augmentation regression. Reference spec §"Cross-cutting risks" line 308-318.

## Things NOT done

<Anything left for the next session — e.g., v1 head-to-head if Investigate, or
non-blocking follow-ups.>
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase2-complete-handoff.md
git commit -m "docs(handoff): Phase 2 complete — verdict <Ship/Investigate/Pivot/Diagnose>"
```

- [ ] **Step 3: Update memory**

Update `~/.claude/projects/-home-ari/memory/project_radio_stage3_design.md` with the verdict and final numbers (≤ 100 words; format consistent with the existing Phase 0 / Phase 1 entries).

- [ ] **Step 4: Open PR for the eval results**

```bash
gh pr create --base main --head feat/stage3-phase2-evaluation \
  --title "RADIO Stage 3 Phase 2 — evaluation + decision gate (verdict: <X>)" \
  --body-file docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase2-complete-handoff.md
```

---

### Task 9: (Conditional) v1 head-to-head if v2 fails Strong

**Run this task only if Task 7's verdict is Investigate, Pivot, or Diagnose.** Skip entirely on Ship.

**Goal:** confirm whether v1's marginally-different val_loss (0.169 vs v2's 0.164) translates into a different lieder onset_f1, which would inform the residual-error analysis.

**Files:**
- No code changes.
- Generated: `eval/results/lieder_stage3_v1/`, `eval/results/lieder_stage3_v1.csv`

- [ ] **Step 1: Run lieder inference on v1 best.pt**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m eval.run_lieder_eval --checkpoint checkpoints\full_radio_stage3_v1\stage3-radio-systems-frozen-encoder_best.pt --config configs\train_stage3_radio_systems.yaml --name stage3_v1 --beam-width 5 --max-decode-steps 512'
```

- [ ] **Step 2: Score**

```bash
ssh 10.10.1.29 'cd Clarity-OMR-Train-RADIO && venv-cu132\Scripts\python.exe -m eval.score_lieder_eval --predictions-dir eval\results\lieder_stage3_v1 --reference-dir data\openscore_lieder\scores --name stage3_v1 --cheap-jobs 8 --tedn-jobs 4'
```

- [ ] **Step 3: Compare to v2**

```bash
scp 10.10.1.29:Clarity-OMR-Train-RADIO/eval/results/lieder_stage3_v1.csv eval/results/
python3 -c "
import csv, statistics
for ckpt in ['v1', 'v2']:
    rows = list(csv.DictReader(open(f'eval/results/lieder_stage3_{ckpt}.csv')))
    f1s = [float(r['onset_f1']) for r in rows if r.get('onset_f1') not in (None,'')]
    print(f'{ckpt}: n={len(f1s)} mean_onset_f1={statistics.mean(f1s):.4f}')
"
```

Append the comparison to the Phase 2 handoff doc.

- [ ] **Step 4: Commit**

```bash
git add eval/results/lieder_stage3_v1.csv docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase2-complete-handoff.md
git commit -m "feat(eval): v1 head-to-head lieder eval (followup to verdict <X>)"
```

---

## Pre-flight checklist (must hold before saying "go")

- [ ] Branch `feat/stage3-phase2-evaluation` exists and is pushed.
- [ ] GPU box has v2 ship artifact at `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt`.
- [ ] GPU box has Stage 2 v2 best.pt for the cameraprimus baseline (Task 2).
- [ ] Lieder reference set at `data/openscore_lieder/scores/` and `data/clarity_demo/mxl/` per-dataset references all populated on GPU box.
- [ ] Phase 1 → Phase 2 launch handoff doc written (Task 0).

## Run the tasks

Operational tasks (1, 2, 3, 4, 5, 9) are SSH-driven; new-tool tasks (6, 7) are local TDD. Tasks 1 and 2 can run concurrently (different checkpoints, no GPU contention beyond memory). Task 3 must wait for the GPU to be free after Task 1; Task 4 likewise.

## Phase 2 → completion gate

Task 7's exit code:
- 0 (Ship or Investigate): proceed to Task 8 PR.
- 1 (Pivot or Diagnose): return to user with the report; do not auto-PR.

## What goes in the post-completion handoff

- Verdict + evidence (decision_gate report)
- Inputs (paths to all CSVs)
- Architectural-question paragraph (did frozen-encoder rebuild deliver?)
- Stratified breakdown
- Next-step plan based on verdict
- Things NOT done

---

## Test plan

For the two TDD tools:

- `tests/eval/test_stratified_lieder_analysis.py` — 5 cases: groups by staves_in_system, lc6548281 sanity (PASS/FAIL/NOT EVALUATED), missing onset_f1 rows excluded.
- `tests/eval/test_decision_gate.py` — 12 cases: all-pass-Strong→Ship, all-pass-Mixed→Investigate, all-pass-Flat→Pivot, threshold boundaries, per-dataset failure→Diagnose, dynamic cameraprimus floor, MusicXML not gated, report-render contains evidence.

Run before Task 7 commit: `python3 -m pytest tests/eval/ -v` — all green.

For the operational tasks (1-5, 9), the "test" is reading the resulting CSV and confirming non-empty / non-NaN columns. Each task's Step 4 (or equivalent) does this inline.

---

## Self-review notes (for the executing engineer)

- **Don't fast-path the smoke test (Task 1).** A successful smoke run materially de-risks Tasks 4 and 9. Greedy + 1 piece is < 1 minute; skipping it to "save time" is the bad trade.
- **Per-dataset eval (Task 3) takes the most wall.** Five datasets × ~hundreds of pieces × beam=5. Expect ~30-60 min. Run it on the GPU box overnight if scheduling is tight; the lieder eval (Task 4) can run after.
- **The decision-gate (Task 7) is the load-bearing tool.** If you find yourself thinking "let me just eyeball the numbers and write the verdict" — don't. The 11-case test suite exists exactly to catch boundary mistakes that look obvious but aren't (0.30 vs 0.2999, etc.).
- **If `--dataset synthetic_systems` doesn't exist on `run_clarity_demo_eval`, that's the first real ambiguity.** Synthetic data lives in the synthetic split, not the demo corpus. Document the workaround in the handoff (one-off scoring script, or extend the driver — whichever is smaller).
- **Cameraprimus floor is dynamic.** Don't hard-code 75; the decision-gate already does the right thing — pass Task 2's number to it.
- **Verdict ≠ ship recommendation by itself.** The handoff (Task 8) is where the architectural question gets answered in prose. The gate gives you the evidence; you write the conclusion.
