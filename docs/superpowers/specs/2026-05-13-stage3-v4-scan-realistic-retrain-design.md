# Stage 3 v4 — Scan-Realistic Retrain Design

**Date:** 2026-05-13
**Status:** Spec — pending user review
**Branch (will be):** `feat/stage3-v4-scan-realistic`
**Supersedes:** Stage 3 v3 (the data-rebalance retrain at `checkpoints/full_radio_stage3_v3/`)
**Motivating audit:** [2026-05-12-real-world-scan-failure-modes.md](../../audits/2026-05-12-real-world-scan-failure-modes.md) — TL;DR Issue 3: bass-clef-as-treble misread on real-world piano scans
**Prior context docs:**
- [2026-05-12-stage3-v3-data-rebalance-results.md](../../audits/2026-05-12-stage3-v3-data-rebalance-results.md) — v3 outcome: lieder mean 0.2398, near-ship; demo eval flat
- [2026-05-12-data-pipeline-and-capacity-audit.md](../../audits/2026-05-12-data-pipeline-and-capacity-audit.md) — Phase 1 data-pipeline audit informing v3

## Problem

The Stage 3 v3 best.pt mis-reads the bottom (bass-clef) staff of beginner-piano scans as treble clef. Pitches are entangled with the clef assumption — they're emitted in treble register on what is visually a bass staff. Post-decode cleanup (PR #55) addresses two of the three failure modes from the real-world-scan audit (`drop_phantom_staves` and the merged-system collapse), but the bass-clef misread cannot be safely fixed by post-processing: correcting the clef tag alone leaves pitches off by ~20 semitones.

The audit identifies the root cause as a training-data gap: the Stage 3 v3 corpus mix is 100% clean engraving (synthetic + grandstaff + primus + cameraprimus). The only real-scan corpus in the repo, AudioLabs v2 (`omr_layout_real`), has only layout-level labels (system bboxes for Stage A YOLO) and cannot be added to Stage 3 without token-level transcriptions.

This spec retrains Stage 3 with a new scan-realistic corpus derived from the existing engraved `grandstaff_systems`, gated by cheap validation that confirms (a) bass-clef-misread is the dominant lieder failure cluster and (b) the training pipeline can still overfit a small clean subset.

## Goal & success criteria

### Primary gate (must pass for v4 to ship)

- **Bethlehem** (`Scanned_20251208-0833.jpg`) through `scripts/predict_pdf.py` with v4 best.pt at default settings: all 4 visual systems detected, every bass-staff emits `clef-F4`, AND pitches in the bass part have median octave ≤ 3 (i.e., bass register, not treble register). Manual MXL inspection.
- **TimeMachine** (`Receipt - Restoration Hardware.pdf`) through `predict_pdf.py` at default settings: same — bass clefs and bass-register pitches on all 4 systems.

If primary fails, v4 is not shippable and we escalate (Section 7).

### Secondary criteria (must not regress, ideally improve)

- Lieder corpus-mean onset_f1 ≥ 0.2398 (v3's value). Run on the same 139-piece eval split (hash `8e7d206f53ae3976`).
- Bottom-quartile lieder (33 pieces with v3 onset_f1 < 0.10): at least half (≥17) move to onset_f1 ≥ 0.10.
- A3 per-corpus token accuracy: grandstaff_systems stays ≥ 0.80 (was 0.904 in v3). Don't trade off clean-engraving performance for scan robustness.

## Phase architecture

```
Phase 0 — Cheap validation (parallel, ~half-day total, no GPU contention)
  ├── 0a: Bottom-quartile failure-mode analysis on 33 lieder pieces (CPU, half-day)
  └── 0b: Overfit smoke test — trainer overfits 20 easy clean grand-staff samples (GPU, ~10 min)

Phase 1 — Build scanned_grandstaff_systems corpus (~3-5h on seder)
  ├── 1a: Calibrate scan-degradation parameters against real scans (CPU, ~2h)
  ├── 1b: Build degraded corpus + manifest (CPU, ~30 min)
  └── 1c: Rebuild encoder cache from scratch with the new corpus (GPU, ~3-4h)

Phase 2 — Stage 3 v4 retrain (~1h on seder)
  └── v3 config + scanned_grandstaff_systems added; 9000 steps cached

Phase 3 — Evaluation (~3-4h on seder)
  ├── 3a: Bethlehem + TimeMachine through predict_pdf (primary gate)
  ├── 3b: 139-piece lieder ship-gate (secondary)
  ├── 3c: 4-piece HF demo (signal only)
  └── 3d: A3 token-accuracy spot check including new corpus

Total wall-clock: ~1-2 days end-to-end.
```

## Phase 0 — Cheap validation

### 0a — Bottom-quartile lieder failure-mode analysis

**Input:** `eval/results/lieder_stage3_v3_best_scores.csv` — filter to the 33 pieces with `onset_f1 < 0.10`.

**Pipeline:**
1. Re-run each piece through `scripts/predict_pdf.py --dump-tokens` against v3 best.pt to capture raw decoder token streams (no inference re-run if the existing run dumped tokens; otherwise this is ~1h GPU time).
2. Extract per-system features: predicted staff count, predicted clef tags per staff, predicted pitch register median per staff, ground-truth clef tags (from source `.mxl`), page DPI, system count, source library (Schubert/Schumann/etc.).
3. Cluster pieces. Flag each piece as one or more of:
   - **bass-clef-misread**: bottom staff predicted `clef-G2` but ground truth `clef-F4` (Bethlehem/TimeMachine pattern).
   - **phantom-staff residual**: post-decode emitted >2 staves on a 2-staff grand-staff (post-#55 should have caught these — sanity check).
   - **key/time-signature first-emission residual**: PR #47 collapse pattern — pieces with non-4/4 time or non-C-major key.
   - **other**: anything outside the above.

**Output:** `docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md` — per-cluster counts, 2-3 example pieces per cluster, recommendation.

**Gate:** if bass-clef-misread ≥ 30% of bottom-quartile (≥10 of 33 pieces), proceed to Phase 1. If <30%, pause and re-scope before committing GPU to a retrain.

### 0b — Overfit smoke test

**Subset selection:** 20 clean engraved grand-staff systems from `grandstaff_systems` train split, deterministically pre-picked (e.g., first 20 in lexicographic source-path order), kept in a static manifest at `data/manifests/overfit_smoke_v4.jsonl`. **Bethlehem and TimeMachine stay out** — they're the held-out primary gate. Selection criteria:
- 2-staff grand-staff systems only (the target structure).
- Token sequence length ≤ 256 (sanity-check learnability — keep loss signal tight).
- v3 already gets these right at A3 spot-check (so the smoke isolates "trainer overfits" from "data is hard").

**Config:** `configs/train_stage3_v4_overfit_smoke.yaml`:
- 20-sample manifest; `effective_samples_per_epoch=20`, `batch_size=4`, `epochs=125` → 625 opt-steps.
- No augmentation. `tier_grouped_sampling=false` (small enough for naive batching).
- `lr_dora=0.002`, `weight_decay=0.0`, `label_smoothing=0.0`, `contour_loss_weight=0.0`.
- Validate every 50 steps on the SAME 20 samples (val_loss should track train_loss → both near zero).
- Init from `checkpoints/full_radio_stage2_systems_v2/_best.pt`.

**Gate:**
- Train CE loss < 0.05 at step 500.
- Val CE loss < 0.10 at step 500 (val=train; allows a small generalization margin for dropout/eval-mode differences).
- Both pass → training pipeline sound, proceed to Phase 1c (cache rebuild).
- Either fails → stop. Investigate before any further GPU spend. Likely candidates: tokenizer regression, data loader bug, optimizer config, model factory regression.

**Deliverable:** the smoke config stays in the repo as a 10-minute pre-flight check for any future retrain.

### Phase 0 dependencies

0a and 0b are independent. 0a is CPU-only; 0b uses GPU. They can run in parallel on the same day.

Phase 1 cannot start until BOTH 0a's gate passes AND 0b's gate passes.

## Phase 1 — Build scanned_grandstaff_systems corpus

### 1a — Calibrate scan-degradation parameters

**Inputs to characterize:** Bethlehem.jpg, TimeMachine.pdf (rendered as page images), plus 5 IMSLP scans of beginner piano scores (sourced from IMSLP, ~30 min hand-curation).

**Measurements per real scan:**
- Estimated page rotation (Hough transform on staff lines).
- Estimated JPEG quality (via `jpeginfo` or heuristic on quantization tables; for PDFs, decompose embedded images).
- Noise level (variance of high-frequency residual after Gaussian smooth).
- Brightness/contrast statistics.
- Perceived blur (Laplacian variance).

**Output:** `docs/audits/2026-05-13-real-scan-degradation-calibration.md` — distribution summary, recommended parameter ranges for the offline pipeline.

### 1b — Degradation pipeline + corpus build

New module: `src/data/scan_degradation.py`. Pure-Python (PIL + numpy + cv2 OK), no GPU.

```python
def apply_scan_degradation(image: PIL.Image, seed: int) -> PIL.Image:
    """Deterministic scan-realistic degradation, seeded per source for reproducibility."""
    rng = numpy.random.default_rng(seed)
    # Pipeline (ranges calibrated in 1a, defaults below are starting point):
    #   1. Rotation: uniform ±1.5°, border replicate
    #   2. Perspective warp: corners offset up to 3% of dimension
    #   3. Brightness ±0.10, contrast ±0.12
    #   4. Paper-texture overlay: low-amplitude noise field, sigma=0.04
    #   5. Light Gaussian blur: kernel 3, sigma uniform[0.3, 0.9]
    #   6. Salt-and-pepper noise: fraction 0.0015
    #   7. JPEG round-trip: quality uniform[75, 90]
```

**Reproducibility invariant:** seed = `hash(source_path) & 0xFFFFFFFF`. Same input always produces same output. No per-run randomness.

**Unit tests (CPU-only, run locally):** `tests/data/test_scan_degradation.py`
- `apply_scan_degradation` is deterministic for fixed seed.
- Output image has same dimensions as input.
- Output is not pixel-identical to input (degradation actually happened).
- Output is grayscale if input is grayscale.

**Build script:** `scripts/data/build_scanned_grandstaff_systems.py`:
- Reads `data/processed/grandstaff_systems/manifests/synthetic_token_manifest.jsonl`.
- For each entry, loads source image, applies degradation, writes to `data/processed/scanned_grandstaff_systems/images/<same-stem>.png`.
- Token labels are copied verbatim — only the image changes.
- Writes manifest `data/processed/scanned_grandstaff_systems/manifests/synthetic_token_manifest.jsonl` with same schema as grandstaff but updated image paths.
- Inherits the same train/val split from grandstaff (no leakage — a source in grand-train stays in scanned-grand-train).
- Idempotent: if output exists with matching seed, skip.

**Verification:** post-build, spot-check 10 random outputs visually. Run `scripts/visualize_audiolabs_systems.py`-style overlay to confirm tokens still align (they should — image transforms preserve per-staff y-positions within rotation tolerance).

**Wall-clock estimate:** ~30 min CPU on seder for ~107k grandstaff systems.

### 1c — Encoder cache rebuild

Build a new cache from scratch covering synthetic + grandstaff + scanned_grandstaff + primus. Cameraprimus stays live-tier.

```bash
# On seder, in venv-cu132:
python scripts/build_encoder_cache.py \
    --datasets synthetic_systems grandstaff_systems scanned_grandstaff_systems primus_systems \
    --output data/cache/encoder \
    --encoder radio_h \
    --encoder-checkpoint checkpoints/full_radio_stage2_systems_v2/_best.pt
```

**Output:** new `cache_hash16` (recorded in cache metadata). Old cache `ac8948ae4b5be3e9` stays on disk as fallback.

**Verification (PHASE 1c gate):**
- A2 parity test on a 10-sample slice of the new cache: `max_abs_diff < 0.01` for BILINEAR resize (v3 had a known A2 fail from cache-vs-live preprocessing mismatch — v4's fresh cache should pass).
- If A2 still fails, that's a pre-existing condition unrelated to this retrain; document and proceed. (v3 shipped despite A2 fail.)

**Wall-clock estimate:** ~3-4h on RTX 5090 (~215k entries from v3 + ~107k new = ~322k samples through C-RADIO-H encoder).

## Phase 2 — Stage 3 v4 retrain

**Config:** `configs/train_stage3_radio_systems_v4.yaml`

```yaml
stage_name: stage3-radio-systems-frozen-encoder-v4
stage_b_encoder: radio_h
epochs: 1
effective_samples_per_epoch: 9000    # match v3 step count
batch_size: 1                        # SENTINEL — unused with tier_grouped_sampling
max_sequence_length: 1024
grad_accumulation_steps: 1           # SENTINEL — unused with tier_grouped_sampling

tier_grouped_sampling: true
b_cached: 16
b_live: 2
grad_accumulation_steps_cached: 1
grad_accumulation_steps_live: 8
cached_data_ratio: 0.875             # cameraprimus = 0.125 live-tier
cache_root: data/cache/encoder
cache_hash16: <new hash from Phase 1c>

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

# Within-cached weights (must sum to 1.0); total share = 0.875 × ratio.
# Plan: synth 0.15 / grand 0.30 / scanned_grand 0.30 / primus 0.125 ; camera 0.125 (live).
#   synth:        6/35 = 0.171428 → 0.875 × 6/35 = 0.150
#   grand:       12/35 = 0.342857 → 0.875 × 12/35 = 0.300
#   scanned:     12/35 = 0.342857 → 0.875 × 12/35 = 0.300
#   primus:       5/35 = 0.142857 → 0.875 × 5/35 = 0.125
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

**Init checkpoint:** `checkpoints/full_radio_stage2_systems_v2/_best.pt` (same as v3 — not v3 best.pt, because we want to re-learn from S2v2's encoder with the new corpus mix, not warm-start from v3's clean-only fit).

**Checkpoint dir:** `checkpoints/full_radio_stage3_v4/`. Logs: `logs/full_radio_stage3_v4_steps.jsonl`.

**Run command (seder):**
```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python -u src/train/train.py \
    --stage-configs configs/train_stage3_radio_systems_v4.yaml \
    --mode execute \
    --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \
    --start-stage stage3-radio-systems-frozen-encoder-v4 \
    --checkpoint-dir checkpoints/full_radio_stage3_v4 \
    --token-manifest src/data/manifests/token_manifest_stage3.jsonl \
    --step-log logs/full_radio_stage3_v4_steps.jsonl'
```

**Smoke test before main run:** 500-step short cycle, verify pre-flight assertion fires (`[freeze] trainable encoder params: 0`), checkpoint writes cleanly. Recovers v3's resume-mid-stage pattern (`--start-stage` is included on first launch from S2v2; on crash recovery, drop it).

**Wall-clock estimate:** ~1h on RTX 5090 (matches v3's 45min + new corpus overhead).

## Phase 3 — Evaluation

### 3a — Primary gate (Bethlehem + TimeMachine)

Run both real scans through `scripts/predict_pdf.py` with v4 best.pt at **default settings** (phantom-drop on, bass-clef repair OFF, default YOLO conf):

```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf \
    Scanned_20251208-0833.jpg /tmp/bethlehem_v4.musicxml \
    --stage-b-weights checkpoints/full_radio_stage3_v4/<best.pt> \
    --dump-tokens /tmp/bethlehem_v4.tokens.jsonl'
```

**Verification:** parse the resulting MXL with `music21`, extract per-system per-staff clef tags AND pitch register medians. Assert:
- Bethlehem: at least 3 of 4 visual systems detected (Stage A unchanged; Issue 1b not in scope). For each detected system: bass staff has `clef-F4` AND pitch median octave ≤ 3.
- TimeMachine: all 4 systems. Same per-system assertion.

Pass = all detected systems pass. Single failure = primary gate fails.

### 3b — Lieder ship-gate

```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m eval.run_lieder_eval \
    --stage-b-weights checkpoints/full_radio_stage3_v4/<best.pt> \
    --run-name stage3_v4_best \
    --run-scoring'
```

**Verification:** parse `eval/results/lieder_stage3_v4_best_scores.csv`. Compare against v3:
- Corpus-mean onset_f1 ≥ 0.2398 (no regression).
- Among the 33 pieces with v3 onset_f1 < 0.10, count how many now have v4 onset_f1 ≥ 0.10. Target ≥17 of 33 (≥51%).

### 3c — 4-piece HF demo

Signal only; report mean onset_f1 but don't gate on it. v3's demo eval was 0.0510 (flat vs v2's 0.0589), known to be a small-sample outlier. v4 may or may not move it.

### 3d — A3 per-corpus token-accuracy spot check

5 samples per corpus (synthetic, grandstaff, scanned_grandstaff, primus, cameraprimus). Compute token-level accuracy via teacher-forced decoder. Expected:
- grandstaff_systems ≥ 0.80 (was 0.904 in v3; small drop acceptable — must not collapse).
- scanned_grandstaff_systems ≥ 0.50 (new corpus, no prior baseline; signal that the model is learning it at all).
- Other corpora within 5pp of v3 values.

If grandstaff drops below 0.80, the new corpus mix is hurting clean-engraving performance — flag and consider rebalancing for v4.1.

## Escalation paths (documented; not part of v4 plan-of-record)

If Phase 2 misses the primary gate:

- **(B) Multi-variant cache rebuild.** Build 3 degraded variants per scanned_grandstaff source (different seeds for different parameter draws). Cache size grows ~3× for that corpus. Tier sampler picks one variant per step. Wall-clock: ~half-day cache rebuild + ~1h retrain. Spec'd as v4.1 in a follow-up doc.

- **(C) No-cache live-tier fine-tune.** Drop the encoder cache for grandstaff + scanned_grandstaff (or all corpora). Run online image augmentation (`_apply_online_augmentations` in `src/train/train.py`) through the live encoder. Unfreeze encoder LoRA. Estimated 10-50h training. Highest cost, highest ceiling. Only if (B) also fails.

(A) → (B) → (C) is the escalation ladder. Each step buys more augmentation diversity at increasing cost.

## Open questions / risks

1. **Risk: degradation calibration is wrong.** If the offline pipeline doesn't match real-scan distribution, v4 trains on a different distribution than test. Mitigation: Phase 1a calibration step, with a 5-10 scan reference set including Bethlehem/TimeMachine.

2. **Risk: bass-clef misread is NOT the dominant lieder failure.** Phase 0a's gate guards against this — if <30% of bottom-quartile failures are bass-clef-misread, we pause before retraining.

3. **Risk: trainer regression silently makes v4 worse than v3 on engraved corpora.** Phase 0b overfit smoke test catches infrastructure regressions; Phase 3d A3 spot check catches engraving regressions on the real run.

4. **Open: should we add real AudioLabs v2 pages with auto-labeled tokens (e.g., decoded via v3 then hand-corrected) to the mix as a small-but-real corpus?** Defer — that's a separate sub-project (label production cost is significant; lieder eval is already a real-distribution proxy).

5. **Open: how is the new encoder cache versioned?** Old `ac8948ae4b5be3e9` stays as v3 fallback. v4 cache gets a fresh hash16 and replaces v3 as the active cache for new training runs. Consider keeping a `cache_versions.md` log entry.

6. **Risk: Stage A YOLO missed-system case (Issue 1b in the audit) is NOT addressed by this retrain.** Bethlehem-style missed-system failures will persist; recovery still requires `--yolo-conf 0.05`. Issue 1b is a separate sub-project (audit Recommendation #4: `--yolo-conf-retry`). v4's primary gate accepts "at least 3 of 4 visual systems detected" for Bethlehem to avoid coupling this work to YOLO retraining.

## Branch & PR plan

- Branch: `feat/stage3-v4-scan-realistic` off `main`.
- Commits, roughly:
  1. `feat(data): scan_degradation module + tests`
  2. `feat(data): build_scanned_grandstaff_systems script`
  3. `feat(config): overfit smoke config + manifest`
  4. `feat(config): Stage 3 v4 training config`
  5. `audit: bottom-quartile lieder cluster analysis (Phase 0a)`
  6. `audit: real-scan degradation calibration (Phase 1a)`
  7. `audit: Stage 3 v4 results` (after Phase 3)
- One bundled PR at the end with all of the above + the v4 best.pt artifact path documented. Mirrors the v3 branch pattern.

## Out of scope for this spec

- AudioLabs v2 token-label production.
- `--yolo-conf-retry` (Stage A Issue 1b).
- Pitch-aware bass-clef post-decode repair (audit Recommendation #2). v4 aims to make this unnecessary; if v4 succeeds, that recommendation is dropped.
- Cache versioning system / cache-vs-live preprocessing parity fix (A2 long-standing fail).
