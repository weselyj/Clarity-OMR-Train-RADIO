# Stage 3 v3 — train-vs-inference gap diagnostic plan

**Date:** 2026-05-12
**Status:** investigation needed (Phase 3 verdict pending lieder)
**Parent:** [data-pipeline-and-capacity-audit](2026-05-12-data-pipeline-and-capacity-audit.md)

## One-paragraph problem statement

The Stage 3 v3 retrain landed every Phase 2 fix correctly — encoder freeze worked (verified: 774/774 encoder keys byte-identical between Stage 2 v2 best.pt and Stage 3 v3 best.pt with zero drift), synthetic val split landed (984 val entries), dataset_mix rebalanced toward grandstaff, max_sequence_length raised to 1024, retrained for 9000 steps in 45 minutes. **Every training metric improved dramatically.** Per-corpus token accuracy on training data (A3): grandstaff 0.400 → 0.904, synthetic 0.113 → 0.257, primus 0.753 → 0.810, cameraprimus 0.865 → 0.869. Final val_loss per dataset: grandstaff 0.171, primus 0.159, cameraprimus 0.143, synthetic 0.789. **Yet the 4-piece HF demo onset_f1 mean is 0.0510 vs v2's 0.0589 — flat.** This is the same "training metrics improve, inference metrics don't move" pattern the Frankenstein diagnostic surfaced for the encoder-drift hypothesis (PR #49). Two consecutive fixes (encoder freeze, data rebalance) have failed to move the demo metric, suggesting the failure mode lives somewhere else in the pipeline.

## Verified vs ruled out

| Hypothesis | Result | Evidence |
|---|---|---|
| Encoder DoRA drift (PR #48 audit's primary suspect) | RULED OUT | Frankenstein experiment (PR #49) — replacing drifted v2 encoder with Stage 2 v2 encoder gave +0.001 onset_f1 |
| Encoder freeze regressed in v3 | DID NOT REGRESS | 774/774 encoder keys identical between S2v2 and S3v3 best.pt; pre-flight assertion fired correctly |
| Training-data imbalance is the dominant cause | DID NOT TRANSLATE | v3 fixed the imbalance per the Phase 1 audit; training metrics jumped (grandstaff 0.40→0.90) but demo onset_f1 didn't move |
| Decoder undersized for multi-staff | UNLIKELY | Stream 5 audit: decoder 116M ≈ upstream Clarity-OMR reference 120M |

## The leading hypothesis: train-inference distribution shift at the encoder boundary

The strongest candidate for the residual gap is **the cache-vs-live encoder feature mismatch confirmed by A2 in every audit run since v2**:

- A2 (v3, 2026-05-12): max_abs_diff 514 (BILINEAR), 46.7 (LANCZOS), with 15 of 15 compared samples failing the < 0.01 PASS threshold
- A2 (v2, PR #48): max_abs_diff 85.15 — originally attributed to encoder DoRA drift
- v3 freeze fix proved that drift was not the cause (encoder weights identical); the cache itself differs from what the live encoder produces at inference for the same image

**Implication.** The decoder is trained on one feature distribution (cache features) but evaluated on a different one (live encoder features). On training data (A3), the decoder can rely on memorized associations with specific cache feature vectors and score 0.90 token accuracy. On held-out demo pieces, the live encoder produces features the decoder has never seen, and generalization collapses.

**Why Frankenstein didn't fix it.** Frankenstein replaced the S3v2 *drifted* encoder with S2v2's encoder at inference. But the cache was apparently NOT built from S2v2's encoder either (A2 still fails with S2v2 weights, as v3 demonstrates). So Frankenstein didn't actually close the train-inference loop — it swapped one mismatched encoder for another mismatched encoder.

## Three falsifiable next experiments (in priority order)

### Experiment 1 — Live-encoder inference on a training piece (4 hours)

**Question.** Is the gap "cache-vs-live distribution shift" or "OOD generalization on demo pieces specifically"?

**Method.** Take 5 pieces from the **training set** (any corpus). Run the full inference pipeline (live encoder + decoder + assembly + scoring) on them, exactly like the demo eval does. Score against their reference MusicXML.

Expected outcomes:
- If onset_f1 on training pieces is **high (≥ 0.5)** → the issue is OOD generalization on demo pieces; train-inference shift is acceptable. Pivot to architecture/data investigation.
- If onset_f1 on training pieces is **low (≤ 0.15)** → the issue is the cache-vs-live shift, biting in-distribution too. Pivot to fixing the cache.

This is the most diagnostic single experiment. It separates "model can't generalize" from "model can't even reproduce what it trained on under live-encoder inference."

### Experiment 2 — Rebuild the encoder cache from Stage 2 v2's encoder (~6 hours)

**Question.** If we rebuild the cache so it actually matches the encoder that v3 uses, does the demo metric move?

**Method.** Rerun the encoder cache builder against Stage 2 v2's encoder, producing a new cache hash. Update the v3 config to point at the new cache. Retrain Stage 3 v3.5. Re-run the four Phase 3 evals.

Expensive (6h cache rebuild + 45min retrain + ~1h eval), but the most direct test of the cache-mismatch hypothesis. Only run this if Experiment 1 confirms train-inference shift is the issue.

### Experiment 3 — Train Stage 3 v3 LIVE (no cache, ~12-24 hours)

**Question.** Same as Experiment 2, but without rebuilding the cache.

**Method.** Disable the encoder cache entirely. Train Stage 3 v3 with live encoding throughout. This is slower (cache buys ~6x throughput) but eliminates the train-inference shift completely.

Cheaper than Experiment 2 only if the cache rebuild fails or has its own bugs. Otherwise the cache rebuild is preferred.

## What Lieder (currently running) tells us

The 50-piece lieder eval is in flight (background task `bbhk4u5os`). Two information-theoretic outcomes:

- **Lieder mean onset_f1 ≈ 0.05-0.08 (matches demo).** Demo result was real, not noise. The 4-piece flat pattern reflects the 50-piece corpus too. Proceed with Experiment 1 to determine whether it's distribution shift or architecture.
- **Lieder mean onset_f1 noticeably higher (≥ 0.15).** Demo result was unusually pessimistic; v3 may have actually helped. Reassess whether the gap is real before launching experiments.

The lieder result will likely land within an hour of writing this and will steer the next move.

## Decision matrix

| Lieder corpus mean | Experiment 1 result | Action |
|---|---|---|
| ≥ 0.15 | (skip Exp 1) | Demo was noisy. Project may be closer to ship than thought. Re-eval demo with different beam/preprocessing variations. |
| < 0.15 | training pieces ≥ 0.5 onset_f1 | OOD generalization is the bottleneck. Sub-project: architecture investigation (DaViT, larger decoder, broader training data). |
| < 0.15 | training pieces < 0.15 onset_f1 | Train-inference shift is the bottleneck. Sub-project: cache rebuild (Exp 2) or live-train (Exp 3). |

## What to commit (regardless of verdict)

Phase 2 produced durable infrastructure improvements that stay in tree:

- `src/train/train.py` — auto-freeze + pre-flight assertion. Prevents future encoder drift on any cache-backed run.
- `tests/train/test_freeze_encoder.py` — 3 regression tests.
- `scripts/data/add_val_split_synthetic.py` — by-source val split.
- `configs/train_stage3_radio_systems_v3.yaml` — rebalanced config, useful as v3.X baseline.
- 5 Phase 1 audit scripts (`scripts/audit/inspect_synthetic_samples.py`, etc.) — durable audit infrastructure.

The v3 checkpoint (best val_loss 0.2806 @ step 8500) is preserved on seder at `checkpoints/full_radio_stage3_v3/` for follow-up experiments.

## TL;DR for handoff

Phase 2 implementation was perfect, training metrics jumped, demo metric didn't move. Encoder freeze is verified working. Cache-vs-live encoder mismatch persists from the v2 era (was misdiagnosed as encoder drift in PR #48). Highest-leverage next experiment is running live-encoder inference on a training piece — if that scores well, the issue is OOD; if it scores poorly, the issue is the cache and we rebuild it. Lieder eval is in flight to confirm whether the demo result is corpus-stable.
