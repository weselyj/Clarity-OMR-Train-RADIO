# Stage 3 v2 Training Audit Report

**Date:** 2026-05-11
**Plan:** [docs/superpowers/plans/2026-05-11-stage3-v2-training-audit-plan.md](../superpowers/plans/2026-05-11-stage3-v2-training-audit-plan.md)
**Spec:** [docs/superpowers/specs/2026-05-11-stage3-v2-training-audit-design.md](../superpowers/specs/2026-05-11-stage3-v2-training-audit-design.md)
**Checkpoint audited:** `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt` (step 4000, val_loss 0.148)
**Branch:** `audit/stage3-v2-training`
**Audit head:** `c65cd2d`

## TL;DR

**Bug found:** the encoder-side DoRA adapters were NOT frozen during Stage 3 v2 training despite the config and checkpoint filename claiming `frozen-encoder`. Stage 3 trained the decoder against cached encoder features built from the Stage 2 v2 checkpoint, while simultaneously updating the encoder's LoRA weights — so the encoder the decoder learned to translate and the encoder that emits features at inference are no longer the same model. A2 (encoder output parity) measured a max absolute difference of **85.15** between cached features and the trained-checkpoint's live encoder output (pass threshold: 0.01). The leading hypothesis going in — LANCZOS-vs-BILINEAR preprocessing skew between cache builder and inference — was tested explicitly and disproven; LANCZOS-prepped images produce a near-identical 78.62 max-abs delta against the cache, so the resampling kernel is not the cause.

**Recommendation:** Foundation has a bug requiring retraining to validate. Fix `_prepare_model_for_dora` in `src/train/train.py:1327-1331` so encoder-side LoRA parameters are excluded from the `requires_grad=True` set, then re-train Stage 3 (three options described under Recommendation below) and re-evaluate against `eval/run_clarity_demo_radio_eval.py` and the 50-piece lieder corpus before drawing any conclusion about whether the gap is a generalization failure.

## Phase A: Round-trip data verification

### A1 — Image preprocessing parity

**Result:** PASS
**Evidence:** `audit_results/a1_preprocessing.json` on seder. Script: `scripts/audit/a1_preprocessing_parity.py` at commit `609aa57`. 20 samples (5 per corpus: synthetic_systems, grandstaff_systems, primus_systems, cameraprimus_systems).

**Numbers:**
- `overall_max_abs_diff = 0.0` (bit-identical)
- All 20 samples returned `status = "compared"` with `nonzero_pixel_fraction = 0`

**Interpretation.** The training-time path (`_load_raster_image_tensor` in `src/train/train.py:413`) and the inference-time path (`_load_stage_b_crop_tensor` in `src/inference/decoder_runtime.py:16`) produced bit-identical tensors on every sample. Whatever else is wrong, the image preprocessing pipeline is not the cause: training and inference see the same pixel arrays for the same source images.

**Latent fragility observed (not a blocker, not surfaced by this run).** During code review of A1, the two preprocessing paths use algebraically equivalent but not bit-for-bit guaranteed float arithmetic for the target-width computation (the two formulas commute mathematically but interleave division and rounding differently). On the 20 samples chosen here, the two paths landed on identical integer widths and identical tensors, but on a source image whose aspect ratio happens to fall on a rounding boundary the two paths could diverge by a single pixel-column, which would in turn change the encoder output non-trivially. This is a pre-existing fragility worth tightening if either preprocessing path is touched again; it is not the source of the Stage 3 v2 eval gap.

### A2 — Encoder output parity (cached vs live)

**Result:** FAIL — catastrophic, dispositive.
**Evidence:** `audit_results/a2_encoder.json` on seder. Script: `scripts/audit/a2_encoder_parity.py` at commit `c65cd2d`. 20 samples (5 per corpus); cameraprimus has no cache tier so 5 are `no_cache_tier` and 15 are `compared`.

**Numbers (BILINEAR preprocessing, matches training and inference):**
- `bilinear_vs_cache_overall_max_abs = 85.15`
- `bilinear_vs_cache_overall_mean_abs = 0.6029`
- Pass criterion was `max <= 0.01 AND mean <= 0.001`
- Failed by **8,500x on max and 600x on mean**
- All 15 compared samples fail individually (every per-sample `max_abs_diff` is orders of magnitude above the threshold; no corpus is innocent)

**LANCZOS hypothesis (disproven).** The cache builder at `scripts/build_encoder_cache.py:93` uses `Image.LANCZOS`; both the trainer and inference use `Image.Resampling.BILINEAR`. Going in, this was the leading hypothesis: maybe the cache contained LANCZOS-resampled features and the decoder had to translate BILINEAR features at inference, so the eval gap was a resampling-kernel mismatch fixable by either rebuilding the cache with BILINEAR or switching inference to LANCZOS. To test, the A2 script also runs the encoder on LANCZOS-prepped images and diffs against the cache:
- `lanczos_vs_cache_overall_max_abs = 78.62`
- `lanczos_vs_cache_overall_mean_abs = 0.6126`

Nearly indistinguishable from the BILINEAR delta. **The preprocessing resampling kernel is not the cause.** The features in the cache simply do not come from the model that's now on disk.

**Root cause (diagnosed via direct checkpoint comparison and code inspection).** Two complementary lines of evidence converge on a single root cause:

1. **Checkpoint diff.** Loading both `checkpoints/full_radio_stage2_v2/_best.pt` and `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt` on seder and walking the LoRA parameter set shows 384/384 encoder-side LoRA keys changed between the two checkpoints (not floating-point noise — sample per-key max absolute differences are on the order of 0.028, 0.060, 0.077). 264/264 decoder-side LoRA keys also changed, which is expected. The encoder cache was produced from the Stage 2 v2 encoder; if encoder weights changed during Stage 3 training, the cache is stale relative to the trained model, which exactly matches A2's observation.

2. **Code inspection.** `src/train/train.py:1327-1331`, in `_prepare_model_for_dora`, contains:

   ```python
   for parameter in model.parameters():
       parameter.requires_grad = False
   for name, parameter in model.named_parameters():
       if "lora_" in name or any(marker in name for marker in new_module_keywords):
           parameter.requires_grad = True
   ```

   The second loop flips `requires_grad=True` for any parameter whose name contains `lora_`, with **no encoder-side filter** — no `"encoder" not in name` guard, no allowlist of decoder modules. All 648 LoRA parameters (384 encoder + 264 decoder) are activated. A `grep -rn` over `src/` for `freeze`, `frozen`, and `encoder.*requires_grad` returns zero relevant hits (only `frozen=True` on dataclasses and `frozenset` usages). The config docstring at `configs/train_stage3_radio_systems.yaml:4-5` states the encoder and encoder-side DoRA adapters are frozen, and the checkpoint filename embeds `frozen-encoder`, but no code path implements the freeze.

**Interpretation.** Stage 3 v2 ran with the same shape as a frozen-encoder + cached-features training step (the cached-features dataloader path was used for ~90% of opt steps per the config), but the encoder's LoRA adapters silently received gradients on the live-encoder tier and drifted away from the model that produced the cache. The decoder optimized against features from one encoder while at inference it sees features from a different one. Val loss looked good because validation was scored on the same cached features the decoder learned to translate, not on a live forward pass. The eval gap (val_loss 0.148 vs demo mean onset_f1 ~0.06 vs decision gate 0.241) is the cost of evaluating the trained pair (cached-features-trained decoder + drifted live encoder) on real images.

### A3 — Decoder on training data

**Result:** SKIPPED — fast-exit triggered by A2.
**Evidence:** spec §"Decision gate", plan Task 5.2 — any Phase A failure halts further auditing because the failure itself accounts for the eval gap and the fix-and-verify loop is the appropriate next step.

A3 would have measured whether the trained decoder reproduces its own training labels at inference. With A2 dispositive (the encoder distribution at inference is not the one the decoder was trained against), A3's likely outcome is "low token accuracy on training data" — which is the symptom the audit was investigating, not new information. Re-running A3 *after* the fix is the right time to do it: an A3 PASS on the retrained model is a useful sign-off that the fix took.

## Phase B: Process audit

**Not run — fast-exit triggered by A2 failure.** Per the spec, Phase B is conditional on Phase A passing cleanly; A2's catastrophic failure makes it dispositive on its own. Phase B becomes relevant again only if the fix-and-retrain produces a model whose Phase A passes but whose downstream evals still disappoint. At that point B1-B9 (dataloader, augmentation, model architecture, loss, optimizer + schedule, gradient handling, mixed precision, checkpoint integrity, validation loop) would still be worth walking through; the checklist is preserved in the plan for that contingency.

## Bug list (sorted by likely impact)

1. **HIGH — encoder DoRA not frozen during Stage 3 v2 training; cache stale relative to trained model.** `src/train/train.py:1327-1331` (`_prepare_model_for_dora`) activates every `lora_` parameter unconditionally, with no encoder-side filter. The config at `configs/train_stage3_radio_systems.yaml:4-5` and the checkpoint filename `stage3-radio-systems-frozen-encoder_best.pt` both assert a frozen encoder; the assertion is unsupported by the training code. A2 measures a max-abs delta of 85.15 between cached and live encoder features (pass threshold 0.01); the checkpoint diff confirms 384/384 encoder-side LoRA keys moved between Stage 2 v2 and Stage 3 v2. This is the load-bearing finding.

2. **OBSERVATION (not a bug per se) — latent train/inference width-rounding divergence detected during A1 review.** The two preprocessing pipelines (`src/train/train.py:413` and `src/inference/decoder_runtime.py:16`) compute target image width via algebraically equivalent but not bit-equal float expressions. A1's 20 samples all hit identical integer widths; a source image whose aspect ratio sits on a rounding boundary could land the two paths a pixel apart, which would propagate to non-trivial encoder-feature differences. Not a current cause of any failure; flagged as a pre-existing fragility worth normalizing if either preprocessing path is touched in the future.

## Recommendation

**Foundation has bug #1 requiring retraining to validate.**

**Specific code change.** Add an encoder-side guard to the LoRA activation loop in `_prepare_model_for_dora` (`src/train/train.py:1327-1331`). Equivalent forms:

- `if "lora_" in name and "encoder" not in name`, or
- explicitly partition by module path (preferred when the param names also distinguish cross-attention vs self-attention LoRA), or
- collect the frozen submodules first and walk only the trainable ones in the second loop.

Whichever form is chosen, the post-condition is: after `_prepare_model_for_dora` returns, every encoder-side parameter (including LoRA weights, magnitude vectors, and any DoRA-added modules embedded in the encoder) has `requires_grad=False`. This should be asserted in the train script (cheap, valuable) with a count comparison against the cached-encoder configuration. A regression test that constructs the model with the frozen-encoder config and checks `sum(p.requires_grad for n,p in model.named_parameters() if "encoder" in n) == 0` would have caught this and should be added.

**After the fix, three retrain options.** The choice depends on whether we want to keep the existing encoder cache or change the training contract:

(a) **Rebuild the encoder cache against the fixed configuration and retrain Stage 3 with a truly frozen encoder.** This is the closest to the original spec intent. The Stage 2 v2 encoder produces the features; the cache is rebuilt only if it wasn't already produced from Stage 2 v2 (per Phase 0 memo it was — `ac8948ae4b5be3e9`). With the freeze actually enforced, training reuses the existing cache as-is.

(b) **Train Stage 3 with the encoder fully frozen using the existing Stage 2 v2 cache.** Functionally identical to (a) if the existing cache was built from Stage 2 v2, which the Phase 0 memo asserts. This is the cheapest path: fix the code, kick off the retrain.

(c) **Train Stage 3 with the encoder unfrozen and the LIVE encoder (no cache).** This is the alternative training contract — useful if the underlying intuition is "the decoder benefits from co-adapting the encoder." If chosen, the config docstring and checkpoint filename must be updated to remove the `frozen-encoder` framing and the cached-features dataloader path retired. Expect a substantially longer wall-clock time per step.

**Recommended path:** start with (b), since the cache hash already corresponds to a Stage-2-v2-derived feature set per Phase 0. If (b) reproduces the eval gap, then either the freeze isn't the only issue (run A3 + Phase B) or the architecture genuinely doesn't generalize from this corpus mix and an architecture/data sub-project is warranted.

**Re-evaluation after retrain.** Re-run both `eval/run_clarity_demo_radio_eval.py` (4-piece HF demo) and the 50-piece lieder ship-gate evaluation from Subproject 4. Move on from this audit only when A2 passes on the retrained checkpoint (max_abs_diff <= 0.01) and the demo mean onset_f1 exceeds the 0.241 decision-gate threshold — or when both A2 passes and demo eval still disappoints, in which case Phase B becomes the right next step.

## Out of scope

- Architecture comparison (DaViT vs RADIO) — separate parallel thread.
- Audiveris benchmark — separate parallel thread.
- Stage 1 / Stage 2 training audit — these would only matter if A2 had passed and the upstream encoder were implicated by Phase B; A2 places the bug squarely in Stage 3's training-loop construction.
- The actual fix and the retrain — out of scope for this audit. This report is diagnostic; the fix lives in a follow-on sub-project.
