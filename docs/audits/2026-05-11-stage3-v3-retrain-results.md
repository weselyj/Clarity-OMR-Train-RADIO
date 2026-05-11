# Stage 3 v3 Retrain Results

**Date:** 2026-05-11
**Spec:** [docs/superpowers/specs/2026-05-11-stage3-v3-retrain-design.md](../superpowers/specs/2026-05-11-stage3-v3-retrain-design.md)
**Plan:** [docs/superpowers/plans/2026-05-11-stage3-v3-retrain-plan.md](../superpowers/plans/2026-05-11-stage3-v3-retrain-plan.md)
**Status:** in progress

## v2 `_best.pt` discrepancy investigation (Task 1)

The discrepancy between the Stage 3 v2 audit header ("step 4000, val_loss 0.148") and the step log ("best at step 5500 = 0.164") is fully explained: the audit doc header contained a data-entry error. The checkpoint itself (`stage3-radio-systems-frozen-encoder_best.pt`) reports `global_step: 5500`, `best_val_loss: 0.1642527211671337` — matching the step log exactly. Weight comparison across three sampled tensors (encoder LoRA A/B weights and magnitude vectors) shows zero max-absolute-difference against `_step_0005500.pt` and nonzero differences (up to 0.012) against `_step_0004000.pt`, confirming that `_best.pt` holds step 5500's weights and never contained step 4000's weights. The trainer's save-best logic (`train.py:3041-3065`) is correct: it compares `current_val < best_val_loss` (strict less-than, against `val_loss` from the validation result dict), updates `best_val_loss` before writing, and saves immediately — no off-by-one, no wrong metric, no stale comparison. The `_save_checkpoint` function (`train.py:1660-1679`) writes `global_step` and `best_val_loss` at save time, so metadata in `_best.pt` accurately reflects the step at which it was written. Conclusion: the save-best logic is correct, the metadata in `_best.pt` is accurate, and v3's `_best.pt` will be trustworthy. The only error was in the audit report's header line, which stated step 4000 / 0.148 rather than step 5500 / 0.164.

## Phase 1 — Diagnostic (frankenstein checkpoint)

**Checkpoint build:** The frankenstein script ran cleanly against Stage 2 v2 `_best.pt` (step 4000) and Stage 3 v2 `_best.pt` (step 5500). Both checkpoints had exactly 1342 keys. The merge took 774 encoder keys from Stage 2 v2 and 568 non-encoder keys from Stage 3 v2; 0 encoder keys were missing in Stage 2, 0 encoder keys were exclusive to Stage 2, yielding an encoder-key mismatch rate of **0.00%** — the experiment is not confounded. The frankenstein checkpoint was written to `checkpoints/_frankenstein_s2enc_s3dec.pt` on seder (gitignored).

**Loader fix:** The eval driver's `src/checkpoint_io.py` initially failed to load the frankenstein because it looked only for `model_state_dict` as the top-level key, but the frankenstein script writes the state dict under `model`. A one-line fix to the loader (`payload.get("model_state_dict", payload.get("model", payload))`) resolved this and was deployed to seder before re-running the eval. The fixed `checkpoint_io.py` is committed alongside this report.

**Demo eval results (frankenstein: S2 v2 encoder + S3 v2 decoder):**

| Piece | onset_f1 | note_f1 |
|---|---|---|
| clair-de-lune-debussy | 0.05258 | 0.01363 |
| fugue-no-2-bwv-847-in-c-minor | 0.07051 | 0.02712 |
| gnossienne-no-1 | 0.07809 | 0.03688 |
| prelude-in-d-flat-major-op31-no1-scriabin | 0.03834 | 0.00613 |
| **mean** | **0.0599** | **0.0209** |

**Comparison against Stage 3 v2 post-first-emission baseline (PR #47):**

| Piece | S3 v2 baseline | Frankenstein | Delta |
|---|---|---|---|
| clair | 0.0340 | 0.0526 | +0.019 |
| fugue | 0.0631 | 0.0705 | +0.007 |
| gnoss | 0.1032 | 0.0781 | **-0.025** |
| prelude | 0.0352 | 0.0383 | +0.003 |
| mean | 0.0589 | 0.0599 | +0.001 |

**Gate decision: STOP_HALT.**

The frankenstein mean onset_f1 of 0.0599 falls below the 0.10 gate threshold, triggering the "diagnosis incomplete" stop. The improvement over the Stage 3 v2 baseline is +0.001 — within noise, and not the meaningful uplift (>>0.06, target ≥0.15) that would confirm encoder drift as the dominant failure mode. Gnossienne actually regressed 0.025 onset_f1 with the S2 encoder. The hypothesis that the Stage 3 v2 training bug (encoder DoRA adapters not frozen) caused the performance gap is **not confirmed** by this experiment.

**Conclusion:** The retrain plan (v3, 9000 steps) is NOT justified by this evidence. There are failure modes beyond encoder drift. Do NOT proceed to Phase 2/3 (code fix + retrain) without a follow-up investigation to identify what is actually causing the low onset_f1 scores. Opening a sub-project investigation is recommended before committing 8h of GPU time and several hours of human work to a retrain that the diagnostic evidence does not support.

## Phase 2 — Code fix

<TASK 4 SUMMARY GOES HERE>

## Phase 3 — Retrain (v3, 9000 steps)

<TASK 7 SUMMARY GOES HERE>

## Phase 4 — Re-evaluation

<TASK 8 RESULTS GO HERE>

## Verdict

<TASK 9 CONTENT GOES HERE>
