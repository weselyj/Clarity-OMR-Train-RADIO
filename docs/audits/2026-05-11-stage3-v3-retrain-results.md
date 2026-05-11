# Stage 3 v3 Retrain Results

**Date:** 2026-05-11
**Spec:** [docs/superpowers/specs/2026-05-11-stage3-v3-retrain-design.md](../superpowers/specs/2026-05-11-stage3-v3-retrain-design.md)
**Plan:** [docs/superpowers/plans/2026-05-11-stage3-v3-retrain-plan.md](../superpowers/plans/2026-05-11-stage3-v3-retrain-plan.md)
**Status:** in progress

## v2 `_best.pt` discrepancy investigation (Task 1)

The discrepancy between the Stage 3 v2 audit header ("step 4000, val_loss 0.148") and the step log ("best at step 5500 = 0.164") is fully explained: the audit doc header contained a data-entry error. The checkpoint itself (`stage3-radio-systems-frozen-encoder_best.pt`) reports `global_step: 5500`, `best_val_loss: 0.1642527211671337` — matching the step log exactly. Weight comparison across three sampled tensors (encoder LoRA A/B weights and magnitude vectors) shows zero max-absolute-difference against `_step_0005500.pt` and nonzero differences (up to 0.012) against `_step_0004000.pt`, confirming that `_best.pt` holds step 5500's weights and never contained step 4000's weights. The trainer's save-best logic (`train.py:3041-3065`) is correct: it compares `current_val < best_val_loss` (strict less-than, against `val_loss` from the validation result dict), updates `best_val_loss` before writing, and saves immediately — no off-by-one, no wrong metric, no stale comparison. The `_save_checkpoint` function (`train.py:1660-1679`) writes `global_step` and `best_val_loss` at save time, so metadata in `_best.pt` accurately reflects the step at which it was written. Conclusion: the save-best logic is correct, the metadata in `_best.pt` is accurate, and v3's `_best.pt` will be trustworthy. The only error was in the audit report's header line, which stated step 4000 / 0.148 rather than step 5500 / 0.164.

## Phase 1 — Diagnostic (frankenstein checkpoint)

<TASK 3 RESULTS GO HERE>

## Phase 2 — Code fix

<TASK 4 SUMMARY GOES HERE>

## Phase 3 — Retrain (v3, 9000 steps)

<TASK 7 SUMMARY GOES HERE>

## Phase 4 — Re-evaluation

<TASK 8 RESULTS GO HERE>

## Verdict

<TASK 9 CONTENT GOES HERE>
