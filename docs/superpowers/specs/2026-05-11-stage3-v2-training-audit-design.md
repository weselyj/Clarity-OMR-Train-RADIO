# Stage 3 v2 Training Audit

**Status:** design approved 2026-05-11 — ready for implementation plan
**Branch off:** `main` at HEAD `05b9862` (post-PR #46) or `fix/first-emission-key-time` after PR #47 merges
**Predecessor:** [`2026-05-11-stage-d-part-alignment-fix.md`](../../archive/handoffs/2026-05-11-stage-d-part-alignment-fix.md) — handoff that exposed the 0.054 mean onset_f1 on the 4-piece HF demo eval and the model-emits-4/4-for-everything failure mode

## Goal

Verify the Stage 3 v2 training was sound. Decide between two competing explanations for the gap between val_loss 0.148 and downstream mean onset_f1 ≈ 0.06:

1. **The foundation is sound, the gap is genuine generalization failure.** Closing the gap requires a different training recipe (rebalanced sampling, larger image height, auxiliary key/time-sig head, etc.) — but the trainer itself is not broken.
2. **The foundation has a bug.** The model isn't "the best it can be given its training" because the training itself is corrupted (preprocessing skew, vocab mismatch, loss bug, encoder cache divergence, etc.). Retraining without fixing this would reproduce the same gap.

The audit's output is a written report that picks one of those branches and provides the evidence. The fix or retrain that follows is a separate sub-project.

## Why now

- The user invested 4+ months in the RADIO architecture (Stage 1 v2 → Stage 2 v2 → Stage 3 v2). The Stage 3 v2 checkpoint produces poor downstream metrics despite reasonable val_loss.
- A memory entry from the Stage 2 retrain decision ("audit trainer against PyTorch tuning guide") flagged this audit as a prerequisite for any future training run. It was deferred and never executed.
- The just-merged `fix/first-emission-key-time` patch (PR #47) addressed one failure mode at the pipeline level. The deeper failure modes (25-50% note under-generation per piece, wrong key signatures on 2/4 demo pieces, low pitch precision) are likely model-level. Before committing to any retrain, we need to know whether the trainer is sound.

## Non-goals

- **Not closing the gap to the 0.241 decision gate.** The audit only diagnoses; the corrective action follows.
- **Not auditing Stage 1 or Stage 2 training initially.** Those produced acceptable val_loss and are downstream of Stage 3 v2's failure. We expand scope only if A surfaces evidence pointing earlier.
- **Not evaluating architecture choices.** RADIO vs. DaViT, per-system vs. per-staff, image height, beam width — all out of scope. Those are valid questions for the post-audit decision.
- **Not running DaViT baseline reproduction or Audiveris benchmark.** Both are valuable parallel threads but logically independent of this audit.

## Scope

| Layer | In scope? |
|---|---|
| Stage 3 v2 trainer code | yes |
| Stage 3 v2 dataloader + augmentation | yes |
| Stage 3 v2 loss function | yes |
| Stage 3 v2 optimizer + LR schedule | yes |
| Encoder feature cache (built Phase 0, 2026-05-09) | yes — its parity with live encoder is a known risk |
| Stage 3 v2 checkpoint loading | yes |
| Stage 1 / Stage 2 trainers | no (unless A surfaces evidence) |
| Stage 4+ retraining design | no (output of audit triggers separate sub-project) |
| Architecture comparison (DaViT, per-staff) | no |

## Architecture

Two phases, sequential, with a fast exit on the first phase if it surfaces a clear bug.

### Phase A — Round-trip data verification (cheap, narrow, high-information)

For ~20 training samples spread across all 4 corpora (synthetic_v2, grandstaff, primus, camera-primus), run three sub-experiments. Each tests parity between training and inference at a different layer.

| Sub-experiment | What it tests | Pass criterion |
|---|---|---|
| **A1: Image preprocessing parity** | Image → tensor parity between training-time and inference-time preprocessing pipelines | Tensors bit-identical (or within fp32 noise on stochastic augmentations) |
| **A2: Encoder output parity** | Stage 3 trained on cached encoder features; inference runs encoder live. Compare cached feature tensor vs live encoder output for the same image | Features within fp16 precision noise (~1e-3 L∞), no systematic bias |
| **A3: Decoder behavior on training data** | End-to-end inference on training samples should approximately reproduce the labels (the model has seen them) | Token accuracy ≥ 80% on training data; exact-match rate ≥ 30% on simple corpora |

**Fast exit:** any sub-experiment failing identifies a concrete bug. Stop Phase A, fix that bug, re-evaluate `eval/run_clarity_demo_radio_eval.py` to see if the fix alone closes much of the gap. Only proceed to Phase B if Phase A passes cleanly OR if the user explicitly wants thoroughness.

### Phase B — Process audit against PyTorch tuning guide (broader, deeper, conditional)

Only run if Phase A passes cleanly. Walk every training component end-to-end with the PyTorch performance tuning guide open. Component checklist:

| Component | What to verify |
|---|---|
| Dataloader | `num_workers > 0`, `pin_memory=True`, `persistent_workers=True`, shuffling on train but not val, deterministic seeding, no leak from val into train manifest |
| Augmentation | Train-only application, augmentations don't destroy information (no crops that hide time-sig glyph), random seed handling |
| Model architecture | Causal mask on decoder, positional encoding correct, vocab size matches embedding/output, encoder→decoder cross-attention wiring |
| Loss function | Standard CE, padding excluded via `ignore_index`, no accidental weighting that down-weights rare tokens, label smoothing usage matches design intent |
| Optimizer + LR schedule | AdamW with betas (0.9, 0.999), weight decay ~0.01-0.1 (transformer-appropriate), LR within 1e-4 to 5e-4 for decoder fine-tune, warmup, cosine/linear decay (not LR=0 by end) |
| Gradient handling | Clipping in place (typical `max_norm=1.0`), gradient accumulation scales loss correctly, training logs free of NaN/Inf gradient warnings |
| Mixed precision | FP16/BF16 setup consistent between train and inference, `GradScaler` used correctly if FP16 |
| Checkpoint integrity | `_best.pt` corresponds to claimed step and val_loss; load restores all required state (model weights; optimizer/scheduler state if resume was intended) |
| Validation loop | `val_loss` computed identically to `train_loss` (same masking, same reduction). val data truly held out from train manifest |

For each component: read code top-to-bottom, cross-reference against the PyTorch tuning guide, flag deviations with code refs. Spot-check loss curves and gradient norms from training logs if available.

## Deliverable

Single audit report at `docs/audits/2026-05-11-stage3-v2-training-audit.md` containing:

1. **A1 / A2 / A3 results** — pass/fail per sub-experiment, with quantitative evidence (token accuracy numbers, tensor diffs, per-class accuracy on time-sig / key-sig / note tokens).
2. **Phase B checklist results** (if reached) — pass/fail per component with code references.
3. **Bug list** — concrete issues found, ranked by likely impact on the train/eval gap (high / medium / low).
4. **Recommendation** — exactly one of three branches:
   - *"Foundation is sound, gap is genuine generalization failure"* → triggers a post-audit retraining-strategy sub-project (rebalancing, aux heads, larger image height, etc.).
   - *"Foundation has bug X; fix and re-evaluate before any retraining decision"* → triggers a fix + re-evaluation loop without retraining.
   - *"Foundation has bug X requiring re-training to validate"* → triggers a fix + targeted re-training run.

## Risks and decisions made

- **Risk:** Phase A passes cleanly but Phase B doesn't reveal anything either. We'd then have spent ~1 week and still not know what's wrong.
  **Mitigation:** the audit's recommendation in this case is "foundation is sound" — that itself is valuable, even if no bug is found. It de-risks the retraining decision.
- **Risk:** Encoder cache divergence (A2 fails) implies we need to rebuild the 1.38 TB cache before re-training. That's expensive.
  **Mitigation:** if A2 fails, the audit recommendation captures the cache rebuild cost in scope. Caller decides whether to rebuild or to re-train without caching.
- **Risk:** Phase A's "training data should approximately memorize" assumption may be too strong if training used heavy augmentation. A3 thresholds (80% token accuracy, 30% exact-match) are guesses.
  **Mitigation:** thresholds are intended as triage, not pass/fail certification. If A3 falls between (e.g., 60% accuracy), the audit reports the actual numbers and the recommendation contextualizes them rather than mechanically applying a threshold.

## Effort estimate

| Phase | Time |
|---|---|
| Phase A (3 sub-experiments) | 1-2 days |
| Phase B (conditional, full process audit) | 3-5 days |
| Report writeup | 0.5 day |
| **Total** | **1.5-7.5 days** depending on findings |

## What follows the audit

The audit's `Recommendation` field triggers exactly one of:

- **Retraining-strategy sub-project** (foundation sound) — separate spec with rebalanced sampling / aux head / image-height variations as candidate interventions.
- **Fix-and-verify loop** (concrete bug) — fix is committed, full demo eval re-run; if scores rise meaningfully, ship; if not, re-enter audit for the next layer.
- **Fix + retraining sub-project** (bug requires retraining to validate) — fix committed, then a single retraining run with the corrected pipeline; eval against demo + lieder corpora.

In all three cases, DaViT baseline reproduction and Audiveris benchmark remain available as parallel diagnostic threads. They do not block this audit.
