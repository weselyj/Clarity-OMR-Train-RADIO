# Stage 3 v4 Results

Tracking ground truth for the Phase 0 gates and (eventually) Phase 3 outcomes.
See [spec](../superpowers/specs/2026-05-13-stage3-v4-scan-realistic-retrain-design.md) for the project framing.

## Phase 0a — Bottom-quartile lieder cluster (2026-05-13)

**Strict gate: FAIL** — `bass-clef-misread` came in at 4/33 (12.1%), below the 30% threshold.

**Reframed: PROCEED.** The dominant cluster was `phantom-staff-residual` at 25/33 (75.8%), which shares the same root cause (decoder confused about staff structure on multi-staff piano with no scan-realistic training exposure). Predicted MXLs on flagged pieces have 6 parts where GT has 3 — assembly striping the decoder's confused staff-count output into double-the-parts. The unified "decoder structural confusion" cluster is ~85% of the bottom quartile. The retrain hypothesis (`scanned_grandstaff_systems`) targets the same root cause.

See [bottom-quartile cluster report](2026-05-13-bottom-quartile-lieder-cluster.md) and the [spec's Phase 0a addendum](../superpowers/specs/2026-05-13-stage3-v4-scan-realistic-retrain-design.md#phase-0a-addendum-2026-05-13-run-result).

## Phase 0b — Overfit smoke (2026-05-13)

**Gate PASS.** Pipeline can overfit a 20-sample clean grand-staff subset.

| step | train_loss | val_loss | train gate < 0.05 | val gate < 0.10 |
|---:|---:|---:|---|---|
|  50 | 0.5094 | 0.5191 | — | — |
| 100 | 0.1170 | 0.0556 | — | ✓ |
| 150 | 0.0656 | 0.0223 | — | ✓ |
| 200 | 0.0262 | 0.0200 | ✓ | ✓ |
| 250 | 0.0120 | 0.0124 | ✓ | ✓ |
| 300 | 0.0167 | 0.0138 | ✓ | ✓ |
| 400 | 0.0109 | 0.0121 | ✓ | ✓ |
| **500** | **0.0061** | **0.0129** | **✓** | **✓** |
| 600 | 0.0157 | 0.0125 | ✓ | ✓ |

Training: 625 opt-steps, batch_size 1 with grad_accum 4 (effective batch 4), max_sequence_length 512, lr_dora 0.002, cosine schedule, no augmentation, no label smoothing, no contour-loss weighting. Init from `checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt`. Wall-clock on RTX 5090: ~9 min.

Config divergence from spec: original spec specified `batch_size: 4` + `max_sequence_length: 1024` + `tier_grouped_sampling: false`. First attempt OOM'd on the live C-RADIO-H encoder (no cache); reduced to `batch_size: 1` + `max_sequence_length: 512` + `grad_accumulation_steps: 4` for an equivalent effective batch. Spec gate criteria unchanged.

Step log: `logs/full_radio_stage3_v4_overfit_smoke_steps.jsonl` (also on seder at the same path).

**Implication for Phase 1+:** training infrastructure is healthy and the live-encoder data path works end-to-end. No blocker for Phase 1.

## Phase 1 — Build scanned_grandstaff_systems corpus

_Not yet executed. See plan Tasks 6-12._

## Phase 2 — v4 retrain

_Not yet executed. See plan Tasks 13-15._

## Phase 3 — Evaluation

_Not yet executed. See plan Tasks 16-19._
