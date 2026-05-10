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
