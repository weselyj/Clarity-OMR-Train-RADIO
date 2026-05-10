# Stage 3 Phase 1 — Launch Handoff (Pre-Flight)

> Plan C is implementation-complete. This handoff captures the launch checklist for the user to review before training begins.

## TL;DR

- Branch `feat/stage3-phase1-training` is ready to merge or train-from.
- All Phase 1 trainer code + configs land in this branch.
- Pre-flight script: `scripts/preflight_stage3_phase1.py`. Run on GPU box; expect "READY".
- Launch command: see "Run the trainer" below.
- Step target: 4500 opt-steps. Manual extension gates at 4500 -> 6000 -> 7500.

## Pre-flight checklist (must hold before saying "go")

1. [ ] `feat/stage3-phase1-training` branch is up to date and synced to GPU box at `10.10.1.29`.
2. [ ] On GPU box, run: `venv-cu132\Scripts\python scripts\preflight_stage3_phase1.py --dry-run` and confirm exit 0.
3. [ ] Cache directory `data/cache/encoder/ac8948ae4b5be3e9/` exists with `samples_processed=215985`.
4. [ ] Manifest `src/data/manifests/token_manifest_stage3.jsonl` exists with 303,663 rows.
5. [ ] Init checkpoint `checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt` exists.
6. [ ] Disk has >= 50 GB free for `checkpoints/full_radio_stage3_v1/` (~ 25 GB best.pt + step ckpts).
7. [ ] No active GPU jobs on the box (check `nvidia-smi`).

## Run the trainer

```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python -u src/train/train.py --stage-configs configs/train_stage3_radio_systems.yaml --mode execute --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/full_radio_stage3_v1 --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v1_steps.jsonl'
```

Wall-time projection: 1.5-3h to opt-step 4500 on the RTX 5090 (per spec).

## Monitor

Tail the step-log; expect every 500 opt-steps to write a row with `val_loss`, `val_loss_per_dataset`, and `wall_time_s`.

```bash
ssh 10.10.1.29 'powershell.exe -Command "Get-Content -Wait logs/full_radio_stage3_v1_steps.jsonl"'
```

Watch for:
- Sanity halt: `[train] HALT (sanity): val_loss > 5.0` in first 200 steps OR NaN at any time.
- Per-dataset val_loss divergence: cameraprimus_systems > 1.5 x Stage 2 v2's analog signal = early warning.
- VRAM: `nvidia-smi --query-gpu=memory.used --format=csv -l 30 -f vram.csv` (expect ~47% used).

## At opt-step 4500: extension decision

Per spec lines 201-210, pause and review the val_loss curve over the last 750 opt-steps:
- Still descending -> extend to 6000.
- Plateaued or regressed -> finalize at 4500.
- At 6000, repeat for 7500 cap.

## Phase 1 -> Phase 2 gate

After best.pt is finalized, run Plan D (Phase 2 eval). Phase 1 only ends when all five exit criteria hold (see plan section "Phase 1 Exit Criteria").

## What goes in the post-training handoff

- Final `_best.pt` opt-step + per-dataset val_loss values
- Wall time + VRAM peak
- Whether sanity halt fired (and why if so)
- Whether step-extension protocol triggered
- All step-log rows compressed into a summary table
