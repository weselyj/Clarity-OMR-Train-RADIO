# Archived Stage 2 Experiment Configs

These are historical Stage 2 variant configs from the cu132 rollout and DoRA tuning phase.
They were active during the iterative search for stable hyperparameters and the first
working RADIO Stage 2 checkpoint (see `train_stage2_radio_mvp.yaml`).

**Canonical Stage 2 config:** `configs/train_stage2_radio.yaml`

These files are kept in git history for reproducibility but are not part of the current
training pipeline. No launcher or test loads them at runtime except `scripts/mvp_inner.ps1`
(historical MVP launcher, retained for reference).
