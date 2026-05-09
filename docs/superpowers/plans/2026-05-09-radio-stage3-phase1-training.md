# Stage 3 Phase 1 — Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Stage 3 (frozen-encoder, two-tier dataloader) end-to-end on the RTX 5090 GPU box and produce a `_best.pt` checkpoint that meets all five Phase 1 → Phase 2 gates: per-dataset val_loss regression floors hold, sanity halt did not fire, MusicXML validity rate eval driver runs successfully, training reached at least 4500 opt-steps, and the run handed off cleanly to Plan D (Phase 2 evaluation).

**Architecture:** Frozen encoder (RADIO C-RADIO v4-H + encoder-side DoRA, loaded from Stage 2 v2 `_best.pt` via the DoRA-aware loader). Trainable surface = decoder + cross-attention + LM head + positional_bridge. Two-tier dataloader: cached tier (`b_cached=16`, `grad_accum_cached=1`) reads pre-computed bf16 encoder features from `data/cache/encoder/ac8948ae4b5be3e9/` (215,985 samples across synthetic_systems / grandstaff_systems / primus_systems); live tier (`b_live=2`, `grad_accum_live=8`) processes cameraprimus_systems with full augmentation. Tier-grouped sampler emits batches in tier-pure runs — cached batches as singleton opt-steps, live batches in contiguous 8-batch opt-step blocks — so the trainer never sees an interrupted accumulation window. Per-dataset val_loss is tracked separately for synthetic_systems / grandstaff_systems / primus / cameraprimus. MusicXML validity rate is enabled in the eval driver (Stage 2 v2 left it at None). Step target: 4500 opt-steps initially, with manual extension gates at 4500 → 6000 → 7500 based on val_loss curve.

**Tech Stack:** Python 3.14, PyTorch (bf16 autocast + flash-attention), pytest, YAML configs, SSH to GPU box at `10.10.1.29` (Windows host, `venv-cu132\Scripts\python`).

---

## Decisions locked at plan time

1. **"90/10 data mix" interpretation = opt-step ratio, not batch ratio.** Spec §"Sampler architecture" line 178 says "Sampler interleaves cached and live batches in proportion to the data mix (90/10), not within batches." Two readings exist: (a) 9 out of 10 emitted batches are cached, or (b) 90% of opt-step gradient signal is cached. Reading (a) gives only ~61 live opt-steps over 4500 (≈8% of cameraprimus seen once); reading (b) gives 450 live opt-steps (~7,200 cameraprimus samples seen). Reading (b) is consistent with the spec's intent — "data mix" is a training-signal concept, not a per-emission concept. **Lock: 90% of opt-steps are cached, 10% are live.** Math: with `b_cached=16`, `b_live=2`, `grad_accum_live=8`, achieving 90% opt-step ratio = batch-ratio `cached_batch_ratio = (0.9/16) / ((0.9/16) + (0.1/2)) ≈ 0.529`. Total batches for 4500 opt-steps = `4500 / (0.529 + 0.471/8) ≈ 7653`: 4050 cached batches/opt-steps + 3600 live batches → 450 live opt-steps.

2. **Tier-grouped sampler emits live batches in contiguous 8-blocks.** Refactor `build_tier_grouped_sampler` to take `(n_cached_opt_steps, n_live_opt_steps, grad_accum_cached, grad_accum_live)` directly (replacing the current `cached_ratio`/`total_batches` pair). For each scheduled "live opt-step," the sampler emits `grad_accum_live` consecutive live batches; cached opt-steps remain singletons. The randomized tier sequence has length `n_cached_opt_steps + n_live_opt_steps` and is expanded to per-batch granularity at the end. **Why:** prevents "interrupted live accumulation" where a cached batch arrives mid-window with stale partial grads — the trainer can clear `optimizer.zero_grad()` at every tier transition without losing in-progress accumulation. Backward compat: keep the old positional-arg form available for existing tier-sampler tests (Task 3 covers the API shim).

3. **`StageTrainingConfig` gains optional tier-aware fields; `tier_grouped_sampling=True` toggles the new code path.** Adding 7 fields (all `Optional`, default `None` so legacy YAMLs keep working): `b_cached`, `b_live`, `grad_accumulation_steps_cached`, `grad_accumulation_steps_live`, `cached_data_ratio`, `cache_root`, `cache_hash16`, plus a sentinel bool `tier_grouped_sampling`. The new path engages only when `tier_grouped_sampling=True`. **Why:** the existing Stage 1 / Stage 2 v2 configs and tests must keep passing with a single trainer codebase — new mode is opt-in per stage.

4. **Per-dataset val_loss = 4 disjoint passes over a separate per-dataset val_loader, not a single mixed pass.** Each pass uses the tier-aware code path and reports `{dataset_name: val_loss}` in the validation result dict. The aggregate `val_loss` is a sample-weighted mean (proportional to cached_data_ratio components 70/10/10/10) so the existing best-tracking and sanity-halt logic see a single scalar. **Why:** spec line 216 requires per-data-source val_loss; mixed-tier sampling within a single eval pass would couple per-dataset numbers. Cost: 4× the validation walltime (~8s each at `validation_batches=2` so still negligible). The 4 datasets:
   - `synthetic_systems` (cached)
   - `grandstaff_systems` (cached)
   - `primus_systems` (cached)
   - `cameraprimus_systems` (live)

5. **Sanity halt fires on val_loss only (not per-step loss).** Spec line 226: "val_loss > 5.0 in first 200 steps OR NaN." First-200 check evaluates after the first validation pass at step 500 if val_loss > 5.0. NaN check fires at any validation pass. Both call `sys.exit(1)` after writing `[train] HALT: ...` to stdout and a structured row to the step-log JSONL. **Why:** existing trainer has per-micro-batch corruption detection (`accum_corruption`), but spec sanity halt is val_loss-based. Different signal — additive, not redundant.

6. **Sampler resume = deterministic re-derivation by seed.** The tier-grouped sampler is a pure function of `(entries, cached_datasets, live_datasets, n_cached_opt_steps, n_live_opt_steps, b_cached, b_live, grad_accum_cached, grad_accum_live, seed)`. Resume restores `last_batch_idx` from the checkpoint, rebuilds the full batch list with the same seed, and `next()`s past the consumed prefix. **Why:** simpler than serializing `random.Random` state; the determinism property already holds. Cost: O(total_batches) memory at resume (fine — list of int-lists is < 1 MB for 7653 batches). Test: build → consume N → save → rebuild → consume → assert prefix matches and remainder matches the original.

7. **`cached_features` in batch dict is renamed from `encoder_hidden` only at the model-forward boundary, not in the batch dict itself.** The dataset returns `batch_dict["encoder_hidden"]` (already in code at `train.py:670, 776`); the trainer remaps to `cached_features=...` when calling `model(...)`. **Why:** keeps the data layer's existing key untouched (collate_fn, dataset tests, sampler tests all reference `encoder_hidden`); the model API uses `cached_features` (its existing kwarg name from Phase 0 Task 3). Two names, one boundary — documented at the call site.

---

## Files to create or modify

**New files:**
- `configs/train_stage3_radio_systems.yaml` — Stage 3 trainer config (Task 0)
- `tests/train/test_stage_training_config_tier_fields.py` — config dataclass round-trip + YAML load (Task 1)
- `tests/train/test_train_loop_tier_dispatch.py` — train-loop micro-batch dispatch on `tier` key (Task 2)
- `tests/train/test_tier_grouped_sampler_opt_steps.py` — refactored sampler API (Task 3)
- `tests/train/test_train_loop_tier_grad_accum.py` — opt-step boundary semantics across tier transitions (Task 4)
- `tests/train/test_run_validation_tier_dispatch.py` — `_run_validation` tier-aware batch handling (Task 5)
- `tests/train/test_per_dataset_val_loss.py` — 4 disjoint dataset passes + sample-weighted aggregate (Task 6)
- `tests/train/test_sanity_halt.py` — val_loss > 5.0 in first 200 steps OR NaN (Task 7)
- `tests/train/test_sampler_resume.py` — last_batch_idx persistence + deterministic re-derivation (Task 8)
- `tests/train/test_stage2_v2_init_checkpoint_load.py` — DoRA-aware load smoke test on a synthetic ckpt (Task 9)
- `scripts/preflight_stage3_phase1.py` — pre-flight ready check (Task 11)
- `docs/superpowers/handoffs/2026-05-09-radio-stage3-phase1-launch-handoff.md` — pre-launch handoff (Task 11)
- `docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase1-complete-handoff.md` — final handoff (Task 14, date filled at completion)

**Modified files:**
- `src/train/train.py` — `StageTrainingConfig` fields (Task 1); `_run_stage` tier-aware batch dispatch + grad_accum (Tasks 2, 4); `_run_validation` tier-aware (Task 5); per-dataset val_loss (Task 6); sanity halt (Task 7); sampler resume (Task 8). Estimated edit envelope: ~250 LOC across the file, all in `_run_stage`, `_run_validation`, and `_save_checkpoint`/resume paths.
- `src/train/tier_sampler.py` — refactored API (Task 3).
- `src/eval/run_radio_eval.py` (or wherever the Stage 2 v2 eval driver lives — Task 10 will locate it) — enable MusicXML validity rate (Task 10).

---

## Phase 1 Exit Criteria

All five must hold before Phase 2 (Plan D) launches. Drawn from spec §"Inner gate (training-time)" lines 221–226 and §"Phase 1 → Phase 2 gate" lines 212–217.

1. **Training reached ≥ 4500 opt-steps** without uncaught exceptions and without sanity-halt firing on the val_loss curve.
2. **`_best.pt` was written** at some opt-step ≤ stage_total_steps, with `best_val_loss < 0.5` (a generous floor — Stage 2 v2 hit 0.148; ≥ 0.5 means the warm start failed catastrophically).
3. **Per-dataset val_loss regression floors all hold** at the best checkpoint:
   - `grandstaff_systems` ≤ Stage 2 v2's per-dataset val_loss × 1.10 (i.e. no more than 10% regression)
   - `primus_systems` ≤ Stage 2 v2's per-dataset val_loss × 1.10
   - `cameraprimus_systems` ≤ Stage 2 v2's per-dataset val_loss × 1.10
   - `synthetic_systems` ≤ 0.50 (it's the new bulk distribution; this is a "did the architectural bet take" floor)
   - **Note:** Stage 2 v2 did not track per-dataset val_loss. The Stage 2 v2 baselines are the per-dataset eval-driver quality numbers (96.8 / 93.4 / 83.1 / 75.2) that map to MusicXML quality, not val_loss. Plan D evaluates against those. For Phase 1's own gate, we measure that per-dataset val_loss did not diverge — sanity, not the production criterion.
4. **MusicXML validity rate ≥ 0.50** on the lieder eval set (per spec §2 line 264, Stage 2 v2 left this at None — Stage 3 enables and gates on it).
5. **Step-log telemetry is complete and parseable** — every `validate_every_steps` boundary has a JSONL row with `val_loss`, `val_loss_per_dataset`, `global_step`, and `wall_time_s`.

If criteria 1–5 all hold: hand off to Plan D (Phase 2 eval). If any fails: triage in `docs/superpowers/handoffs/` and return to user before retraining.

---

## Tasks

### Task 0: Branch + Stage 3 trainer config YAML

**Files:**
- Create: `configs/train_stage3_radio_systems.yaml`

**Why this task:** Establishes the Phase 1 feature branch and writes the trainer config that all subsequent code paths read. The YAML is the source of truth for tier sizes, step targets, init checkpoint, and cache pointers.

- [ ] **Step 1: Create feature branch from `main`**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git fetch origin && git checkout -b feat/stage3-phase1-training origin/main
```
Expected: `Switched to a new branch 'feat/stage3-phase1-training'`

- [ ] **Step 2: Create `configs/train_stage3_radio_systems.yaml`**

```yaml
# Stage 3 Phase 1 — frozen-encoder training on systems-aware multi-staff data.
#
# Resumes from Stage 2 v2 best.pt (val_loss 0.148, step 4000) with the encoder
# (and encoder-side DoRA adapters) frozen. Trainable surface: decoder +
# cross-attention + LM head + positional_bridge.
#
# Two-tier dataloader: 90% opt-step share on cached features (215,985 samples,
# read from data/cache/encoder/ac8948ae4b5be3e9/), 10% on live cameraprimus
# with augmentation.
#
# Run command (executed on GPU box at 10.10.1.29):
#   ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
#     venv-cu132\Scripts\python -u src/train/train.py \
#       --stage-configs configs/train_stage3_radio_systems.yaml \
#       --mode execute \
#       --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \
#       --start-stage stage3-radio-systems-frozen-encoder \
#       --checkpoint-dir checkpoints/full_radio_stage3_v1 \
#       --token-manifest src/data/manifests/token_manifest_stage3.jsonl \
#       --step-log logs/full_radio_stage3_v1_steps.jsonl'

stage_name: stage3-radio-systems-frozen-encoder
stage_b_encoder: radio_h

# Step target: 4500 opt-steps. Manual extension protocol at 4500 → 6000 → 7500
# (see spec lines 201–210).
epochs: 1
effective_samples_per_epoch: 122448  # 7653 batches * mean-batch-size 16  → covers 4500 opt-steps
batch_size: 1                        # SENTINEL — unused when tier_grouped_sampling=True
max_sequence_length: 512
grad_accumulation_steps: 1           # SENTINEL — unused when tier_grouped_sampling=True

# Tier-aware mode (Stage 3 only). When tier_grouped_sampling=true the trainer
# uses b_cached/b_live + grad_accumulation_steps_cached/grad_accumulation_steps_live
# in place of batch_size + grad_accumulation_steps.
tier_grouped_sampling: true
b_cached: 16
b_live: 2
grad_accumulation_steps_cached: 1
grad_accumulation_steps_live: 8
cached_data_ratio: 0.90              # 90% opt-step gradient signal on cached tier
cache_root: data/cache/encoder
cache_hash16: ac8948ae4b5be3e9

# Optimizer (matches Stage 2 v2 LR pattern; encoder-side DoRA is frozen, so
# lr_dora applies only to decoder-side adapters).
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

# dataset_mix is the per-dataset DRAW PROBABILITY on the cached side; the live
# side is implicitly 1.0 cameraprimus_systems. The cached_data_ratio above sets
# the cached:live opt-step balance globally.
#
# Cached-tier internal mix (sums to 1.0): synthetic 70/cached_total + grandstaff_systems 11.7 + primus 11.7
# Wait — actual cached-tier composition is set by the manifest weights, not these ratios.
# These ratios drive the WeightedRandomSampler entry weights for the cached-tier dataset only.
dataset_mix:
  - dataset: synthetic_systems
    ratio: 0.7777     # 70 / 90 within the cached tier
    split: train
    required: true
  - dataset: grandstaff_systems
    ratio: 0.1111     # 10 / 90
    split: train
    required: true
  - dataset: primus_systems
    ratio: 0.1111     # 10 / 90
    split: train
    required: true
  - dataset: cameraprimus_systems
    ratio: 0.0        # weight 0 within the cached weighting; live tier draws from this dataset directly
    split: train
    required: true
```

> **Note on `dataset_mix` semantics in tier-grouped mode:** In legacy mode the WeightedRandomSampler used `dataset_mix` ratios to weight individual entries. In tier-grouped mode the cached/live split is enforced at the batch-list level by `build_tier_grouped_sampler`; the `dataset_mix` ratios are applied only WITHIN the cached tier's index pool (cameraprimus_systems gets weight 0 there because it's the live tier's exclusive dataset). Within the live tier, all entries are drawn uniformly. This is an implementation detail of Task 4 — the YAML stays declarative.

- [ ] **Step 3: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add configs/train_stage3_radio_systems.yaml
git commit -m "feat(configs): add Stage 3 Phase 1 trainer config"
```

> **Review:** Confirm `cache_hash16` matches Phase 0 build (`ac8948ae4b5be3e9` per memory `project_radio_stage3_design.md` line 43). Confirm `b_cached=16` matches Phase 0d sweep recommendation. Confirm step target = 4500 opt-steps via the `epochs * effective_samples_per_epoch` math (computed in Task 4).

---

### Task 1: `StageTrainingConfig` tier-aware fields (TDD)

**Files:**
- Modify: `src/train/train.py:73-147` (the `StageTrainingConfig` dataclass + `load_stage_config` parser)
- Test: `tests/train/test_stage_training_config_tier_fields.py`

**Why this task:** Adds the 7 optional tier-aware fields and the `tier_grouped_sampling` toggle. Dataclass must round-trip cleanly from YAML and must validate that tier fields are all-set or all-None.

- [ ] **Step 1: Write failing tests**

```python
"""Tier-aware fields on StageTrainingConfig — round-trip and validation."""
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest


def test_legacy_yaml_loads_with_tier_fields_none():
    """Stage 1/2 YAML (no tier fields) loads with tier fields = None and tier_grouped_sampling=False."""
    from src.train.train import load_stage_config

    yaml_text = textwrap.dedent("""
        stage_name: stage2-test
        stage_b_encoder: radio_h
        epochs: 1
        effective_samples_per_epoch: 1000
        batch_size: 2
        grad_accumulation_steps: 8
        max_sequence_length: 512
        lr_dora: 0.0005
        lr_new_modules: 0.0003
        warmup_steps: 100
        schedule: cosine
        weight_decay: 0.01
        label_smoothing: 0.01
        contour_loss_weight: 0.01
        checkpoint_every_steps: 500
        validate_every_steps: 500
        dataset_mix:
          - dataset: grandstaff_systems
            ratio: 1.0
            split: train
            required: true
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = Path(fh.name)
    try:
        cfg = load_stage_config(path)
    finally:
        path.unlink()

    assert cfg.tier_grouped_sampling is False
    assert cfg.b_cached is None
    assert cfg.b_live is None
    assert cfg.grad_accumulation_steps_cached is None
    assert cfg.grad_accumulation_steps_live is None
    assert cfg.cached_data_ratio is None
    assert cfg.cache_root is None
    assert cfg.cache_hash16 is None


def test_stage3_yaml_loads_tier_fields():
    """Stage 3 YAML with tier_grouped_sampling=true populates all 7 tier fields."""
    from src.train.train import load_stage_config

    yaml_text = textwrap.dedent("""
        stage_name: stage3-test
        stage_b_encoder: radio_h
        epochs: 1
        effective_samples_per_epoch: 7653
        batch_size: 1
        grad_accumulation_steps: 1
        max_sequence_length: 512
        lr_dora: 0.0005
        lr_new_modules: 0.0003
        warmup_steps: 500
        schedule: cosine
        weight_decay: 0.01
        label_smoothing: 0.01
        contour_loss_weight: 0.01
        checkpoint_every_steps: 500
        validate_every_steps: 500
        tier_grouped_sampling: true
        b_cached: 16
        b_live: 2
        grad_accumulation_steps_cached: 1
        grad_accumulation_steps_live: 8
        cached_data_ratio: 0.9
        cache_root: data/cache/encoder
        cache_hash16: ac8948ae4b5be3e9
        dataset_mix:
          - dataset: synthetic_systems
            ratio: 0.7777
            split: train
            required: true
          - dataset: grandstaff_systems
            ratio: 0.1111
            split: train
            required: true
          - dataset: primus_systems
            ratio: 0.1111
            split: train
            required: true
          - dataset: cameraprimus_systems
            ratio: 0.0
            split: train
            required: true
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = Path(fh.name)
    try:
        cfg = load_stage_config(path)
    finally:
        path.unlink()

    assert cfg.tier_grouped_sampling is True
    assert cfg.b_cached == 16
    assert cfg.b_live == 2
    assert cfg.grad_accumulation_steps_cached == 1
    assert cfg.grad_accumulation_steps_live == 8
    assert cfg.cached_data_ratio == pytest.approx(0.9)
    assert cfg.cache_root == "data/cache/encoder"
    assert cfg.cache_hash16 == "ac8948ae4b5be3e9"


def test_tier_grouped_true_requires_all_tier_fields():
    """tier_grouped_sampling=true with missing tier fields raises ValueError."""
    from src.train.train import load_stage_config

    yaml_text = textwrap.dedent("""
        stage_name: stage3-broken
        stage_b_encoder: radio_h
        epochs: 1
        effective_samples_per_epoch: 1000
        batch_size: 1
        grad_accumulation_steps: 1
        max_sequence_length: 512
        lr_dora: 0.0005
        lr_new_modules: 0.0003
        warmup_steps: 100
        schedule: cosine
        weight_decay: 0.01
        label_smoothing: 0.01
        contour_loss_weight: 0.01
        checkpoint_every_steps: 500
        validate_every_steps: 500
        tier_grouped_sampling: true
        b_cached: 16
        # missing: b_live, grad_accum_*, cached_data_ratio, cache_root, cache_hash16
        dataset_mix:
          - dataset: synthetic_systems
            ratio: 1.0
            split: train
            required: true
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = Path(fh.name)
    try:
        with pytest.raises(ValueError, match="tier_grouped_sampling=true requires"):
            load_stage_config(path)
    finally:
        path.unlink()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_stage_training_config_tier_fields.py -v`
Expected: 3 failures (`AttributeError: 'StageTrainingConfig' object has no attribute 'tier_grouped_sampling'` or similar).

- [ ] **Step 3: Add tier-aware fields to `StageTrainingConfig`**

Edit `src/train/train.py` lines 73–93. Add 8 fields after `stage_b_encoder`:

```python
@dataclass(frozen=True)
class StageTrainingConfig:
    stage_name: str
    epochs: int
    effective_samples_per_epoch: int
    batch_size: int
    max_sequence_length: int
    lr_dora: float
    lr_new_modules: float
    warmup_steps: int
    schedule: str
    label_smoothing: float
    contour_loss_weight: float
    weight_decay: float
    checkpoint_every_steps: int
    validate_every_steps: int
    grad_accumulation_steps: int
    loraplus_lr_ratio: float
    dataset_mix: Tuple[DatasetMix, ...]
    stage_b_encoder: str = "davit"
    # --- Stage 3 tier-aware fields (all None for legacy stages) ---
    tier_grouped_sampling: bool = False
    b_cached: Optional[int] = None
    b_live: Optional[int] = None
    grad_accumulation_steps_cached: Optional[int] = None
    grad_accumulation_steps_live: Optional[int] = None
    cached_data_ratio: Optional[float] = None
    cache_root: Optional[str] = None
    cache_hash16: Optional[str] = None
```

- [ ] **Step 4: Extend `load_stage_config` parser**

Edit `src/train/train.py:128-147` (inside `load_stage_config`). Replace the `return StageTrainingConfig(...)` block with:

```python
    tier_grouped = bool(raw.get("tier_grouped_sampling", False))
    tier_field_names = (
        "b_cached", "b_live",
        "grad_accumulation_steps_cached", "grad_accumulation_steps_live",
        "cached_data_ratio", "cache_root", "cache_hash16",
    )
    tier_field_values = {name: raw.get(name) for name in tier_field_names}
    if tier_grouped:
        missing = [n for n, v in tier_field_values.items() if v is None]
        if missing:
            raise ValueError(
                f"tier_grouped_sampling=true requires all tier fields to be set in {path}; "
                f"missing: {missing}"
            )

    return StageTrainingConfig(
        stage_name=str(raw["stage_name"]),
        epochs=int(raw["epochs"]),
        effective_samples_per_epoch=int(raw["effective_samples_per_epoch"]),
        batch_size=int(raw["batch_size"]),
        max_sequence_length=int(raw["max_sequence_length"]),
        lr_dora=float(raw["lr_dora"]),
        lr_new_modules=float(raw["lr_new_modules"]),
        warmup_steps=int(raw["warmup_steps"]),
        schedule=str(raw.get("schedule", "cosine")).lower(),
        label_smoothing=float(raw.get("label_smoothing", 0.0)),
        contour_loss_weight=float(raw.get("contour_loss_weight", 0.1)),
        weight_decay=max(0.0, float(raw.get("weight_decay", 0.01))),
        checkpoint_every_steps=max(1, int(raw.get("checkpoint_every_steps", 1000))),
        validate_every_steps=max(1, int(raw.get("validate_every_steps", 500))),
        grad_accumulation_steps=max(1, int(raw.get("grad_accumulation_steps", 1))),
        loraplus_lr_ratio=float(raw.get("loraplus_lr_ratio", 1.0)),
        dataset_mix=tuple(mix),
        stage_b_encoder=str(raw.get("stage_b_encoder", "davit")).lower().strip(),
        tier_grouped_sampling=tier_grouped,
        b_cached=int(tier_field_values["b_cached"]) if tier_field_values["b_cached"] is not None else None,
        b_live=int(tier_field_values["b_live"]) if tier_field_values["b_live"] is not None else None,
        grad_accumulation_steps_cached=int(tier_field_values["grad_accumulation_steps_cached"]) if tier_field_values["grad_accumulation_steps_cached"] is not None else None,
        grad_accumulation_steps_live=int(tier_field_values["grad_accumulation_steps_live"]) if tier_field_values["grad_accumulation_steps_live"] is not None else None,
        cached_data_ratio=float(tier_field_values["cached_data_ratio"]) if tier_field_values["cached_data_ratio"] is not None else None,
        cache_root=str(tier_field_values["cache_root"]) if tier_field_values["cache_root"] is not None else None,
        cache_hash16=str(tier_field_values["cache_hash16"]) if tier_field_values["cache_hash16"] is not None else None,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_stage_training_config_tier_fields.py -v`
Expected: 3 passes.

- [ ] **Step 6: Run full train-test suite to verify no regressions**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/ -q`
Expected: all tests pass (legacy configs still load).

- [ ] **Step 7: Commit**

```bash
git add src/train/train.py tests/train/test_stage_training_config_tier_fields.py
git commit -m "feat(train): add tier-aware fields to StageTrainingConfig"
```

> **Review:** Confirm new fields are all `Optional[...]` with default `None`, and that `tier_grouped_sampling: bool = False` is the toggle. Confirm legacy YAMLs (Stage 1, Stage 2 v2) still load — `pytest tests/train/` is the proxy.

---

### Task 2: Train loop tier-aware batch dispatch (TDD) — `images` vs `cached_features`

**Files:**
- Modify: `src/train/train.py:2189-2244` (the `_run_stage` per-step h2d + forward block)
- Test: `tests/train/test_train_loop_tier_dispatch.py`

**Why this task:** The current `_run_stage` accesses `_batch_dict["images"]` unconditionally (line 2218–2220). For cached batches the batch dict contains `encoder_hidden`, `_h16`, `_w16` instead — and the model takes `cached_features=...` not `pixel_values=...`. This task adds the dispatch.

- [ ] **Step 1: Write failing test**

```python
"""Train loop dispatches on batch_dict['tier'] when forwarding through the model."""
from __future__ import annotations
from unittest.mock import MagicMock

import torch
import pytest


def _make_cached_batch():
    """Mimics StageBDataset.collate_fn output for a cached-tier batch (b=2)."""
    return {
        "tier": "cached",
        "encoder_hidden": torch.zeros(2, 156, 1280, dtype=torch.bfloat16),
        "_h16": 16,
        "_w16": 156,
        "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
        "labels": torch.zeros(2, 511, dtype=torch.long),
        "contour_targets": torch.zeros(2, 32, dtype=torch.long),
    }


def _make_live_batch():
    """Mimics StageBDataset.collate_fn output for a live-tier batch (b=2)."""
    return {
        "tier": "live",
        "images": torch.zeros(2, 1, 250, 2500, dtype=torch.float32),
        "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
        "labels": torch.zeros(2, 511, dtype=torch.long),
        "contour_targets": torch.zeros(2, 32, dtype=torch.long),
        "content_widths": torch.tensor([2500, 2500], dtype=torch.long),
    }


def test_dispatch_cached_batch_calls_model_with_cached_features():
    from src.train.train import _forward_batch_for_train

    model = MagicMock()
    model.return_value = {
        "logits": torch.zeros(2, 511, 100),
        "contour_logits": torch.zeros(2, 3, 32),
    }
    device = torch.device("cpu")
    batch = _make_cached_batch()

    _forward_batch_for_train(model, batch, device, bf16_enabled=False, channels_last=False)

    args, kwargs = model.call_args
    assert "cached_features" in kwargs
    assert "pixel_values" not in kwargs
    assert kwargs["cached_features"].shape == (2, 156, 1280)
    assert kwargs["_h16"] == 16
    assert kwargs["_w16"] == 156


def test_dispatch_live_batch_calls_model_with_pixel_values():
    from src.train.train import _forward_batch_for_train

    model = MagicMock()
    model.return_value = {
        "logits": torch.zeros(2, 511, 100),
        "contour_logits": torch.zeros(2, 3, 32),
    }
    device = torch.device("cpu")
    batch = _make_live_batch()

    _forward_batch_for_train(model, batch, device, bf16_enabled=False, channels_last=False)

    args, kwargs = model.call_args
    assert "pixel_values" in kwargs
    assert "cached_features" not in kwargs
    assert kwargs["pixel_values"].shape == (2, 1, 250, 2500)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_train_loop_tier_dispatch.py -v`
Expected: 2 failures (`ImportError: cannot import name '_forward_batch_for_train'`).

- [ ] **Step 3: Add `_forward_batch_for_train` helper**

Insert into `src/train/train.py` immediately after `_run_validation` (around line 1328, before `_save_checkpoint`):

```python
def _forward_batch_for_train(
    model,
    batch_dict: "Dict[str, object]",
    device: "torch.device",
    *,
    bf16_enabled: bool,
    channels_last: bool,
) -> "Dict[str, object]":
    """Dispatch a batch through the model based on its tier.

    Returns the model output dict (logits + contour_logits + ...). Does NOT
    move decoder_inputs / labels / contour_targets to device — caller is
    responsible for that (kept here for symmetry with the existing _run_stage
    h2d block).

    For cached batches: passes ``cached_features=encoder_hidden``, ``_h16``, ``_w16``.
    For live batches: passes ``pixel_values=images``.
    """
    import torch as _torch

    tier = batch_dict.get("tier", "live")
    decoder_inputs = batch_dict["decoder_inputs"].to(device, non_blocking=True)
    if tier == "cached":
        cached_features = batch_dict["encoder_hidden"].to(device, non_blocking=True)
        with _torch.autocast(
            device_type=device.type,
            dtype=_torch.bfloat16,
            enabled=bf16_enabled,
        ):
            outputs = model(
                cached_features=cached_features,
                input_ids=decoder_inputs,
                _h16=int(batch_dict["_h16"]),
                _w16=int(batch_dict["_w16"]),
                return_aux=True,
            )
    else:
        if channels_last:
            images = batch_dict["images"].to(device, non_blocking=True, memory_format=_torch.channels_last)
        else:
            images = batch_dict["images"].to(device, non_blocking=True)
        with _torch.autocast(
            device_type=device.type,
            dtype=_torch.bfloat16,
            enabled=bf16_enabled,
        ):
            outputs = model(pixel_values=images, input_ids=decoder_inputs, return_aux=True)
    return outputs
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_train_loop_tier_dispatch.py -v`
Expected: 2 passes.

- [ ] **Step 5: Wire `_forward_batch_for_train` into `_run_stage`**

Edit `src/train/train.py:2216-2244`. Replace the explicit `images = _batch_dict["images"].to(...)` h2d + autocast + `model(pixel_values=...)` block with a call to the helper. The block currently looks like:

```python
                with timer.cpu("h2d"):
                    if channels_last:
                        images = _batch_dict["images"].to(device, non_blocking=True, memory_format=torch.channels_last)
                    else:
                        images = _batch_dict["images"].to(device, non_blocking=True)
                    decoder_inputs = _batch_dict["decoder_inputs"].to(device, non_blocking=True)
                    labels = _batch_dict["labels"].to(device, non_blocking=True)
                    contour_targets = _batch_dict["contour_targets"].to(device, non_blocking=True)

                accum_steps = stage.grad_accumulation_steps
                is_accum_step = (stage_step % accum_steps) == 0 or stage_step == stage_total_steps
                if (stage_step - 1) % accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                    accum_corruption = torch.zeros((), dtype=torch.bool, device=device)
                with timer.gpu("forward"):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=bf16_enabled,
                    ):
                        outputs = model(pixel_values=images, input_ids=decoder_inputs, return_aux=True)
                        token_loss = F.cross_entropy(
                            outputs["logits"].reshape(-1, vocab_size),
                            labels.reshape(-1),
                            ignore_index=-100,
                            label_smoothing=stage.label_smoothing,
                        )
                        contour_loss = F.cross_entropy(outputs["contour_logits"], contour_targets)
                        loss = token_loss + (stage.contour_loss_weight * contour_loss)
                        if accum_steps > 1:
                            loss = loss / accum_steps
```

Replace with:

```python
                with timer.cpu("h2d"):
                    labels = _batch_dict["labels"].to(device, non_blocking=True)
                    contour_targets = _batch_dict["contour_targets"].to(device, non_blocking=True)

                # Per-tier accum_steps (Task 4 will wire this into a tier-aware path).
                # Until then this still uses stage.grad_accumulation_steps for legacy stages.
                accum_steps = stage.grad_accumulation_steps
                is_accum_step = (stage_step % accum_steps) == 0 or stage_step == stage_total_steps
                if (stage_step - 1) % accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                    accum_corruption = torch.zeros((), dtype=torch.bool, device=device)
                with timer.gpu("forward"):
                    outputs = _forward_batch_for_train(
                        model, _batch_dict, device,
                        bf16_enabled=bf16_enabled, channels_last=channels_last,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=bf16_enabled,
                    ):
                        token_loss = F.cross_entropy(
                            outputs["logits"].reshape(-1, vocab_size),
                            labels.reshape(-1),
                            ignore_index=-100,
                            label_smoothing=stage.label_smoothing,
                        )
                        contour_loss = F.cross_entropy(outputs["contour_logits"], contour_targets)
                        loss = token_loss + (stage.contour_loss_weight * contour_loss)
                        if accum_steps > 1:
                            loss = loss / accum_steps
```

- [ ] **Step 6: Run full train tests; expect no regression**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/ -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/train/train.py tests/train/test_train_loop_tier_dispatch.py
git commit -m "feat(train): tier-aware batch dispatch in _run_stage forward path"
```

> **Review:** Confirm the helper does NOT move labels/contour_targets to device (caller still owns those). Confirm both branches use the same autocast scope so mixed-tier stages produce identical bf16 numerics.

---

### Task 3: Tier-grouped sampler API refactor — opt-step semantics (TDD)

**Files:**
- Modify: `src/train/tier_sampler.py`
- Test: `tests/train/test_tier_grouped_sampler_opt_steps.py`

**Why this task:** Lock decision #2 — the sampler must emit live batches in contiguous `grad_accum_live`-sized blocks so opt-step boundaries align with tier transitions. The new API takes `(n_cached_opt_steps, n_live_opt_steps, grad_accum_cached, grad_accum_live)`; the helper computes total batches and emits the correct sequence. The legacy `cached_ratio`/`total_batches` API is preserved for the existing Phase 0 tests.

- [ ] **Step 1: Write failing tests**

```python
"""Tier-grouped sampler opt-step semantics — live batches in contiguous blocks."""
from __future__ import annotations

import pytest


CACHED_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}
LIVE_DATASETS = {"cameraprimus_systems"}


def _make_entries(n_cached: int = 100, n_live: int = 100):
    out = []
    for i in range(n_cached):
        out.append({"dataset": "synthetic_systems", "split": "train", "sample_id": f"c{i}"})
    for i in range(n_live):
        out.append({"dataset": "cameraprimus_systems", "split": "train", "sample_id": f"l{i}"})
    return out


def test_opt_step_api_emits_correct_batch_counts():
    from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

    batches = build_tier_grouped_sampler_by_opt_steps(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        n_cached_opt_steps=10,
        n_live_opt_steps=2,
        b_cached=16,
        b_live=2,
        grad_accum_cached=1,
        grad_accum_live=8,
        seed=42,
    )

    # 10 cached opt-steps × 1 batch each = 10 cached batches
    # 2 live opt-steps × 8 batches each = 16 live batches
    # Total = 26 batches
    assert len(batches) == 26


def test_live_batches_are_contiguous_8_blocks():
    """Each live opt-step's 8 batches must be emitted contiguously."""
    from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

    batches = build_tier_grouped_sampler_by_opt_steps(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        n_cached_opt_steps=20,
        n_live_opt_steps=3,
        b_cached=16,
        b_live=2,
        grad_accum_cached=1,
        grad_accum_live=8,
        seed=42,
    )
    entries = _make_entries()

    def _tier_of(batch):
        idx = batch[0]
        return "cached" if entries[idx]["dataset"] in CACHED_DATASETS else "live"

    # Walk batch sequence; live runs must be exactly 8 long.
    i = 0
    while i < len(batches):
        if _tier_of(batches[i]) == "live":
            run_len = 0
            while i < len(batches) and _tier_of(batches[i]) == "live":
                run_len += 1
                i += 1
            assert run_len == 8, f"expected live run of 8 batches, got {run_len}"
        else:
            i += 1


def test_all_batches_tier_pure():
    """Same invariant as Phase 0 tier sampler — every batch is single-tier."""
    from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

    batches = build_tier_grouped_sampler_by_opt_steps(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        n_cached_opt_steps=50,
        n_live_opt_steps=5,
        b_cached=16,
        b_live=2,
        grad_accum_cached=1,
        grad_accum_live=8,
        seed=7,
    )
    entries = _make_entries()
    for batch in batches:
        tiers = set()
        for idx in batch:
            ds = entries[idx]["dataset"]
            tiers.add("cached" if ds in CACHED_DATASETS else "live")
        assert len(tiers) == 1, f"mixed-tier batch: {tiers}"


def test_grad_accum_cached_greater_than_one():
    """If grad_accum_cached > 1, cached batches also emit in contiguous blocks."""
    from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

    batches = build_tier_grouped_sampler_by_opt_steps(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        n_cached_opt_steps=5,
        n_live_opt_steps=2,
        b_cached=8,
        b_live=2,
        grad_accum_cached=2,  # 2 micro-batches per cached opt-step
        grad_accum_live=8,
        seed=0,
    )

    # 5 cached opt-steps × 2 batches + 2 live opt-steps × 8 batches = 26
    assert len(batches) == 26
    # First batch's tier dictates the run; verify cached runs are exactly 2.
    entries = _make_entries()

    def _tier_of(batch):
        return "cached" if entries[batch[0]]["dataset"] in CACHED_DATASETS else "live"

    i = 0
    while i < len(batches):
        tier = _tier_of(batches[i])
        run_len = 0
        while i < len(batches) and _tier_of(batches[i]) == tier:
            run_len += 1
            i += 1
        expected = 2 if tier == "cached" else 8
        assert run_len == expected, f"{tier} run length {run_len} != {expected}"


def test_legacy_api_still_works():
    """Phase 0's build_tier_grouped_sampler API stays callable."""
    from src.train.tier_sampler import build_tier_grouped_sampler

    batches = build_tier_grouped_sampler(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.9,
        total_batches=100,
        b_cached=8,
        b_live=2,
        seed=0,
    )
    assert len(batches) == 100
```

- [ ] **Step 2: Run tests; expect failure**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_tier_grouped_sampler_opt_steps.py -v`
Expected: 4 failures on the new opt-step API tests; 1 pass on the legacy API test.

- [ ] **Step 3: Add new opt-step API to `tier_sampler.py`**

Append to `src/train/tier_sampler.py`:

```python
def build_tier_grouped_sampler_by_opt_steps(
    entries: list[dict],
    cached_datasets: set[str],
    live_datasets: set[str],
    *,
    n_cached_opt_steps: int,
    n_live_opt_steps: int,
    b_cached: int,
    b_live: int,
    grad_accum_cached: int,
    grad_accum_live: int,
    seed: int = 0,
) -> list[list[int]]:
    """Build a list of tier-pure batched index lists where opt-step boundaries
    coincide with tier transitions.

    The sampler emits per-opt-step blocks: each cached opt-step is
    ``grad_accum_cached`` consecutive cached batches; each live opt-step is
    ``grad_accum_live`` consecutive live batches. Opt-step blocks are then
    randomly interleaved so the trainer sees an unpredictable cached/live
    mix at the opt-step level — but a cached batch never interrupts a live
    accumulation window.

    Args:
        entries: Full dataset entries list (same order as dataset.entries).
        cached_datasets: Set of dataset names that are in the cached tier.
        live_datasets: Set of dataset names that are in the live tier.
        n_cached_opt_steps: Number of cached-tier opt-steps to emit.
        n_live_opt_steps: Number of live-tier opt-steps to emit.
        b_cached: Batch size for cached batches.
        b_live: Batch size for live batches.
        grad_accum_cached: Number of cached batches per cached opt-step.
        grad_accum_live: Number of live batches per live opt-step.
        seed: Random seed for reproducibility.

    Returns:
        A flat list of batches, total length
        ``n_cached_opt_steps * grad_accum_cached + n_live_opt_steps * grad_accum_live``.
    """
    import random

    rng = random.Random(seed)

    cached_indices = [i for i, e in enumerate(entries) if e.get("dataset") in cached_datasets]
    live_indices = [i for i, e in enumerate(entries) if e.get("dataset") in live_datasets]

    if not cached_indices and n_cached_opt_steps > 0:
        raise ValueError(
            f"build_tier_grouped_sampler_by_opt_steps: n_cached_opt_steps={n_cached_opt_steps} "
            f"but no entries match cached_datasets={cached_datasets}"
        )
    if not live_indices and n_live_opt_steps > 0:
        raise ValueError(
            f"build_tier_grouped_sampler_by_opt_steps: n_live_opt_steps={n_live_opt_steps} "
            f"but no entries match live_datasets={live_datasets}"
        )

    def _draw_batch(pool: list[int], size: int) -> list[int]:
        return [rng.choice(pool) for _ in range(size)]

    # Build per-opt-step blocks.
    cached_blocks: list[list[list[int]]] = [
        [_draw_batch(cached_indices, b_cached) for _ in range(grad_accum_cached)]
        for _ in range(n_cached_opt_steps)
    ]
    live_blocks: list[list[list[int]]] = [
        [_draw_batch(live_indices, b_live) for _ in range(grad_accum_live)]
        for _ in range(n_live_opt_steps)
    ]

    # Interleave at the opt-step block level.
    block_order: list[str] = (["cached"] * n_cached_opt_steps) + (["live"] * n_live_opt_steps)
    rng.shuffle(block_order)

    cached_iter = iter(cached_blocks)
    live_iter = iter(live_blocks)
    result: list[list[int]] = []
    for tier in block_order:
        block = next(cached_iter) if tier == "cached" else next(live_iter)
        result.extend(block)

    return result
```

- [ ] **Step 4: Run tests; verify all pass**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_tier_grouped_sampler_opt_steps.py tests/train/test_tier_grouped_sampler.py -v`
Expected: all pass (new API + legacy API both green).

- [ ] **Step 5: Commit**

```bash
git add src/train/tier_sampler.py tests/train/test_tier_grouped_sampler_opt_steps.py
git commit -m "feat(train): tier-grouped sampler with opt-step boundary alignment"
```

> **Review:** Confirm legacy `build_tier_grouped_sampler` still passes its Phase 0 tests untouched. Confirm the new API does NOT consume `b_cached × n_cached_opt_steps` worth of memory eagerly — it's still O(total_batches) of small int lists, fine.

---

### Task 4: Train loop wires tier-aware sampler + grad_accum (TDD)

**Files:**
- Modify: `src/train/train.py:2095-2200` (the `_run_stage` dataloader-build + per-step loop preamble)
- Test: `tests/train/test_train_loop_tier_grad_accum.py`

**Why this task:** Engages the new tier-grouped sampler when `stage.tier_grouped_sampling=True`. Replaces the single `accum_steps = stage.grad_accumulation_steps` with a per-batch tier-derived value. Asserts opt-step boundaries align with tier transitions. **This is the largest task** — the train loop's accumulation arithmetic and the dataloader-build path both change. Tests are integration-flavored (run a few steps end-to-end on a tiny model).

- [ ] **Step 1: Write failing test**

```python
"""Tier-aware grad accumulation in _run_stage.

Verifies opt-step counter increments correctly across cached and live
batches with different grad_accum values, and that the sampler is
build_tier_grouped_sampler_by_opt_steps when tier_grouped_sampling=True.
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch

import torch
import pytest


def test_compute_tier_aware_total_batches():
    """The trainer's helper computes total batches from opt-step targets correctly."""
    from src.train.train import _compute_tier_grouped_batch_plan

    plan = _compute_tier_grouped_batch_plan(
        target_opt_steps=4500,
        cached_data_ratio=0.9,
        b_cached=16,
        b_live=2,
        grad_accum_cached=1,
        grad_accum_live=8,
    )
    # n_cached_opt_steps = 4500 * 0.9 = 4050; n_live_opt_steps = 450
    assert plan.n_cached_opt_steps == 4050
    assert plan.n_live_opt_steps == 450
    assert plan.total_batches == 4050 * 1 + 450 * 8  # 4050 + 3600 = 7650


def test_per_batch_grad_accum_dispatch_on_tier():
    """_grad_accum_for_batch returns 1 for cached, 8 for live."""
    from src.train.train import _grad_accum_for_batch

    cached_batch = {"tier": "cached", "encoder_hidden": torch.zeros(16, 156, 1280)}
    live_batch = {"tier": "live", "images": torch.zeros(2, 1, 250, 2500)}

    assert _grad_accum_for_batch(cached_batch, grad_accum_cached=1, grad_accum_live=8) == 1
    assert _grad_accum_for_batch(live_batch, grad_accum_cached=1, grad_accum_live=8) == 8
```

- [ ] **Step 2: Run; verify failure**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_train_loop_tier_grad_accum.py -v`
Expected: 2 failures on missing imports.

- [ ] **Step 3: Add helper functions to `train.py`**

Insert these helpers immediately after `_forward_batch_for_train` (added in Task 2):

```python
@dataclass(frozen=True)
class _TierGroupedBatchPlan:
    """Result of converting opt-step targets into batch counts."""
    n_cached_opt_steps: int
    n_live_opt_steps: int
    n_cached_batches: int
    n_live_batches: int
    total_batches: int


def _compute_tier_grouped_batch_plan(
    *,
    target_opt_steps: int,
    cached_data_ratio: float,
    b_cached: int,
    b_live: int,
    grad_accum_cached: int,
    grad_accum_live: int,
) -> "_TierGroupedBatchPlan":
    """Convert (target_opt_steps, cached_data_ratio) into tier batch counts.

    cached_data_ratio is the OPT-STEP gradient share (locked decision #1).
    Cached opt-steps = round(target_opt_steps * cached_data_ratio); live
    opt-steps = target_opt_steps - cached_opt_steps.
    """
    n_cached_opt = int(round(target_opt_steps * cached_data_ratio))
    n_live_opt = max(0, target_opt_steps - n_cached_opt)
    n_cached_batches = n_cached_opt * grad_accum_cached
    n_live_batches = n_live_opt * grad_accum_live
    return _TierGroupedBatchPlan(
        n_cached_opt_steps=n_cached_opt,
        n_live_opt_steps=n_live_opt,
        n_cached_batches=n_cached_batches,
        n_live_batches=n_live_batches,
        total_batches=n_cached_batches + n_live_batches,
    )


def _grad_accum_for_batch(batch_dict: "Dict[str, object]", *, grad_accum_cached: int, grad_accum_live: int) -> int:
    """Return the per-tier grad_accum for the batch's tier."""
    tier = batch_dict.get("tier", "live")
    return grad_accum_cached if tier == "cached" else grad_accum_live
```

- [ ] **Step 4: Run helper tests; verify pass**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_train_loop_tier_grad_accum.py -v`
Expected: 2 passes.

- [ ] **Step 5: Wire tier-grouped sampler build into `_run_stage`**

Edit `src/train/train.py:2109-2185` (the StageBDataset + sampler + train_loader build). After the existing `_stage_ds = StageBDataset(...)` instantiation, add:

```python
            # Stage 3: tier-grouped sampler path.
            if stage.tier_grouped_sampling:
                from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

                # Pass cache pointers to the dataset so cached entries route through read_cache_entry.
                cache_root_path = (project_root / stage.cache_root).resolve() if stage.cache_root else None
                _stage_ds = StageBDataset(
                    stage,
                    grouped_entries,
                    project_root=project_root,
                    image_height=image_height,
                    image_width=image_width,
                    max_sequence_length=stage.max_sequence_length,
                    augment=True,
                    rng_seed=seed,
                    cache_root=cache_root_path,
                    cache_hash16=stage.cache_hash16,
                )
                # Compute opt-step → batch plan. stage_total_steps comes from
                # the existing scheduler arithmetic and counts opt-steps.
                _plan = _compute_tier_grouped_batch_plan(
                    target_opt_steps=stage_total_steps,
                    cached_data_ratio=stage.cached_data_ratio,
                    b_cached=stage.b_cached,
                    b_live=stage.b_live,
                    grad_accum_cached=stage.grad_accumulation_steps_cached,
                    grad_accum_live=stage.grad_accumulation_steps_live,
                )
                _batch_list = build_tier_grouped_sampler_by_opt_steps(
                    entries=_stage_ds.entries,
                    cached_datasets=_CACHED_DATASETS,
                    live_datasets={"cameraprimus_systems"},
                    n_cached_opt_steps=_plan.n_cached_opt_steps,
                    n_live_opt_steps=_plan.n_live_opt_steps,
                    b_cached=stage.b_cached,
                    b_live=stage.b_live,
                    grad_accum_cached=stage.grad_accumulation_steps_cached,
                    grad_accum_live=stage.grad_accumulation_steps_live,
                    seed=seed,
                )
                _train_loader = torch.utils.data.DataLoader(
                    _stage_ds,
                    batch_sampler=_TierGroupedBatchSampler(_batch_list),
                    num_workers=num_workers,
                    pin_memory=_pin_memory,
                    persistent_workers=(num_workers > 0),
                    prefetch_factor=_effective_prefetch,
                    collate_fn=StageBDataset.collate_fn,
                    worker_init_fn=stage_b_worker_init_fn,
                )
            else:
                # Legacy path (Stage 1, Stage 2 v2): unchanged.
                _stage_total_train_samples = (
                    stage_total_steps * stage.batch_size * stage.grad_accumulation_steps
                )
                _train_sampler = build_stage_b_sampler(
                    stage, _stage_ds,
                    total_samples=_stage_total_train_samples,
                    seed=seed,
                )
                _train_loader = torch.utils.data.DataLoader(
                    _stage_ds,
                    batch_size=stage.batch_size,
                    sampler=_train_sampler,
                    num_workers=num_workers,
                    pin_memory=_pin_memory,
                    persistent_workers=(num_workers > 0),
                    prefetch_factor=_effective_prefetch,
                    collate_fn=StageBDataset.collate_fn,
                    worker_init_fn=stage_b_worker_init_fn,
                )
            _train_iter = iter(_train_loader)
            vocab_size = _stage_ds._vocab.size
```

- [ ] **Step 6: Add `_TierGroupedBatchSampler` class**

Insert into `src/train/train.py` near the other sampler code (after `build_stage_b_sampler`, around line 900):

```python
class _TierGroupedBatchSampler(torch.utils.data.Sampler):
    """Wraps a pre-computed list of batched index lists for use with DataLoader.

    Yields each inner list (one batch worth of indices). DataLoader's
    batch_sampler arg expects exactly this shape: an iterable that yields
    lists of indices.
    """

    def __init__(self, batches: list[list[int]]) -> None:
        super().__init__(data_source=None)
        self._batches = batches
        # Resume support (Task 8): on resume, set this to the consumed prefix length.
        self._start_idx: int = 0

    def set_start_idx(self, idx: int) -> None:
        """Skip the first ``idx`` batches when iterating (used on resume)."""
        if idx < 0 or idx > len(self._batches):
            raise ValueError(f"start_idx={idx} out of range [0, {len(self._batches)}]")
        self._start_idx = idx

    def __iter__(self):
        for batch in self._batches[self._start_idx:]:
            yield batch

    def __len__(self) -> int:
        return len(self._batches) - self._start_idx
```

- [ ] **Step 7: Replace per-step `accum_steps` lookup with tier-aware version**

In the per-step loop (`src/train/train.py` lines 2225–2226), where it currently does:

```python
                accum_steps = stage.grad_accumulation_steps
                is_accum_step = (stage_step % accum_steps) == 0 or stage_step == stage_total_steps
                if (stage_step - 1) % accum_steps == 0:
```

Replace with:

```python
                if stage.tier_grouped_sampling:
                    accum_steps = _grad_accum_for_batch(
                        _batch_dict,
                        grad_accum_cached=stage.grad_accumulation_steps_cached,
                        grad_accum_live=stage.grad_accumulation_steps_live,
                    )
                    # Tier-grouped sampler emits batches in contiguous tier blocks of
                    # size accum_steps. The opt-step boundary is at the LAST batch of
                    # each block: track per-tier-block micro-batch index.
                    is_accum_step = (
                        _tier_block_micro_idx + 1 == accum_steps
                    ) or stage_step == stage_total_steps
                    if _tier_block_micro_idx == 0:
                        optimizer.zero_grad(set_to_none=True)
                        accum_corruption = torch.zeros((), dtype=torch.bool, device=device)
                    _tier_block_micro_idx = (_tier_block_micro_idx + 1) % accum_steps
                else:
                    accum_steps = stage.grad_accumulation_steps
                    is_accum_step = (stage_step % accum_steps) == 0 or stage_step == stage_total_steps
                    if (stage_step - 1) % accum_steps == 0:
                        optimizer.zero_grad(set_to_none=True)
                        accum_corruption = torch.zeros((), dtype=torch.bool, device=device)
```

Initialize `_tier_block_micro_idx = 0` once before the per-step loop (after `timer.reset_step()` near line 2187).

- [ ] **Step 8: Run dry-run smoke test on legacy stage to confirm no regression**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/ -q`
Expected: all tests pass (legacy stage path is unchanged).

- [ ] **Step 9: Commit**

```bash
git add src/train/train.py tests/train/test_train_loop_tier_grad_accum.py
git commit -m "feat(train): wire tier-grouped sampler + per-tier grad accumulation"
```

> **Review:** Confirm `tier_grouped_sampling=False` (legacy stages) still uses `stage.batch_size` and `stage.grad_accumulation_steps` unchanged. Confirm the per-step loop's `_tier_block_micro_idx` resets cleanly at every block boundary — a misalignment here causes silent loss-scaling bugs that are very expensive to detect at training time.

---

### Task 5: `_run_validation` tier-aware batch dispatch (TDD)

**Files:**
- Modify: `src/train/train.py:1254-1327` (the `_run_validation` function)
- Test: `tests/train/test_run_validation_tier_dispatch.py`

**Why this task:** `_run_validation` currently does `images = batch_dict["images"]` (line 1294) which fails on cached batches. Phase 1's val loader will yield both tiers; validation must dispatch on `tier` like the train loop does (Task 2's `_forward_batch_for_train` is reused).

- [ ] **Step 1: Write failing test**

```python
"""_run_validation handles cached and live batches transparently."""
from __future__ import annotations
from unittest.mock import MagicMock

import torch
import pytest


def _make_mixed_val_loader(n_cached: int = 1, n_live: int = 1):
    """A simple iterable that yields a mix of cached and live collated batches."""
    cached = {
        "tier": "cached",
        "encoder_hidden": torch.zeros(2, 156, 1280, dtype=torch.bfloat16),
        "_h16": 16,
        "_w16": 156,
        "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
        "labels": torch.zeros(2, 511, dtype=torch.long),
        "contour_targets": torch.zeros(2, 32, dtype=torch.long),
    }
    live = {
        "tier": "live",
        "images": torch.zeros(2, 1, 250, 2500, dtype=torch.float32),
        "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
        "labels": torch.zeros(2, 511, dtype=torch.long),
        "contour_targets": torch.zeros(2, 32, dtype=torch.long),
        "content_widths": torch.tensor([2500, 2500], dtype=torch.long),
    }
    return [cached] * n_cached + [live] * n_live


def test_run_validation_handles_cached_and_live_batches():
    """_run_validation iterates a mixed-tier loader without KeyError on 'images'."""
    from src.train.train import _run_validation, StageTrainingConfig, DatasetMix

    stage = StageTrainingConfig(
        stage_name="test",
        stage_b_encoder="radio_h",
        epochs=1, effective_samples_per_epoch=100, batch_size=2, max_sequence_length=512,
        lr_dora=0.0, lr_new_modules=0.0, warmup_steps=0, schedule="cosine",
        weight_decay=0.0, label_smoothing=0.0, contour_loss_weight=0.01,
        checkpoint_every_steps=500, validate_every_steps=500,
        grad_accumulation_steps=1, loraplus_lr_ratio=1.0,
        dataset_mix=(DatasetMix(dataset="grandstaff_systems", ratio=1.0),),
    )

    model = MagicMock()
    model.return_value = {
        "logits": torch.zeros(2, 511, 100),
        "contour_logits": torch.zeros(2, 3, 32),
    }

    loader = _make_mixed_val_loader(n_cached=2, n_live=1)
    result = _run_validation(
        model, stage, iter(loader), torch.device("cpu"),
        bf16_enabled=False, validation_batches=3, vocab_size=100,
    )

    assert result is not None
    assert "val_loss" in result
    # 3 calls (2 cached + 1 live), one of which used cached_features:
    cached_calls = [c for c in model.call_args_list if "cached_features" in c.kwargs]
    live_calls = [c for c in model.call_args_list if "pixel_values" in c.kwargs]
    assert len(cached_calls) == 2
    assert len(live_calls) == 1
```

- [ ] **Step 2: Run; verify failure**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_run_validation_tier_dispatch.py -v`
Expected: 1 failure (`KeyError: 'images'` when the cached batch is dispatched).

- [ ] **Step 3: Refactor `_run_validation` to use `_forward_batch_for_train`**

Replace `_run_validation`'s body at `src/train/train.py:1287-1327`:

```python
    losses: List[float] = []
    contour_losses: List[float] = []
    model.eval()
    with torch.no_grad():
        val_iter = iter(val_loader)
        for _ in range(validation_batches):
            try:
                batch_dict = next(val_iter)
            except StopIteration:
                break
            labels = batch_dict["labels"].to(device, non_blocking=True)
            contour_targets = batch_dict["contour_targets"].to(device, non_blocking=True)
            outputs = _forward_batch_for_train(
                model, batch_dict, device,
                bf16_enabled=bf16_enabled, channels_last=channels_last,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16_enabled,
            ):
                token_loss = F.cross_entropy(
                    outputs["logits"].reshape(-1, vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                    label_smoothing=stage.label_smoothing,
                )
                contour_loss = F.cross_entropy(outputs["contour_logits"], contour_targets)
                total_loss = token_loss + (stage.contour_loss_weight * contour_loss)
            losses.append(float(total_loss.item()))
            contour_losses.append(float(contour_loss.item()))
    model.train()
    if not losses:
        return None
    return {
        "val_loss": float(sum(losses) / len(losses)),
        "val_contour_loss": float(sum(contour_losses) / len(contour_losses)),
    }
```

(The `import torch.nn.functional as F` at the top of `_run_validation` stays; the `import torch` similarly. The `images = ...` block and the inline `model(pixel_values=...)` call are now gone.)

- [ ] **Step 4: Run val + train test suites**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/train/train.py tests/train/test_run_validation_tier_dispatch.py
git commit -m "feat(train): tier-aware batch dispatch in _run_validation"
```

> **Review:** Confirm `_run_validation`'s legacy callers (Stage 1, Stage 2 v2) still receive batches with `images` key (they do — `_forward_batch_for_train` falls through to the live branch when `tier != "cached"`).

---

### Task 6: Per-dataset val_loss tracking (TDD)

**Files:**
- Modify: `src/train/train.py` — extend `_run_validation` (or add `_run_validation_per_dataset`); update `_run_stage` to call the per-dataset variant when `tier_grouped_sampling=True`; extend step-log row schema.
- Test: `tests/train/test_per_dataset_val_loss.py`

**Why this task:** Spec line 216 requires per-dataset val_loss separately. Decision #4 is 4 disjoint passes over per-dataset val loaders. The aggregate val_loss (sample-weighted) feeds the existing best-tracking + sanity halt logic.

- [ ] **Step 1: Write failing test**

```python
"""Per-dataset val_loss: 4 disjoint passes + sample-weighted aggregate."""
from __future__ import annotations
from unittest.mock import MagicMock

import torch
import pytest


def test_run_validation_per_dataset_returns_one_loss_per_dataset_plus_aggregate():
    """The per-dataset entry point returns a dict with one val_loss per dataset
    and one aggregate val_loss (sample-weighted by dataset_mix)."""
    from src.train.train import _run_validation_per_dataset, StageTrainingConfig, DatasetMix

    stage = StageTrainingConfig(
        stage_name="stage3-test",
        stage_b_encoder="radio_h",
        epochs=1, effective_samples_per_epoch=100, batch_size=2, max_sequence_length=512,
        lr_dora=0.0, lr_new_modules=0.0, warmup_steps=0, schedule="cosine",
        weight_decay=0.0, label_smoothing=0.0, contour_loss_weight=0.01,
        checkpoint_every_steps=500, validate_every_steps=500,
        grad_accumulation_steps=1, loraplus_lr_ratio=1.0,
        dataset_mix=(
            DatasetMix(dataset="synthetic_systems", ratio=0.7),
            DatasetMix(dataset="grandstaff_systems", ratio=0.1),
            DatasetMix(dataset="primus_systems", ratio=0.1),
            DatasetMix(dataset="cameraprimus_systems", ratio=0.1),
        ),
        tier_grouped_sampling=True,
        b_cached=2, b_live=2,
        grad_accumulation_steps_cached=1, grad_accumulation_steps_live=1,
        cached_data_ratio=0.9,
        cache_root="x", cache_hash16="x",
    )

    model = MagicMock()
    model.return_value = {
        "logits": torch.zeros(2, 511, 100),
        "contour_logits": torch.zeros(2, 3, 32),
    }

    # Mock per-dataset loaders: each yields 1 batch with all-zeros tensors.
    def _mk_loader(tier: str):
        if tier == "cached":
            batch = {
                "tier": "cached",
                "encoder_hidden": torch.zeros(2, 156, 1280, dtype=torch.bfloat16),
                "_h16": 16, "_w16": 156,
                "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
                "labels": torch.zeros(2, 511, dtype=torch.long),
                "contour_targets": torch.zeros(2, 32, dtype=torch.long),
            }
        else:
            batch = {
                "tier": "live",
                "images": torch.zeros(2, 1, 250, 2500),
                "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
                "labels": torch.zeros(2, 511, dtype=torch.long),
                "contour_targets": torch.zeros(2, 32, dtype=torch.long),
                "content_widths": torch.tensor([2500, 2500], dtype=torch.long),
            }
        return [batch]

    per_dataset_loaders = {
        "synthetic_systems": _mk_loader("cached"),
        "grandstaff_systems": _mk_loader("cached"),
        "primus_systems": _mk_loader("cached"),
        "cameraprimus_systems": _mk_loader("live"),
    }

    result = _run_validation_per_dataset(
        model, stage, per_dataset_loaders, torch.device("cpu"),
        bf16_enabled=False, validation_batches=1, vocab_size=100,
    )

    assert "val_loss_per_dataset" in result
    assert set(result["val_loss_per_dataset"].keys()) == {
        "synthetic_systems", "grandstaff_systems", "primus_systems", "cameraprimus_systems",
    }
    assert "val_loss" in result  # aggregate
    # The aggregate must equal the dataset_mix-weighted mean of per-dataset losses.
    weights = {dm.dataset: dm.ratio for dm in stage.dataset_mix}
    expected = sum(weights[k] * v for k, v in result["val_loss_per_dataset"].items())
    assert result["val_loss"] == pytest.approx(expected, rel=1e-6)
```

- [ ] **Step 2: Run; verify failure**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_per_dataset_val_loss.py -v`
Expected: failure (`ImportError: cannot import name '_run_validation_per_dataset'`).

- [ ] **Step 3: Add `_run_validation_per_dataset`**

Insert into `src/train/train.py` immediately after `_run_validation`:

```python
def _run_validation_per_dataset(
    model,
    stage: "StageTrainingConfig",
    per_dataset_loaders: "Dict[str, object]",
    device,
    *,
    bf16_enabled: bool,
    validation_batches: int,
    vocab_size: int,
    channels_last: bool = False,
) -> "Dict[str, object]":
    """Run validation as 4 disjoint passes (one per dataset).

    Returns:
        {
            "val_loss": float,                       # sample-weighted aggregate
            "val_contour_loss": float,               # sample-weighted aggregate
            "val_loss_per_dataset": Dict[str, float],
            "val_contour_loss_per_dataset": Dict[str, float],
        }

    The aggregate val_loss is weighted by ``stage.dataset_mix`` ratios so it is
    consistent with the previous single-pass aggregate when the val loader had
    the same sampling distribution.
    """
    per_loss: Dict[str, float] = {}
    per_contour: Dict[str, float] = {}
    for dataset_name, loader in per_dataset_loaders.items():
        result = _run_validation(
            model, stage, loader, device,
            bf16_enabled=bf16_enabled, validation_batches=validation_batches,
            vocab_size=vocab_size, channels_last=channels_last,
        )
        if result is None:
            continue
        per_loss[dataset_name] = result["val_loss"]
        per_contour[dataset_name] = result["val_contour_loss"]

    weights = {dm.dataset: dm.ratio for dm in stage.dataset_mix}
    weight_sum = sum(weights.get(k, 0.0) for k in per_loss.keys())
    if weight_sum <= 0:
        return None
    val_loss_agg = sum(weights.get(k, 0.0) * v for k, v in per_loss.items()) / weight_sum
    val_contour_agg = sum(weights.get(k, 0.0) * v for k, v in per_contour.items()) / weight_sum
    return {
        "val_loss": float(val_loss_agg),
        "val_contour_loss": float(val_contour_agg),
        "val_loss_per_dataset": per_loss,
        "val_contour_loss_per_dataset": per_contour,
    }
```

- [ ] **Step 4: Build per-dataset val loaders in `_run_stage` and call the new entry point**

Edit `src/train/train.py:2156-2185` (the `_val_dataset` and `_val_loader` build). Replace it with:

```python
            # Build val-side dataset(s).
            if stage.tier_grouped_sampling:
                # Per-dataset val loaders for stratified val_loss reporting.
                _per_dataset_val_loaders: Dict[str, object] = {}
                cache_root_path = (project_root / stage.cache_root).resolve() if stage.cache_root else None
                for mix_item in stage.dataset_mix:
                    _val_ds_for_dataset = StageBDataset(
                        stage,
                        # Build a grouped_entries dict containing only this dataset.
                        {
                            (mix_item.dataset, "val"): grouped_entries.get((mix_item.dataset, "val"), []),
                        },
                        split="val",
                        project_root=project_root,
                        image_height=image_height,
                        image_width=image_width,
                        max_sequence_length=stage.max_sequence_length,
                        augment=False,
                        rng_seed=seed,
                        cache_root=cache_root_path,
                        cache_hash16=stage.cache_hash16,
                    )
                    if len(_val_ds_for_dataset) == 0:
                        continue
                    # Use unweighted RandomSampler over this dataset only.
                    _val_total = validation_batches * (
                        stage.b_cached if mix_item.dataset in _CACHED_DATASETS else stage.b_live
                    )
                    _ds_sampler = torch.utils.data.RandomSampler(
                        _val_ds_for_dataset, replacement=True, num_samples=_val_total,
                        generator=torch.Generator().manual_seed(seed),
                    )
                    _bs_for_ds = stage.b_cached if mix_item.dataset in _CACHED_DATASETS else stage.b_live
                    _per_dataset_val_loaders[mix_item.dataset] = torch.utils.data.DataLoader(
                        _val_ds_for_dataset,
                        batch_size=_bs_for_ds,
                        sampler=_ds_sampler,
                        num_workers=num_workers,
                        pin_memory=_pin_memory,
                        persistent_workers=(num_workers > 0),
                        prefetch_factor=_effective_prefetch,
                        collate_fn=StageBDataset.collate_fn,
                        worker_init_fn=stage_b_worker_init_fn,
                    )
                _val_loader = None  # Sentinel: trainer uses _per_dataset_val_loaders below
            else:
                # Legacy single-pass val loader (Stage 1 / Stage 2 v2).
                _val_dataset = StageBDataset(
                    stage,
                    grouped_entries,
                    split="val",
                    project_root=project_root,
                    image_height=image_height,
                    image_width=image_width,
                    max_sequence_length=stage.max_sequence_length,
                    augment=False,
                    rng_seed=seed,
                )
                _val_total_samples = validation_batches * stage.batch_size
                _val_sampler = build_stage_b_sampler(
                    stage, _val_dataset,
                    total_samples=_val_total_samples,
                    seed=seed,
                    split_override="val",
                )
                _val_loader = torch.utils.data.DataLoader(
                    _val_dataset,
                    batch_size=stage.batch_size,
                    sampler=_val_sampler,
                    num_workers=num_workers,
                    pin_memory=_pin_memory,
                    persistent_workers=(num_workers > 0),
                    prefetch_factor=_effective_prefetch,
                    collate_fn=StageBDataset.collate_fn,
                    worker_init_fn=stage_b_worker_init_fn,
                )
                _per_dataset_val_loaders = None
```

- [ ] **Step 5: Update validation call site**

Find every call to `_run_validation(...)` inside `_run_stage` (search for `_run_validation(`). Replace with conditional dispatch:

```python
                if stage.tier_grouped_sampling and _per_dataset_val_loaders is not None:
                    validation_result = _run_validation_per_dataset(
                        model, stage, _per_dataset_val_loaders, device,
                        bf16_enabled=bf16_enabled,
                        validation_batches=validation_batches,
                        vocab_size=vocab_size,
                        channels_last=channels_last,
                    )
                else:
                    validation_result = _run_validation(
                        model, stage, _val_loader, device,
                        bf16_enabled=bf16_enabled,
                        validation_batches=validation_batches,
                        vocab_size=vocab_size,
                        channels_last=channels_last,
                    )
```

- [ ] **Step 6: Extend step-log JSONL row to include `val_loss_per_dataset`**

Find where validation_result is logged (search for `step_log` write — likely near line 2370). Where the existing row builds, add:

```python
                if "val_loss_per_dataset" in validation_result:
                    row["val_loss_per_dataset"] = validation_result["val_loss_per_dataset"]
                if "val_contour_loss_per_dataset" in validation_result:
                    row["val_contour_loss_per_dataset"] = validation_result["val_contour_loss_per_dataset"]
```

- [ ] **Step 7: Run all validation tests**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_per_dataset_val_loss.py tests/train/test_run_validation_tier_dispatch.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/train/train.py tests/train/test_per_dataset_val_loss.py
git commit -m "feat(train): per-dataset val_loss with sample-weighted aggregate"
```

> **Review:** Confirm legacy stages still call `_run_validation` (single pass) and never touch the per-dataset path. Confirm the aggregate val_loss is what `best_val_loss` tracks (so `_best.pt` selection logic is unchanged).

---

### Task 7: Sanity halt — val_loss > 5.0 in first 200 steps OR NaN (TDD)

**Files:**
- Modify: `src/train/train.py` — sanity-halt check after each validation pass
- Test: `tests/train/test_sanity_halt.py`

**Why this task:** Spec line 226 requires sanity halt distinct from per-step corruption detection. Triggers: (1) val_loss > 5.0 in any validation pass before opt-step 200, (2) val_loss is NaN at any validation pass.

- [ ] **Step 1: Write failing test**

```python
"""Sanity halt fires on val_loss > 5.0 (first 200 steps) OR NaN (any time)."""
from __future__ import annotations
import math
import pytest


def test_sanity_halt_returns_true_on_val_loss_above_5_in_first_200_steps():
    from src.train.train import _should_sanity_halt

    assert _should_sanity_halt(val_loss=6.0, global_step=100) == ("val_loss>5 in first 200 steps", True)


def test_sanity_halt_returns_false_on_val_loss_above_5_after_200_steps():
    from src.train.train import _should_sanity_halt

    msg, halt = _should_sanity_halt(val_loss=6.0, global_step=300)
    assert halt is False


def test_sanity_halt_returns_true_on_nan_at_any_step():
    from src.train.train import _should_sanity_halt

    msg, halt = _should_sanity_halt(val_loss=math.nan, global_step=100)
    assert halt is True
    msg, halt = _should_sanity_halt(val_loss=math.nan, global_step=10000)
    assert halt is True


def test_sanity_halt_returns_false_on_normal_loss():
    from src.train.train import _should_sanity_halt

    msg, halt = _should_sanity_halt(val_loss=0.3, global_step=100)
    assert halt is False
    msg, halt = _should_sanity_halt(val_loss=4.99, global_step=199)
    assert halt is False
```

- [ ] **Step 2: Run; verify failure**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_sanity_halt.py -v`
Expected: 4 failures (ImportError).

- [ ] **Step 3: Add helper**

Insert into `src/train/train.py` near other helpers (e.g., after `_grad_accum_for_batch`):

```python
def _should_sanity_halt(*, val_loss: float, global_step: int) -> "Tuple[str, bool]":
    """Spec sanity halt: val_loss > 5.0 in first 200 steps OR NaN at any step.

    Returns (message, should_halt). message is the human-readable reason; only
    meaningful when should_halt=True.
    """
    if math.isnan(val_loss):
        return ("val_loss is NaN", True)
    if global_step < 200 and val_loss > 5.0:
        return (f"val_loss={val_loss:.3f} > 5.0 within first 200 opt-steps", True)
    return ("", False)
```

- [ ] **Step 4: Wire into `_run_stage` after each validation pass**

Find the `validation_result = _run_validation(...)` call site (or the new per-dataset variant — Task 6) and add immediately after the result-non-None check:

```python
                halt_msg, should_halt = _should_sanity_halt(
                    val_loss=validation_result["val_loss"],
                    global_step=global_step,
                )
                if should_halt:
                    print(
                        f"[train] HALT (sanity): {halt_msg} at global_step={global_step}",
                        flush=True,
                    )
                    if step_log_path is not None:
                        _write_step_log(
                            step_log_path,
                            {
                                "event": "sanity_halt",
                                "global_step": global_step,
                                "val_loss": validation_result["val_loss"],
                                "reason": halt_msg,
                            },
                        )
                    sys.exit(1)
```

(`_write_step_log` is the existing helper — search for it to confirm name. If it doesn't exist as that exact name, use the same write idiom the rest of the trainer uses for step-log rows.)

- [ ] **Step 5: Run sanity-halt tests + full train suite**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_sanity_halt.py tests/train/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/train/train.py tests/train/test_sanity_halt.py
git commit -m "feat(train): sanity halt on val_loss > 5.0 in first 200 steps OR NaN"
```

> **Review:** Confirm `sys.exit(1)` is the right mechanism (vs raising) — the existing trainer uses `sys.exit` for fatal-but-expected exits. Confirm a sanity-halt JSONL row is written before exit so the post-mortem has the trigger.

---

### Task 8: Sampler resume state in checkpoint (TDD)

**Files:**
- Modify: `src/train/train.py` — `_save_checkpoint` payload + resume path in `_run_stage` + `_TierGroupedBatchSampler.set_start_idx`
- Test: `tests/train/test_sampler_resume.py`

**Why this task:** Decision #6 — sampler resume = same seed + skip first N batches. Without this, resuming a Stage 3 run re-trains over the same prefix of batches, throwing off the sample-exposure budget.

- [ ] **Step 1: Write failing test**

```python
"""Sampler resume: rebuild same list, skip consumed prefix."""
from __future__ import annotations
import torch
import pytest


def test_save_checkpoint_persists_last_batch_idx():
    """_save_checkpoint persists last_batch_idx in the payload when given."""
    from src.train.train import _save_checkpoint
    import tempfile
    from pathlib import Path

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_checkpoint(
            checkpoint_dir=Path(tmpdir),
            model=model, optimizer=optimizer,
            stage_name="stage3-test",
            global_step=500,
            stage_step=500,
            best_val_loss=0.3,
            last_batch_idx=123,
        )
        payload = torch.load(path, map_location="cpu")
    assert payload["last_batch_idx"] == 123


def test_tier_grouped_batch_sampler_skips_prefix_on_set_start_idx():
    from src.train.train import _TierGroupedBatchSampler

    batches = [[1, 2], [3, 4], [5, 6], [7, 8]]
    sampler = _TierGroupedBatchSampler(batches)
    sampler.set_start_idx(2)
    assert list(sampler) == [[5, 6], [7, 8]]
    assert len(sampler) == 2


def test_tier_grouped_batch_sampler_set_start_idx_out_of_range():
    from src.train.train import _TierGroupedBatchSampler
    sampler = _TierGroupedBatchSampler([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        sampler.set_start_idx(10)
    with pytest.raises(ValueError):
        sampler.set_start_idx(-1)
```

- [ ] **Step 2: Run; verify failure**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_sampler_resume.py -v`
Expected: 1 failure (`_save_checkpoint` doesn't accept `last_batch_idx`); 2 likely pass (Task 4's `_TierGroupedBatchSampler.set_start_idx` was already added).

- [ ] **Step 3: Extend `_save_checkpoint` signature + payload**

Edit `src/train/train.py:1330-1365`. Add keyword arg and payload field:

```python
def _save_checkpoint(
    checkpoint_dir: Path,
    model,
    optimizer,
    scheduler=None,
    *,
    stage_name: str,
    global_step: int,
    stage_step: Optional[int] = None,
    stage_steps_total: Optional[int] = None,
    stage_b_config: Optional[Dict[str, object]] = None,
    name_suffix: Optional[str] = None,
    best_val_loss: Optional[float] = None,
    last_batch_idx: Optional[int] = None,
) -> Path:
    ...
    payload = {
        "stage_name": stage_name,
        "global_step": global_step,
        "stage_step": stage_step,
        "stage_steps_total": stage_steps_total,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage_b_config": stage_b_config,
        "best_val_loss": best_val_loss,
        "last_batch_idx": last_batch_idx,
    }
    ...
```

- [ ] **Step 4: Wire batch-idx tracking into `_run_stage`**

In `_run_stage`, track `_batch_idx_consumed` (incremented in the per-step loop, separate from `stage_step`/`global_step`). At every checkpoint save, pass it as `last_batch_idx`:

```python
                ckpt_path = _save_checkpoint(
                    checkpoint_dir, model, optimizer, scheduler,
                    stage_name=stage.stage_name,
                    global_step=global_step,
                    stage_step=stage_step,
                    stage_steps_total=stage_total_steps,
                    stage_b_config=stage_b_config,
                    best_val_loss=best_val_loss,
                    last_batch_idx=(_batch_idx_consumed if stage.tier_grouped_sampling else None),
                )
```

(Find the existing `_save_checkpoint` call sites — there are typically 2-3 in `_run_stage`. Update each.)

- [ ] **Step 5: Wire resume**

Where the trainer loads a resume checkpoint and rebuilds the dataloader, after `_train_loader = ...`:

```python
            if stage.tier_grouped_sampling and resumed_stage:
                last_batch_idx = resume_payload.get("last_batch_idx", 0) or 0
                if last_batch_idx > 0:
                    _train_loader.batch_sampler.set_start_idx(last_batch_idx)
                    _batch_idx_consumed = last_batch_idx
                else:
                    _batch_idx_consumed = 0
            else:
                _batch_idx_consumed = 0
```

- [ ] **Step 6: Run all sampler-resume tests + train suite**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_sampler_resume.py tests/train/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/train/train.py tests/train/test_sampler_resume.py
git commit -m "feat(train): persist last_batch_idx in checkpoint and skip prefix on resume"
```

> **Review:** Confirm legacy stages skip the resume-batch-skip logic (their checkpoint payload's `last_batch_idx` is None and the `if stage.tier_grouped_sampling` guard prevents misuse).

---

### Task 9: Stage 2 v2 init checkpoint load smoke test (TDD)

**Files:**
- Test: `tests/train/test_stage2_v2_init_checkpoint_load.py`

**Why this task:** Plan C must verify the DoRA-aware loader works on the Stage 2 v2 best.pt schema, because the run begins with `--resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt`. Phase 0 plan decision #1 already documents the loader; this test is a regression guard for the load + freeze pattern.

This test is a smoke test that constructs a synthetic Stage 2-shaped checkpoint and round-trips it through the loader; it does not require the actual GPU box file.

- [ ] **Step 1: Write the test**

```python
"""DoRA-aware loader can load a Stage 2 v2-style checkpoint and freeze the encoder."""
from __future__ import annotations
import tempfile
from pathlib import Path

import torch
import pytest


@pytest.mark.slow
def test_dora_aware_loader_freezes_encoder_after_load():
    """The Stage 3 init pattern: load Stage 2 ckpt → freeze encoder + encoder DoRA → trainable surface remains decoder/cross-attention/LM head/positional_bridge."""
    from src.checkpoint_io import load_stage_b_checkpoint
    from src.train.model_factory import (
        ModelFactoryConfig,
        build_stage_b_components,
        model_factory_config_from_checkpoint_payload,
    )

    # Build a tiny Stage 2 v2-style payload by saving a freshly-constructed model.
    factory = ModelFactoryConfig(
        encoder="radio_h",
        decoder_layers=2, decoder_heads=2, decoder_d_model=64,
        max_sequence_length=512, vocab_size=100,
        dora_rank=8, dora_alpha=16,
    )
    components = build_stage_b_components(factory)
    model = components.model
    payload = {
        "model_state_dict": model.state_dict(),
        "stage_b_config": {
            "encoder": "radio_h",
            "decoder_layers": 2, "decoder_heads": 2, "decoder_d_model": 64,
            "max_sequence_length": 512, "vocab_size": 100,
            "dora_rank": 8, "dora_alpha": 16,
        },
        "best_val_loss": 0.148,
        "global_step": 4000,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "stage2_v2_best.pt"
        torch.save(payload, ckpt_path)

        dora_cfg = model_factory_config_from_checkpoint_payload(payload)
        new_model, _ = build_stage_b_components(dora_cfg)
        load_stage_b_checkpoint(ckpt_path, model=new_model.model, dora_config=dora_cfg)

        # Freeze encoder + encoder-side DoRA (the Stage 3 trainable surface).
        for name, p in new_model.model.named_parameters():
            if name.startswith("encoder."):
                p.requires_grad = False
        n_trainable = sum(p.numel() for p in new_model.model.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in new_model.model.parameters() if not p.requires_grad)
        assert n_trainable > 0, "Stage 3 trainable surface must include decoder + LM head"
        assert n_frozen > 0, "encoder must be frozen"
```

- [ ] **Step 2: Run**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && pytest tests/train/test_stage2_v2_init_checkpoint_load.py -v -m slow`
Expected: pass (or skip if `radio_h` weights aren't downloadable in the test env — in which case the test is gated on GPU box; document and move on).

- [ ] **Step 3: If the test requires real RADIO weights and they're unavailable locally, mark it `@pytest.mark.gpu_box_only` and run only on the GPU box.**

If unavailable locally: add a `pytest.skip("RADIO weights unavailable; run on GPU box")` shim guarded by a `which_gpu_box()` check. Otherwise leave as-is.

- [ ] **Step 4: Commit**

```bash
git add tests/train/test_stage2_v2_init_checkpoint_load.py
git commit -m "test(train): smoke test DoRA-aware loader on Stage 2 v2-shaped checkpoint"
```

> **Review:** Confirm this test reflects the loader pattern documented in `project_radio_stage3_design.md` line 55 ("DoRA-PEFT checkpoint loading"). If the test is GPU-box-gated, that's fine — its job is documenting the contract, not gating PR.

---

### Task 10: MusicXML validity rate eval driver hook

**Files:**
- Locate the Stage 2 v2 eval driver (likely `src/eval/run_radio_eval.py` or similar — search the repo with `grep -rn "MusicXML"` and `grep -rn "musicxml_validity"` to find it)
- Modify the eval driver to enable the validity rate metric (Stage 2 v2 left it at None per spec line 264)
- Test: extend the existing eval test for the validity-rate codepath, or add `tests/eval/test_musicxml_validity_rate.py`

**Why this task:** Spec §"3. MusicXML validity rate" line 264: "Enable in eval driver (Stage 2 v2 left it at None per 'things not done' #2)." This is a Phase 2 metric, but Plan C lands the code now (the eval driver is shared between Phase 1 best-checkpoint sanity eval and Phase 2 full eval). Plan D will use the value as a gate.

- [ ] **Step 1: Locate the eval driver**

Run:
```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && grep -rn "musicxml_validity\|MusicXML validity" --include='*.py'
```
Record the file path and the existing call site.

- [ ] **Step 2: Read the existing eval driver**

Use Read on the file from Step 1. Identify (a) where token sequences are decoded into MusicXML, (b) where the validity-rate metric is computed (likely commented out or wrapped in `if False:`), (c) the metric reporting structure.

- [ ] **Step 3: Write a failing test for validity-rate computation**

(Test code shape depends on the existing eval driver structure — read it first. Skeleton: build a fake set of token sequences where some decode to valid MusicXML and some don't; assert the rate matches.)

- [ ] **Step 4: Enable the metric**

Replace the disabled validity-rate path with the enabled one. The tokens → MusicXML decoder already exists (RADIO Subproject 2 audit was 96.9% — see `project_radio_kern_converter_bugs.md`); the validity check is `xml.etree.ElementTree.fromstring(decoded_xml)` plus a MusicXML-specific schema check (or just "did it parse at all" if no schema check exists yet).

- [ ] **Step 5: Run**

Run the eval driver tests + the new validity-rate test. All pass.

- [ ] **Step 6: Commit**

```bash
git add <eval driver file> tests/eval/test_musicxml_validity_rate.py
git commit -m "feat(eval): enable MusicXML validity rate metric"
```

> **Review:** Confirm the validity-rate computation is on by default (no flag needed). Confirm Plan D will be able to read the metric out of the standard eval-result JSON without further changes.

> **Note:** If the eval driver does not currently support reading from cached features (it might run encoder + decoder live for eval), Plan C does not require enabling cache-side eval; Phase 1's `_best.pt` is evaluated by the live eval driver (full encoder forward) per spec.

---

### Task 11: Pre-flight ready check + launch handoff doc

**Files:**
- Create: `scripts/preflight_stage3_phase1.py`
- Create: `docs/superpowers/handoffs/2026-05-09-radio-stage3-phase1-launch-handoff.md`

**Why this task:** Spec hard gate: "explicit user go-ahead before any training run starts." The pre-flight script verifies all prerequisites are in place on the GPU box, dry-runs the trainer for ~10 opt-steps to catch surface-level bugs, and prints a checklist. The handoff doc lays out launch steps for the user to review before saying "go."

- [ ] **Step 1: Create `scripts/preflight_stage3_phase1.py`**

```python
#!/usr/bin/env python3
"""Pre-flight ready check for Stage 3 Phase 1 training launch.

Verifies on the local clone (or via SSH on the GPU box, depending on argv):
1. configs/train_stage3_radio_systems.yaml parses and tier fields validate.
2. data/cache/encoder/<hash16>/ exists and contains the expected sample count.
3. src/data/manifests/token_manifest_stage3.jsonl exists and row count matches.
4. checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt exists.
5. (Optional, --dry-run) Trainer dry-run mode runs for 10 opt-steps without
   raising and writes a step-log.

Exit 0 = ready. Exit 1 = at least one prerequisite missing.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_stage3_radio_systems.yaml")
    ap.add_argument("--manifest", default="src/data/manifests/token_manifest_stage3.jsonl")
    ap.add_argument("--cache-root", default="data/cache/encoder")
    ap.add_argument("--init-ckpt",
                    default="checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run trainer in dry-run mode for 10 opt-steps")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    fails: list[str] = []

    # 1. Config parses and tier fields validate.
    from src.train.train import load_stage_config
    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        fails.append(f"config not found: {config_path}")
    else:
        try:
            cfg = load_stage_config(config_path)
            if not cfg.tier_grouped_sampling:
                fails.append(f"config {config_path} does not have tier_grouped_sampling=true")
            print(f"[preflight] config OK: tier_grouped_sampling=True, b_cached={cfg.b_cached}, b_live={cfg.b_live}")
        except Exception as exc:
            fails.append(f"config parse error: {exc}")

    # 2. Cache exists.
    if cfg is not None:
        cache_dir = (project_root / args.cache_root / cfg.cache_hash16).resolve()
        if not cache_dir.exists():
            fails.append(f"cache dir not found: {cache_dir}")
        else:
            metadata_path = cache_dir / "metadata.json"
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text())
                samples = meta.get("samples_processed", 0)
                print(f"[preflight] cache OK: {samples} samples at {cache_dir}")
                if samples != 215985:
                    fails.append(f"cache sample count {samples} != expected 215985")
            else:
                fails.append(f"cache metadata.json not found at {metadata_path}")

    # 3. Manifest.
    manifest_path = (project_root / args.manifest).resolve()
    if not manifest_path.exists():
        fails.append(f"manifest not found: {manifest_path}")
    else:
        with manifest_path.open() as fh:
            n_rows = sum(1 for line in fh if line.strip())
        print(f"[preflight] manifest OK: {n_rows} rows at {manifest_path}")
        if n_rows != 303663:
            fails.append(f"manifest row count {n_rows} != expected 303663 (combined Stage 3 manifest)")

    # 4. Init checkpoint.
    init_ckpt = (project_root / args.init_ckpt).resolve()
    if not init_ckpt.exists():
        fails.append(f"init checkpoint not found: {init_ckpt}")
    else:
        print(f"[preflight] init ckpt OK: {init_ckpt}")

    # 5. Optional dry-run.
    if args.dry_run and not fails:
        print("[preflight] launching trainer dry-run (10 opt-steps)...")
        import subprocess
        result = subprocess.run([
            sys.executable, "src/train/train.py",
            "--stage-configs", str(args.config),
            "--mode", "dry-run",
            "--max-steps-per-stage", "10",
            "--token-manifest", str(args.manifest),
            "--resume-checkpoint", str(args.init_ckpt),
        ], cwd=project_root, capture_output=True, text=True)
        if result.returncode != 0:
            fails.append(f"dry-run failed: stderr=\n{result.stderr[-2000:]}")
        else:
            print("[preflight] dry-run OK")

    if fails:
        print("\n[preflight] FAIL — prerequisites missing:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\n[preflight] READY — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run preflight on local clone (without --dry-run, since cache is GPU-box-only)**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python scripts/preflight_stage3_phase1.py
```
Expected: 4 of 5 fails (cache + checkpoint + maybe manifest if not synced locally) — that's OK. The script's job is to be runnable on the GPU box for real check.

- [ ] **Step 3: Create the launch handoff doc**

```markdown
# Stage 3 Phase 1 — Launch Handoff (Pre-Flight)

> Plan C is implementation-complete. This handoff captures the launch checklist for the user to review before training begins.

## TL;DR

- Branch `feat/stage3-phase1-training` is ready to merge or train-from.
- All Phase 1 trainer code + configs land in this branch.
- Pre-flight script: `scripts/preflight_stage3_phase1.py`. Run on GPU box; expect "READY".
- Launch command: see "Run the trainer" below.
- Step target: 4500 opt-steps. Manual extension gates at 4500 → 6000 → 7500.

## Pre-flight checklist (must hold before saying "go")

1. [ ] `feat/stage3-phase1-training` branch is up to date and synced to GPU box at `10.10.1.29`.
2. [ ] On GPU box, run: `venv-cu132\Scripts\python scripts/preflight_stage3_phase1.py --dry-run` and confirm exit 0.
3. [ ] Cache directory `data/cache/encoder/ac8948ae4b5be3e9/` exists with `samples_processed=215985`.
4. [ ] Manifest `src/data/manifests/token_manifest_stage3.jsonl` exists with 303,663 rows.
5. [ ] Init checkpoint `checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt` exists.
6. [ ] Disk has ≥ 50 GB free for `checkpoints/full_radio_stage3_v1/` (≈ 25 GB best.pt + step ckpts).
7. [ ] No active GPU jobs on the box (check `nvidia-smi`).

## Run the trainer

```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python -u src/train/train.py --stage-configs configs/train_stage3_radio_systems.yaml --mode execute --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/full_radio_stage3_v1 --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v1_steps.jsonl'
```

Wall-time projection: 1.5–3h to opt-step 4500 on the RTX 5090 (per spec).

## Monitor

Tail the step-log; expect every 500 opt-steps to write a row with `val_loss`, `val_loss_per_dataset`, and `wall_time_s`.

```bash
ssh 10.10.1.29 'powershell.exe -Command "Get-Content -Wait logs/full_radio_stage3_v1_steps.jsonl"'
```

Watch for:
- Sanity halt: `[train] HALT (sanity): val_loss > 5.0` in first 200 steps OR NaN at any time.
- Per-dataset val_loss divergence: cameraprimus_systems > 1.5 × Stage 2 v2's analog signal = early warning.
- VRAM: `nvidia-smi --query-gpu=memory.used --format=csv -l 30 -f vram.csv` (expect ~47% used).

## At opt-step 4500: extension decision

Per spec lines 201–210, pause and review the val_loss curve over the last 750 opt-steps:
- Still descending → extend to 6000.
- Plateaued or regressed → finalize at 4500.
- At 6000, repeat for 7500 cap.

## Phase 1 → Phase 2 gate

After best.pt is finalized, run Plan D (Phase 2 eval). Phase 1 only ends when all five exit criteria hold (see plan §"Phase 1 Exit Criteria").

## What goes in the post-training handoff

- Final `_best.pt` opt-step + per-dataset val_loss values
- Wall time + VRAM peak
- Whether sanity halt fired (and why if so)
- Whether step-extension protocol triggered
- All step-log rows compressed into a summary table
```

- [ ] **Step 4: Commit**

```bash
git add scripts/preflight_stage3_phase1.py docs/superpowers/handoffs/2026-05-09-radio-stage3-phase1-launch-handoff.md
git commit -m "feat(scripts): pre-flight ready check + Phase 1 launch handoff"
```

> **Review:** Confirm the run command exactly matches the YAML's run-command comment. Confirm the handoff lists every concrete prerequisite from §"Phase 1 Exit Criteria."

---

### Task 12: Phase 1 launch — explicit user go-ahead, SSH, monitor

**Why this task:** Spec hard gate: "explicit user go-ahead before any training run starts." This is a manual gate, NOT an automatic step. The plan must surface it explicitly.

- [ ] **Step 1: Confirm pre-flight passes on GPU box**

```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python scripts/preflight_stage3_phase1.py --dry-run'
```
Expected: exit 0, "READY — all checks passed."

- [ ] **Step 2: Surface the go-ahead gate to the user**

Print the launch command + the pre-flight output, then ask the user verbatim:

> "Pre-flight is green. The trainer will run for 1.5–3h on the GPU box, target 4500 opt-steps. Are you ready to launch?"

DO NOT proceed until the user replies "yes" / "go" / equivalent.

- [ ] **Step 3: Launch the trainer (background, with step-log)**

```bash
ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && start /b venv-cu132\Scripts\python -u src/train/train.py --stage-configs configs/train_stage3_radio_systems.yaml --mode execute --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/full_radio_stage3_v1 --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v1_steps.jsonl > logs/full_radio_stage3_v1_stdout.log 2>&1'
```

(Adapt the Windows backgrounding pattern to whatever the GPU box's previous Stage 2 v2 launch used — see `train_stage2_radio_systems.yaml` line 8 for the exact pattern.)

- [ ] **Step 4: Monitor every 30 minutes**

Tail the step-log; check VRAM. Surface any warnings:
- val_loss > 5.0 in first 200 steps → sanity halt expected; the trainer will exit and log the row.
- Step time > 4× Stage 2 v2 baseline → cache I/O regression; flag.

- [ ] **Step 5: At opt-step 4500, surface the step-extension decision**

Print the val_loss curve (last 750 opt-steps) and ask the user verbatim:

> "Opt-step 4500 reached. val_loss curve over last 750 steps: [chart]. Extend to 6000, or finalize at 4500?"

Per spec line 207: still descending → extend; plateaued/regressed → finalize.

- [ ] **Step 6: Repeat at 6000 (7500 cap) if extension was chosen**

Same protocol as Step 5.

> **Review:** This task is procedural — no code lands. Its purpose is to be a checkbox-discipline reminder that the launch + extension gates are user-controlled.

---

### Task 13: Phase 1 handoff doc

**Files:**
- Create: `docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase1-complete-handoff.md` (date filled at completion)

**Why this task:** Per the handoff convention used at every prior phase boundary. Plan D consumes this handoff as input.

- [ ] **Step 1: Run summary scripts on the step-log**

Compute: best opt-step, best val_loss, per-dataset val_loss at best, sanity-halt status, total wall time, peak VRAM, total opt-steps run. Use `jq` or a small Python one-liner over the step-log JSONL.

- [ ] **Step 2: Write the handoff with the full template**

(Template structure mirrors `2026-05-09-radio-stage3-phase0-merged-phase1-launch-handoff.md` — read it for format. Sections: TL;DR, Where to start, State of the project, Things in flight on GPU box, Files changed, Commits, User preferences, Goodbye.)

Critical content for Plan D's input:
- best.pt absolute path (GPU box)
- best.pt opt-step + best_val_loss
- Per-dataset val_loss table at best.pt
- Whether sanity halt fired
- Whether step extension was used
- Phase 1 → Phase 2 gate verdict (4 of the 5 exit criteria; criterion 4 — MusicXML validity — is part of Plan D)

- [ ] **Step 3: Commit + open Plan C → main PR**

```bash
git add docs/superpowers/handoffs/2026-05-XX-radio-stage3-phase1-complete-handoff.md
git commit -m "docs(handoff): Stage 3 Phase 1 complete handoff"
gh pr create --title "Stage 3 Phase 1: training infrastructure + launch" \
  --body "$(cat <<'EOF'
## Summary
- Lands tier-aware StageTrainingConfig fields and `tier_grouped_sampling` toggle
- Adds opt-step-aware tier-grouped sampler API
- Wires per-tier grad accumulation in `_run_stage` + `_run_validation`
- Adds per-dataset val_loss tracking
- Sanity halt on val_loss > 5.0 (first 200 steps) OR NaN
- Sampler resume = deterministic re-derivation by seed + skip prefix
- Pre-flight check + launch / handoff docs

## Test plan
- [ ] All `pytest tests/train/` passes locally
- [ ] Pre-flight script exits 0 on GPU box
- [ ] Phase 1 training run completed; best.pt at val_loss < 0.5
- [ ] Per-dataset val_loss floors hold

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

> **Review:** Final pass — does the handoff cover all 5 exit criteria explicitly? Does the PR link the spec + Plan C? Is best.pt's path canonicalized?

---

## Self-review notes (for the executing engineer)

1. **Plan B (Phase 0) was 14 tasks; Plan C (Phase 1) is 14 tasks** — comparable in scope. Don't be surprised if Tasks 4 + 6 take longer than the others; they touch the trainer's accumulation arithmetic and the validation control flow respectively.
2. **The cross-tier opt-step semantics in Task 4 are the highest-risk implementation** — a misalignment in `_tier_block_micro_idx` produces silent loss-scaling bugs that cost a full training run to detect. Read the test in Task 3 for the contract; trace one cached opt-step + one live opt-step through the loop on paper before writing the wiring.
3. **No file in this plan is created by hand-editing the merged YAML output** — the YAML is constructed by the Stage 3 config dataclass + `load_stage_config`. If the YAML in Task 0 doesn't round-trip cleanly through `load_stage_config`, the config dataclass field defaults are wrong; debug there, not in the YAML.
4. **The Phase 1 → Phase 2 gate (exit criteria) is checked after the run completes**, not during. Don't add a "soft halt at exit criteria failure" — those are user-level decisions.

