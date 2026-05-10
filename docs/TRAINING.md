# Training

This page covers training commands for Stage A (system-level YOLO detection)
and Stage B (system-level RADIO token recognition). Both stages require a
CUDA-capable GPU; see [HARDWARE.md](HARDWARE.md) for the reference hardware
and [INSTALL.md](INSTALL.md) for environment setup.

All commands assume the repo root as the working directory and an activated
`venv-cu132` environment.

## Stage A — System-level YOLO

Stage A detects musical *systems* (cross-staff groupings) on full pages. The
canonical detector is YOLOv26m trained on the mixed Stage A dataset built
from synthetic + real corpora.

```bash
python scripts/train_yolo.py \
  --model yolo26m.pt \
  --data data/processed/mixed_systems_v1/data.yaml \
  --epochs 100 --imgsz 1920 --batch 4 --workers 6 \
  --amp --nan-guard --noise --noise-warmup-steps 2000 \
  --project runs/detect/runs --name yolo26m_systems --patience 30
```

Wall time: ~10–12h on a single RTX 5090. Gate: val mAP50 ≥ 0.95.

Outputs land under `runs/detect/runs/yolo26m_systems/weights/{best,last}.pt`.
See [paths.md](paths.md) for the full output layout.

### Notable flags

- `--noise --noise-warmup-steps 2000` — enables the scan-noise + page-curvature
  augmentation pipeline (defined in `src/train/scan_noise.py`) and ramps the
  augmentation probability from 0 to full over the first 2000 steps. This
  avoids the early-training NaN failure mode where `cls_loss × noise` blows
  up before the model has learned anything.
- `--nan-guard` — zeroes individual NaN/Inf gradients per batch instead of
  skipping the whole step, so the rare warmup gradient explosion doesn't
  disrupt training.
- `--amp` — mixed-precision training; required for batch 4 at 1920px on a
  32 GB GPU.

### Data preparation

The `data/processed/mixed_systems_v1/data.yaml` input is produced by the
data-preparation pipeline documented in [QUICKSTART.md](QUICKSTART.md),
which renders synthetic systems, derives system labels for the real
corpora, and mixes them into the final YOLO-ready split. See also the
"Training data preparation" section below.

## Stage B — System-level RADIO

Stage B is the RADIO-encoder + custom-decoder transcription model. The
trainer is `src/train/train.py`, configured via YAML files under `configs/`.

```bash
# Stage 1 v2 — per-staff RADIO (completed; for re-runs only)
python src/train/train.py --config configs/train_stage1_radio.yaml

# Stage 2 v2 — polyphonic vocab-extension warmup
# Dataset mix is 80% systems / 20% single-staff. The config was historically
# named train_stage2_radio_systems.yaml; see the header in the YAML for the
# rationale behind the rename.
python src/train/train.py --config configs/train_stage2_radio_polyphonic.yaml

# Stage 3 — full system-level retrain with encoder-cache hybrid
# Requires the Phase 0 encoder cache to be pre-built (see below).
python src/train/train.py --config configs/train_stage3_radio_systems.yaml
```

Stage 3 design (encoder caching, hybrid dataset mix, opt-step budget,
correctness + throughput gates) is documented at
`docs/superpowers/specs/2026-05-07-radio-stage3-design.md`.

### Live training monitor

Stage B step-level metrics stream to `src/train/training_steps.jsonl`. The
training-health monitor and live dashboard read this file:

```bash
# One-shot health summary (LR, loss, gradient norms, spike detection)
python src/train/monitor_training.py \
  --step-log src/train/training_steps.jsonl \
  --window 20 --spike-factor 3.0 --grad-threshold 100 \
  --output src/train/training_health_summary.json

# Auto-refreshing TUI dashboard
python src/train/monitor_dashboard.py \
  --step-log src/train/training_steps.jsonl \
  --window 20 --spike-factor 3.0 --grad-threshold 100 \
  --refresh-ms 2000 --tail-limit 40
```

For Stage A, point the same dashboard at the YOLO `results.csv` instead:

```bash
python src/train/monitor_dashboard.py \
  --yolo-results runs/detect/runs/yolo26m_systems/results.csv \
  --refresh-ms 2000 --tail-limit 40
```

## Encoder cache (Stage 3)

Stage 3 training uses a pre-built encoder feature cache so that the frozen
RADIO encoder doesn't re-run every epoch. The cache must be rebuilt whenever
the encoder configuration changes.

```bash
python scripts/build_encoder_cache.py \
  --manifest src/data/manifests/<token-manifest>.jsonl \
  --output data/encoder_cache/
```

The canonical Stage 3 manifest summary committed to the repo is
`src/data/manifests/token_manifest_stage3_audit.json`. That JSON records
total entries, per-dataset breakdown
(`synthetic_systems` / `grandstaff_systems` / `primus_systems` /
`cameraprimus_systems`), per-split counts, and the source `.jsonl` manifest
paths it summarizes — the full per-system `.jsonl` manifests themselves
are gitignored build artifacts produced by the data-preparation pipeline.

Pass the underlying `.jsonl` (not the audit JSON) to
`scripts/build_encoder_cache.py` as `--manifest`.

After (re-)building, verify the cache is consistent with the current code
before resuming a long Stage 3 run:

```bash
python scripts/check_encoder_resume.py
```

`scripts/measure_encoder_cache_throughput.py` benchmarks read throughput
from the cache directory if you suspect I/O is the bottleneck.

## Training data preparation

The data-preparation pipeline (downloading PrIMuS / Camera-PrIMuS /
GrandStaff / OpenScore Lieder, rendering synthetic systems, deriving
system labels, building the mixed Stage A dataset) is documented in
[QUICKSTART.md](QUICKSTART.md). See also [paths.md](paths.md) for
canonical output locations.

## Platform notes

The training scripts are pure Python and run identically on Linux and
Windows; only the venv-activation step differs. Once the `venv-cu132`
environment from [INSTALL.md](INSTALL.md) is active, no additional
platform-specific setup is required.

A few practical notes:

- **CUDA driver.** PyTorch nightly cu132 requires a recent NVIDIA driver
  (R555+ on Linux, R555+ on Windows). Older drivers silently fall back to
  CPU.
- **DataLoader workers.** Linux multiprocess `DataLoader` workers
  (`--workers 6` for Stage A) work out of the box. On Windows, the same
  flag works under PowerShell but each worker re-imports the project —
  keep import-time side effects minimal.
- **No-GPU fallback.** For smoke tests without a GPU, Stage A accepts
  `--device cpu`; Stage B does not currently support CPU training.
