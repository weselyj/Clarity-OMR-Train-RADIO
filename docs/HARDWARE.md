# Hardware Requirements

This project requires a CUDA-capable NVIDIA GPU for both training and
inference. CPU-only execution is not supported.

## Minimum

- **GPU:** CUDA-capable NVIDIA GPU
- **VRAM:** 24 GB for inference; 48+ GB recommended for Stage B training
- **Driver:** NVIDIA 596.21 or later (CUDA 13.x compatible)
- **CUDA toolkit:** Not required — PyTorch ships its own CUDA runtime via the
  cu132 nightly wheel.
- **System RAM:** 32 GB minimum, 64 GB recommended for the lieder corpus eval
  (memory profile dominated by music21 score graphs).

## Tested Configurations

| GPU | VRAM | OS | Python | PyTorch | Notes |
|---|---|---|---|---|---|
| RTX 5090 | 32 GB | Windows 11 | 3.13 | 2.13 dev cu132 | Production reference |

The training and inference paths are CUDA-mandatory:

- Stage A YOLO training (`scripts/train_yolo.py`) calls into ultralytics, which
  uses torchvision NMS ops that resolve only against a CUDA-built torch.
- Stage B RADIO encoder (`src/models/radio_stage_b.py`) loads C-RADIOv4-H at
  bf16 on a CUDA device; the encoder cache pipeline (`scripts/build_encoder_cache.py`)
  is similarly device-bound.
- The unit test suite gates CUDA-required tests via `tests/conftest.py`. On a
  CPU-only environment, those tests skip with a clear "CUDA required" reason
  rather than failing with a cryptic torchvision import error.

## CPU-Only Tests

A subset of pure-Python tests run anywhere — data-layer helpers under
`tests/data/`, plus a handful of top-level utility tests. Everything under
`tests/inference/`, `tests/pipeline/`, `tests/cli/`, `tests/models/`, and
`tests/train/` (and `eval/tests/`) requires a CUDA box and skips cleanly
otherwise.
