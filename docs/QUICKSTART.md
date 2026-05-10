# Quickstart

This guide takes you from a fresh clone to running a single-piece smoke
inference. Full training and corpus evaluation are covered in
[TRAINING.md](TRAINING.md) and [EVALUATION.md](EVALUATION.md).

## Prerequisites

- CUDA-capable GPU and driver. See [HARDWARE.md](HARDWARE.md).
- Python 3.13+.

## 1. Install

Follow [INSTALL.md](INSTALL.md) to clone the repo, create the cu132 venv,
and verify that `torch.cuda.is_available()` returns `True`.

## 2. Download Stage A YOLO weights

Stage A weights are not committed to the repo (the `runs/` tree is
gitignored). Either:

- **Train Stage A** — see [TRAINING.md](TRAINING.md). Requires the mixed
  systems dataset built by `scripts/build_mixed_v2_systems.py`.
- **Download released weights** — see [RESULTS.md](RESULTS.md) for the
  Hugging Face release link (pending).

Place the weights at `runs/detect/runs/yolo26m_systems/weights/best.pt`
(the path the eval driver expects by default).

## 3. Download a Stage B checkpoint

Same options as Stage A. Place the checkpoint at
`checkpoints/full_radio_stage3_v2/_best.pt` (or pass an explicit path).

## 4. Run inference on one PDF

```bash
python -m src.cli.run_system_inference \
    --pdf path/to/score.pdf \
    --out output.musicxml \
    --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt \
    --stage-b-ckpt checkpoints/full_radio_stage3_v2/_best.pt
```

A diagnostics sidecar is also written to `output.musicxml.diagnostics.json`.

## 5. Verify the test suite

Pure-Python tests (no GPU needed):

```bash
pytest tests/data -q
```

Full suite (CUDA required):

```bash
pytest -q
```

## Next steps

- For corpus-level eval: [EVALUATION.md](EVALUATION.md)
- For training: [TRAINING.md](TRAINING.md)
- For the deep architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- For repo-relative artifact paths: [paths.md](paths.md)
