# Clarity-OMR-Train-RADIO

Training pipeline for an optical music recognition model that turns printed-score images into MusicXML.

This repository is a fork of [**clquwu/Clarity-OMR-Train**](https://github.com/clquwu/Clarity-OMR-Train) — the original training pipeline for [Clarity-OMR](https://github.com/clquwu/Clarity-OMR) (the inference repo). The fork extends the upstream project in two directions:

1. **Encoder swap (DaViT → C-RADIOv4-H).** Replaces the 86M-param ImageNet-pretrained DaViT encoder with NVIDIA's ~700M-param RADIO foundation encoder.
2. **System-level architectural rebuild.** Stage A detects full multi-staff systems; Stage B decodes whole systems in one pass with `<staff_idx_N>` marker tokens.

For inference only (PDF → MusicXML), see the upstream [Clarity-OMR](https://github.com/clquwu/Clarity-OMR) repo.

## ⚠️ Hardware Requirement

**This project requires a CUDA-capable GPU. CPU-only execution is not supported.**

See [docs/HARDWARE.md](docs/HARDWARE.md) for tested configurations.

## What it does

```
INPUT: Full-page score image (scan or PDF render)
  │
  ▼
STAGE A — System Detection (YOLO26m)
  │  Detect: full multi-staff systems on the page
  │  Output: ordered list of system bounding boxes, each tagged with its staff count
  │
  ▼
STAGE B — System-Level Recognition (C-RADIOv4-H encoder + RoPE decoder)
  │  Input: cropped system image (multi-staff, all voices in one pass)
  │  Output: token sequence with <staff_idx_N> markers identifying which staff each
  │          note/rest belongs to
  │
  ▼
STAGE C/D — Assembly + MusicXML Serialization
  │  Cross-staff attributes resolved (shared time/key signatures, barline alignment)
  │  Token stream → music21 stream objects → MusicXML export
  │
  ▼
OUTPUT: Valid MusicXML file
```

The earlier per-staff design lost cross-staff coordination signal (ties spanning systems, voice-piano alignment). System-level inputs preserve that context. Single-staff scores are still supported — they're treated as 1-staff systems at inference time.

The legacy per-staff inference pipeline is archived under [archive/per_staff/](archive/per_staff/).

For the full architecture (encoder choice, decoder, grammar FSA, loss, training stability, data augmentation, references), see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project status

| Subproject | Component | Status |
|---|---|---|
| 1 | Stage A system-level YOLO retrain | **Complete** — mAP50 0.995, recall 0.996, precision 0.998 on `mixed_systems_v1` |
| 2 | Kern converter rebuild + Stage 2 v2 trainer optimization | **Complete** — 96.9% kern→OMR token-fidelity audit; Stage 2 v2 val_loss 0.148 at step 4000 |
| 3 | Stage 3 RADIO retrain on system crops | **Training complete** — val_loss 0.148 at step 4000; 5.59h wall on a 5090 |
| 4 | Per-system inference + lieder corpus eval | **Shipped 2026-05-10** — see [docs/EVALUATION.md](docs/EVALUATION.md) |

## Documentation

| Doc | Purpose |
|---|---|
| [QUICKSTART.md](docs/QUICKSTART.md) | Clone → install → smoke inference |
| [HARDWARE.md](docs/HARDWARE.md) | GPU/VRAM/OS requirements |
| [INSTALL.md](docs/INSTALL.md) | cu132 venv setup (Linux + Windows) |
| [TRAINING.md](docs/TRAINING.md) | Stage A + Stage B training commands |
| [EVALUATION.md](docs/EVALUATION.md) | Lieder corpus eval (two-pass) |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full architecture, vocab, FSA, references |
| [paths.md](docs/paths.md) | Repo-relative paths for artifacts |
| [RESULTS.md](docs/RESULTS.md) | Canonical published results (HF release pending) |

## Repository layout

```
configs/                  # training YAML configs
src/
  data/                   # dataset preparation, label derivation
  models/                 # YOLO + RADIO encoder/decoder
  train/                  # training loops, model factory
  tokenizer/              # 495-token music vocabulary
  decoding/               # grammar FSA + beam search
  pipeline/               # cross-system assembly + MusicXML export
  inference/              # SystemInferencePipeline
  eval/                   # checkpoint eval, MusicXML comparison
  cli/                    # one-off CLI entrypoints
scripts/                  # canonical entrypoints (train_yolo, build_*, derive_*, audit_*)
tests/                    # pytest suite (CUDA-gated where required)
eval/                     # corpus eval drivers
docs/                     # documentation (see table above)
archive/                  # per-staff legacy code, archived scripts/results/handoffs
```

For per-file detail, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

GPL-3.0 — see [LICENSE](LICENSE).
