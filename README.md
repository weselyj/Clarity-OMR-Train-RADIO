# Clarity-OMR-Train-RADIO

Training pipeline for an optical music recognition model that turns printed-score images into MusicXML.

This repository is a fork of [**clquwu/Clarity-OMR-Train**](https://github.com/clquwu/Clarity-OMR-Train) — the original training pipeline for [Clarity-OMR](https://github.com/clquwu/Clarity-OMR) (the inference repo). The fork extends the upstream project in two directions:

1. **Encoder swap (DaViT → C-RADIOv4-H).** Replaces the 86M-param ImageNet-pretrained DaViT encoder with NVIDIA's ~700M-param RADIO foundation encoder for richer features (hence the `-RADIO` suffix in this repo's name).
2. **System-level architectural rebuild.** Stage A is being retrained to detect full multi-staff systems (rather than individual staves), and Stage B will be retrained to decode whole systems in one pass with new `<staff_idx_N>` marker tokens. The earlier per-staff retrain confirmed that cropping (not encoder capacity) was the bottleneck on cross-staff coordination, motivating this pivot.

For inference only (PDF → MusicXML), see the upstream [Clarity-OMR](https://github.com/clquwu/Clarity-OMR) repo.

The system-level rebuild has landed on `main` (PR #39, merged 2026-05-08): Stage A system-level YOLO is complete, the kern converter and Stage 2 v2 trainer optimization are complete, and Stage 3 data prep (the four-corpus combined manifest, with the synthetic token-alignment fix) is in place. Stage 3 Phase 0 (encoder-cache infrastructure) is in flight on `feat/stage3-encoder-cache`. The previous per-staff implementation is preserved on older branches.

## What this project does

Given a printed-music score image, the pipeline emits a structurally valid MusicXML file. The architecture splits the problem into three stages:

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

### Why system-level (vs per-staff)

The earlier per-staff design fed individual staff crops to Stage B, then re-stitched the per-staff outputs in Stage C. That recovered most onsets but lost cross-staff coordination signal — ties that span systems, anacrusis split across staves, voice-piano alignment in vocal-piano music. The 2026-05 retrain experiment confirmed this empirically: a clean per-staff-trained checkpoint produced syntactically better outputs but didn't move the headline `onset_f1` metric. System-level inputs preserve cross-staff context that single-staff windows structurally cannot recover.

## Project status

| Subproject | Component | Status |
|---|---|---|
| 1 | Stage A system-level YOLO retrain | **Complete** — mAP50 0.995, recall 0.996, precision 0.998 on `mixed_systems_v1` (gate ≥ 0.95). Weights at `runs/detect/runs/yolo26m_systems/weights/best.pt` |
| 2 | Kern converter rebuild + Stage 2 v2 trainer optimization | **Complete** — 96.9% kern→OMR token-fidelity audit (see `docs/kern_converter_limitations.md`); Stage 2 v2 val_loss 0.148 at step 4000, 5.59h wall on a 5090 (vs 76h spec) |
| 3 | Stage 3 RADIO retrain on system crops | **Data prep complete** — combined 303,663-entry manifest across synthetic_systems (20,583), grandstaff_systems (107,724), primus_systems (87,678), cameraprimus_systems (87,678). Stage 3 design spec approved; Phase 0 encoder-cache infrastructure in flight on `feat/stage3-encoder-cache` |


## Stage A — System detection

The Stage A model is YOLO26m, trained at imgsz=1920 to detect full multi-staff systems. Each detection is a system bbox plus a `staves_in_system` count carried through page metadata.

### Training data

| Corpus | Source | Pages | Systems | How labels are derived |
|---|---|---|---|---|
| `synthetic_v2` | Verovio-rendered MusicXML (OpenScore Lieder + IMSLP) | 6,979 | 21,797 | SVG hierarchy (authoritative) |
| `sparse_augment` | Verovio-rendered (sparse-content augmentation set) | 1,288 | 4,210 | SVG hierarchy (authoritative) |
| `omr_layout_real` | Real scans (AudioLabs typeset corpus) | 919 | ~3,000 | Bracket detection + spatial heuristic |

`mixed_systems_v1` combines all three (20,594 train + 5,142 val pairs, stratified by source).

### Label derivation pipeline

Synthetic and sparse_augment pages use a v15 SVG-hierarchy algorithm that reads the rendered Verovio `<g class="system">` tree directly, sidestepping the Verovio bounding-box rect (which is occasionally undersized). The algorithm:

1. **Parse SVG hierarchy.** Each `<g class="system">` element contains the `<g class="staff">` rows belonging to that system. This tree is authoritative for staff-to-system grouping.
2. **Match per-staff disk labels to SVG rows by y-center distance.** Each staff label inherits the `sys_idx` of its closest SVG row.
3. **Per-row SVG fallback.** If an SVG row has no matching disk label (e.g., a coda fragment whose per-staff labels were never generated), synthesize a bbox from the SVG staff-line geometry plus extended expansion ratios (top=0.4, bottom=1.5) to cover ledger-line chords below the staff.
4. **Single-staff fallback.** Solo treble pieces have no `<g class="system">` elements; each disk label becomes its own one-staff system.
5. **System margins** (production-DPI pixels): `TOP=80`, `BOTTOM=130`, `RIGHT=60`, `LEFTWARD_BRACKET=40`.
6. **Neighbor-aware cap.** When two adjacent systems would expand into each other's territory, both sides cap at the gap midpoint, preventing label overlap on tight layouts.

Real scans (`omr_layout_real`) use a different path because no Verovio SVG exists — visual first-barline detection (`src/data/bracket_detector.py`) plus a spatial fallback (`src/data/derive_systems_from_staves.py`).

The system-bbox derivation is exercised by 23 unit tests in `tests/data/test_build_system_yolo_objects_v15.py` and `tests/data/test_generate_synthetic_systems.py`.

### Training command

```bash
python scripts/train_yolo.py \
  --model yolo26m.pt \
  --data data/processed/mixed_systems_v1/data.yaml \
  --epochs 100 --imgsz 1920 --batch 4 --workers 6 \
  --amp --nan-guard --noise --noise-warmup-steps 2000 \
  --project runs/detect/runs --name yolo26m_systems --patience 30
```

Notable flags:

- `--noise-warmup-steps 2000` — ramps scan-noise augmentation probability from 0 to full over the first 2000 steps. Avoids the early-training NaN failure mode where `cls_loss × noise` blows up.
- `--nan-guard` — zeroes individual NaN/Inf gradients per batch (instead of skipping the whole step) so the rare warmup gradient explosion doesn't disrupt training.
- The data-aug pipeline (scan-noise + page-curvature) lives in `src/train/scan_noise.py`.

Final training metrics: **mAP50 0.995, recall 0.996, precision 0.998** at 100 epochs on `mixed_systems_v1`. Weights at `runs/detect/runs/yolo26m_systems/weights/best.pt`.

## Stage B — System-level recognition

The recognition model uses a **C-RADIOv4-H encoder** paired with a **custom autoregressive Transformer decoder** with RoPE. Stage B reads a *system crop* (all staves of one system, stacked vertically) and emits a token sequence where each note carries a `<staff_idx_N>` marker identifying its staff.

The encoder choice is settled — a per-staff RADIO retrain ran in 2026-05 with a Flat outcome (`onset_f1 = 0.2144`), confirming that *cropping* was the bottleneck rather than the encoder.

**Stage 2 v2** (vocab extension + system-level training warmup) completed in 2026-05: val_loss 0.148 at step 4000 in 5.59h on a 5090 (vs the spec'd 76h with the unoptimized config). The trainer optimization plan is at `docs/superpowers/plans/2026-05-06-radio-stage2-trainer-optimization.md`. Init checkpoint for Stage 3 is `checkpoints/full_radio_stage2_systems_v2/_best.pt`.

**Stage 3** (full system-level retrain on the four-corpus mix with an encoder-cache hybrid) is in active development. Data prep is complete (the 303,663-entry combined manifest above). Phase 0 — encoder-cache infrastructure (`src/data/encoder_cache.py`, `scripts/build_encoder_cache.py`, tier-grouped sampler, cached/live tier dispatch) — is in flight on `feat/stage3-encoder-cache`, with the cache build currently running on the GPU box.

### Encoder: C-RADIOv4-H (~700M parameters)

[NVIDIA RADIO](https://github.com/NVlabs/RADIO) (Reduced All-Domain Into One) is a vision foundation encoder distilled from CLIP, DINOv2, SAM, and DFN. C-RADIOv4-H is the Huge variant — designed to give a single encoder competitive performance across dense vision tasks without per-task pretraining.

- **Hidden dim 1280** (vs DaViT's 768).
- **Patch-16 → 12 × (W/16) feature grid** (vs DaViT's stride-32 → 6 × W/32) — ~4× more memory tokens for cross-attention to consume; the cross-attention has no length cap so this is fine.
- **Loaded via `torch.hub`** from `NVlabs/RADIO` at construction time (`version="c-radio_v4-h"`).
- **bf16 autocast** on the forward pass (RADIO's expected path on the 5090); output cast back to caller dtype.
- **Resolution-snapped input**: RADIO requires HxW divisible by its patch size; the wrapper rounds the input crop to the nearest supported resolution before forward.
- A **deformable attention layer** (3M params) is added on top for dense-notation handling (clustered accidentals, chords, grace notes).

System crops are grayscale, height up to ~768px (3-4 staves stacked), width preserved up to 2048px.

### Positional bridge

1. **2D sinusoidal positional encoding** added to encoder feature map (separate x/y frequencies for time and pitch position).
2. **Flatten** in raster-scan order.
3. **Linear projection** from encoder dim 1280 → decoder dim 768 + LayerNorm.

### Decoder: Custom Transformer

| Property | Value |
|---|---|
| Embedding dim | 768 |
| Heads | 12 |
| FFN dim | 2048 (SwiGLU: 8/3 × 768) |
| Layers | 8 |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Positional encoding | RoPE (smooth sequence extrapolation) |
| Max decode length | 768 tokens (system-level, was 512 per-staff) |
| Vocabulary | ~495 tokens (487 music + 8 staff-index markers) |

Full cross-attention over encoder output.

### DoRA adaptation

All linear layers in both encoder and decoder are adapted with [DoRA](https://arxiv.org/abs/2402.09353) rank-64. DoRA decomposes weight updates into magnitude and direction, outperforming standard LoRA on the same parameter budget.

```python
adapter_config = {
    "method": "DoRA",
    "r": 64,
    "lora_alpha": 64,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "out_proj",      # Self-attention
        "gate_proj", "up_proj", "down_proj",            # SwiGLU MLP
        "cross_attn_q", "cross_attn_k",                 # Cross-attention
        "cross_attn_v", "cross_attn_out"
    ],
    "lora_dropout": 0.10,
    "bias": "none"
}
```

Trained from scratch (not adapted): deformable attention, positional bridge, token embeddings, LM head, decoder norm, pitch contour head, **staff-index marker embeddings**.

## Token vocabulary (~495 tokens)

A custom domain-specific vocabulary. Music-aware encoding achieves ~4× lower error rate than character-level encoding (Alfaro-Contreras et al., WORMS 2023).

| Category | Count | Examples |
|---|---|---|
| Structural | 17 | `<bos>`, `<eos>`, `<measure_start>`, `<voice_1>`, `<chord_start>`, `<tuplet_3>` |
| **Staff-index markers** (new, v2) | 8 | `<staff_idx_0>` … `<staff_idx_7>` |
| Pitch | 87 | `note-C4`, `note-F#5`, `note-Bb3`, `rest` (C2–C7) |
| Grace notes | 35 | `gracenote-C4`, `gracenote-D5` |
| Duration | 9 | `_whole`, `_quarter`, `_sixteenth`, `_dot`, `_double_dot` |
| Clefs | 7 | `clef-G2`, `clef-F4`, `clef-C3`, `clef-G2_8vb` |
| Key signatures | 31 | `keySignature-CM`, `keySignature-Am`, `keySignature-none` |
| Time signatures | 15 | `timeSignature-4/4`, `timeSignature-6/8`, `timeSignature-C` |
| Barlines | 6 | `barline`, `final_barline`, `repeat_start` |
| Articulations | 12 | `staccato`, `accent`, `fermata`, `trill`, `mordent` |
| Dynamics | 10 | `dynamic-pp`, `dynamic-f`, `dynamic-sfz` |
| Connections | 8 | `tie_start`, `tie_end`, `slur_start`, `cresc_start` |
| Other notation | 15 | `ottava_8va_start`, `pedal_start`, `trem_single`, `coda`, `segno` |
| Tempo | 50 | `tempo-Allegro`, `tempo-Andante`, `tempo-Presto` |
| Expression | 74 | `expr-dolce`, `expr-cantabile`, `expr-rit`, `expr-a_tempo` |

Enharmonic spellings are kept separate (`C#` vs `Db`) — essential for correct MusicXML output.

### Encoding example (system-level)

A 3-staff system (vocal + piano grand staff) emits tokens like:

```
<staff_idx_0> clef-G2 keySignature-DM timeSignature-4/4
<measure_start>
  <voice_1> note-F#5 _quarter note-E5 _quarter ...
<measure_end>
<staff_idx_1> clef-G2 keySignature-DM
<measure_start>
  <voice_1> <chord_start> note-D4 note-F#4 note-A4 <chord_end> _half ...
<measure_end>
<staff_idx_2> clef-F4 keySignature-DM
<measure_start>
  <voice_1> note-D2 _half note-A2 _half
<measure_end>
```

The `<staff_idx_N>` marker is what makes system-level decoding possible: the decoder learns to interleave tokens for multiple staves while keeping each staff's stream coherent.

## Grammar FSA (constrained decoding)

A finite-state automaton runs during beam search, producing a binary mask over the vocabulary at each step to enforce structurally valid output.

**Hard constraints (invalid tokens masked to -inf):**

| Rule | Description |
|---|---|
| Token sequence validity | After `<measure_start>`, only note/rest/chord/voice/attribute tokens. After `<staff_end>`, only `<eos>` or another `<staff_idx_N>`. |
| Beat consistency | Track cumulative duration per measure per staff. Force `<measure_end>` when beats are full. |
| Chord well-formedness | Between `<chord_start>` and `<chord_end>`, only pitch tokens. |
| Voice consistency | Voice tokens must alternate properly with explicit voice switching. |
| Staff index validity | `<staff_idx_N>` must reference a staff present in the current system (`N < staves_in_system`). |

**Soft constraints (logit penalties):**

| Rule | Penalty | Description |
|---|---|---|
| Pitch range plausibility | -5.0 | Pitches outside normal range for current clef. |
| Accidental propagation | -3.0 | Contradicting accidentals within a measure. |
| Measure balance | -2.5 × diff | Penalizes `<measure_end>` when beats don't sum correctly. |
| CV note count prior | -3.0 per excess | Penalizes note emissions exceeding computer-vision detection count. |
| CV pitch prior | -0.45 to -4.5 | Multi-tiered penalty for pitch disagreement with CV detection. |

## Loss function

- **Primary:** Token-level cross-entropy with label smoothing (ε=0.05)
- **Auxiliary:** Pitch contour consistency (λ=0.1) — a 2-layer MLP (768→128→3) predicting pitch direction (up/down/same) between adjacent notes, providing gradient signal to the encoder-decoder interface

**Total loss:** L = L_CE + 0.1 · L_contour

## Training stability

- BF16 mixed precision (Stage B); AMP for YOLO
- Gradient clipping at max norm 1.0
- `--nan-guard` (Stage A): zeroes individual NaN/Inf gradients per batch
- Checkpoint every 1,000 steps
- Validation every 500 steps on held-out 5% split
- Gradient norm monitoring per module group

## Data augmentation

| Augmentation | Parameters | Probability | Purpose |
|---|---|---|---|
| Rotation + Scale | ±2.0°, scale 0.92–1.08 | 80% | Scanner misalignment, DPI variation |
| Brightness | ±10% | 65% | Scanner exposure |
| Contrast | ±12% | 65% | Paper/ink density |
| Gaussian blur | σ ∈ [0, 1.0], kernel 3×3 | 45% | Scanner defocus |
| JPEG compression | quality 70–95 | 25% | PDF re-render artifacts |
| Resolution downsample | 85–100% + resize | 25% | Low-DPI simulation |
| Salt-and-pepper noise | 0.1–0.25% of pixels | 30% | Minor scan artifacts |

Applied online during training (not pre-generated). Stage A uses an additional scan-noise + page-curvature pipeline (GridDistortion, ElasticTransform) ramping up over `--noise-warmup-steps`.

## Evaluation

### Metrics

- **Symbol Error Rate (SER):** Edit distance between predicted and ground-truth token sequences, normalized by ground-truth length.
- **Onset F1 (mir_eval):** Note-level precision/recall/F1 (onset_tolerance=50ms, pitch_tolerance=50 cents).
- **Pitch / rhythm accuracy:** Per-note correctness rates.
- **Key/time signature accuracy:** Exact-match.
- **Structural F1:** Barlines, measure boundaries, voice assignments.
- **Stratified by `staves_in_system`** (1, 2, 3, 4+) — checks system-level model holds up at increasing staff counts.

### Decision gates

| Outcome | Mean lieder onset_f1 | Action |
|---|---|---|
| Strong | ≥ 0.30 | Ship: PR, write-up, follow-ups for full-quality decode + HF release prep |
| Mixed | 0.241 ≤ x < 0.30 | Beats DaViT baseline but not transformative; investigate residual error before next major iteration |
| Flat / regressed | < 0.241 | System-level approach also failed; pivot to classical pipeline (Audiveris-style) |

## Repository structure

```
├── configs/                                        # Training YAML configs
│   ├── splits.yaml                                 # Train/val/test split definitions
│   ├── train_stage1.yaml                           # Stage 1 monophonic (legacy DaViT)
│   ├── train_stage1_radio.yaml                     # Stage 1 v2 RADIO config
│   ├── train_stage2.yaml                           # Stage 2 polyphonic (legacy)
│   ├── train_stage2_radio.yaml                     # Stage 2 RADIO config
│   ├── train_stage2_radio_systems.yaml             # Stage 2 v2 system-level config (current)
│   ├── train_stage3.yaml                           # Stage 3 legacy config
│   └── train_stage3_radio.yaml                     # Stage 3 RADIO config (Plan B will add a system-level variant)
│
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py                   # Synthetic data generation + v15 system-bbox derivation + per-staff manifest writer
│   │   ├── yolo_aligned_systems.py                 # Multi-staff system-level helpers (Plan A consumer)
│   │   ├── yolo_aligned_crops.py                   # Per-staff variant (helpers reused by systems module)
│   │   ├── derive_systems_from_staves.py           # Spatial heuristic for AudioLabs (real scans)
│   │   ├── bracket_detector.py                     # Visual first-barline detection (real scans)
│   │   ├── multi_dpi.py                            # Verovio rendering at multiple DPIs
│   │   ├── convert_tokens.py                       # MusicXML ↔ token sequence conversion (kern + MusicXML)
│   │   ├── kern_validation.py                      # Kern token-sequence validation
│   │   ├── omr_layout_import.py                    # AudioLabs typeset corpus import
│   │   ├── sparse_augment.py                       # Sparse-content augmentation set helpers
│   │   ├── build_mixed_dataset.py                  # Mixed-corpus dataset assembly
│   │   ├── dataset_audit.py                        # Dataset-level audit utilities
│   │   ├── index.py                                # Dataset indexing and manifest building
│   │   └── filter_low_ink_samples.py               # Filter out low-quality training samples
│   │
│   ├── models/
│   │   ├── radio_stage_b.py                        # C-RADIOv4-H encoder + RoPE decoder (current Stage B)
│   │   ├── davit_stage_b.py                        # DaViT encoder (decoder/contour-head reused by RADIO module)
│   │   ├── florence_stage_b.py                     # Florence experiment (archived)
│   │   ├── system_postprocess.py                   # Inference-time system postprocessing
│   │   └── yolo_stage_a.py                         # YOLO26m system-detection wrapper
│   │
│   ├── train/
│   │   ├── train.py                                # Main training loop + DoRA setup + Stage 2 v2 perf optimizations
│   │   ├── train_yolo_stage_a.py                   # YOLO fine-tuning script
│   │   ├── scan_noise.py                           # Scan-noise + page-curvature augmentation (YOLO)
│   │   ├── model_factory.py                        # Model instantiation + checkpoint loading
│   │   ├── build_focus_manifest.py                 # Focus manifest builder for staged training
│   │   ├── check_training_data.py                  # Pre-flight training data sanity checks
│   │   ├── monitor_training.py                     # Training progress monitoring
│   │   └── monitor_dashboard.py                    # Training dashboard
│   │
│   ├── tokenizer/
│   │   └── vocab.py                                # ~495-token music + 8 staff-marker vocabulary
│   │
│   ├── decoding/
│   │   ├── grammar_fsa.py                         # Grammar FSA for constrained decoding
│   │   └── beam_search.py                         # Beam search with FSA integration
│   │
│   ├── pipeline/
│   │   ├── assemble_score.py                      # Cross-system assembly (Stage C)
│   │   └── export_musicxml.py                     # Token → music21 → MusicXML (Stage D)
│   │
│   ├── eval/
│   │   ├── evaluate_stage_b_checkpoint.py         # Checkpoint evaluation
│   │   ├── compare_musicxml.py                    # MusicXML comparison (mir_eval metrics)
│   │   ├── metrics.py                             # Training metrics computation
│   │   ├── run_eval.py                            # Batch evaluation runner
│   │   ├── run_stage_a_only.py                    # Stage-A-only eval entry point
│   │   ├── run_subproject2_gate.py                # Subproject 2 acceptance gate
│   │   └── tune_penalties.py                      # Grammar penalty tuning
│   │
│   └── pdf_to_musicxml.py                          # End-to-end pipeline orchestrator
│
├── scripts/
│   │  # Stage A YOLO + dataset builders
│   ├── train_yolo.py                               # Stage A training entry point
│   ├── build_mixed_v2_systems.py                   # Build mixed_systems_v1 (Stage A training data)
│   ├── derive_audiolabs_systems.py                 # Real-scan system-label derivation
│   ├── derive_sparse_augment_systems.py            # sparse_augment system-label derivation (v15)
│   ├── rederive_synthetic_v2_systems.py            # In-place relabeling without re-render
│   ├── verify_v15_labels.py                        # Validate production labels against v15 reference
│   ├── visualize_audiolabs_systems.py              # Spot-check overlay generator
│   ├── smoketest_bracket_detector.py
│   │
│   │  # Plan A — Stage 3 system-level data prep
│   ├── build_synthetic_systems_v1.py               # System-level builder for synthetic_v2
│   ├── build_grandstaff_systems.py                 # System-level builder for grandstaff
│   ├── build_stage3_combined_manifest.py           # Combine the 4 corpora into one Stage 3 manifest
│   ├── retokenize_with_staff_markers.py            # Single-staff retokenizer for primus/cameraprimus
│   ├── rederive_synthetic_v2_per_staff_manifest.py # In-place per-staff manifest fix (alignment fix)
│   │
│   │  # Audits
│   ├── audit_per_staff_alignment.py                # Per-staff manifest alignment audit
│   ├── audit_token_miss_buckets.py                 # Categorize residual token_miss drops
│   ├── audit_kern_fidelity.py                      # Kern→OMR token fidelity audit (96.9% on rebuild)
│   └── audit_kern_manifests.py                     # Kern manifest sanity check
│
├── tests/
│   ├── data/                                       # Data-layer tests (208+ pass)
│   ├── train/                                      # Trainer tests (run on GPU box; need torch)
│   ├── models/                                     # Model-layer tests
│   └── eval/                                       # Eval-pipeline tests
│
├── docs/
│   ├── omr-final-plan.md                           # Full architecture design document
│   ├── kern_converter_limitations.md               # Subproject 2 known gaps
│   ├── stage_a_brace_margin_known_gap.md           # Stage A v15 cosmetic gap
│   ├── TRAINING_COMMANDS.md                        # Training command reference
│   ├── TRAINING_COMMANDS_UBUNTU.md                 # Ubuntu-specific training setup
│   └── superpowers/
│       ├── plans/                                  # Implementation plans (one per workstream)
│       └── audits/                                 # Audit findings docs
│
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (5090 / H100 / A100; 48+ GB VRAM recommended for Stage B)
- PyTorch with CUDA

### Setup

```bash
git clone https://github.com/weselyj/Clarity-OMR-Train-RADIO.git
cd Clarity-OMR-Train-RADIO

# Production setup (Windows): scripts/setup_venv_cu132.ps1 pulls torch nightly cu132,
# cuDNN 9.21.01, project deps, and the sitecustomize for DLL path resolution.

# Manual cu132 install (non-Windows or scripting fails):
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132

# Rollback / reproducibility (cu128, kept on disk as venv/):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

### Data preparation

1. **Download source datasets:**
   - [PrIMuS](https://grfia.dlsi.ua.es/primus/) → `data/primus/`
   - [Camera-PrIMuS](https://grfia.dlsi.ua.es/primus/) → `data/camera-primus/`
   - [GrandStaff](https://github.com/multiscore/GrandStaff) → `data/grandstaff/`
   - [OpenScore Lieder](https://github.com/OpenScore/Lieder) → `data/openscore-lieder/`

2. **Render synthetic_v2 from MusicXML** (Verovio + ImageMagick):
   ```bash
   python src/data/generate_synthetic.py \
     --output data/processed/synthetic_v2/ \
     --num-pages 7000
   ```
   Produces SVGs, PNGs at multiple DPIs, per-staff labels, and v15 system-level labels.

3. **Derive system labels for sparse_augment and AudioLabs:**
   ```bash
   python scripts/derive_sparse_augment_systems.py \
     --pages-dir data/processed/sparse_augment/pages
   python scripts/derive_audiolabs_systems.py
   ```

4. **Build the mixed Stage A dataset:**
   ```bash
   python scripts/build_mixed_v2_systems.py
   ```
   Produces `data/processed/mixed_systems_v1/{train,val}/{images,labels}/` plus `data.yaml` and `audit.json`.

## Training

### Stage A — System-level YOLO

```bash
python scripts/train_yolo.py \
  --model yolo26m.pt \
  --data data/processed/mixed_systems_v1/data.yaml \
  --epochs 100 --imgsz 1920 --batch 4 --workers 6 \
  --amp --nan-guard --noise --noise-warmup-steps 2000 \
  --project runs/detect/runs --name yolo26m_systems --patience 30
```

~10–12h on a 5090. Gate: val mAP50 ≥ 0.95.

### Stage B — System-level RADIO (Stage 2 v2, Stage 3 in development)

```bash
# Stage 1 v2 — per-staff RADIO (completed; for re-runs only)
python src/train/train.py --config configs/train_stage1_radio.yaml

# Stage 2 v2 — system-level vocab-extension warmup (completed; init checkpoint for Stage 3)
python src/train/train.py --config configs/train_stage2_radio_systems.yaml

# Stage 3 — full system-level retrain with encoder-cache hybrid (in development)
# Requires Phase 0 encoder cache infrastructure on feat/stage3-encoder-cache.
# The Stage 3 design spec (encoder caching, hybrid 70/10/10/10 mix, 4500 opt-step target,
# Phase 0d correctness + throughput gates) is at:
#   docs/superpowers/specs/2026-05-07-radio-stage3-design.md (in user repo)
```

See `docs/TRAINING_COMMANDS.md` for detailed commands and options. Plans + audits for the recent
work live under `docs/superpowers/{plans,audits}/`.

## Evaluation

```bash
# Evaluate a Stage B checkpoint
python src/eval/evaluate_stage_b_checkpoint.py --checkpoint checkpoints/stage3_best.pt

# Compare MusicXML outputs using mir_eval metrics
python src/eval/compare_musicxml.py ground_truth.musicxml predicted.musicxml
```

## Architecture rationale (selected)

- **System-level over per-staff inputs:** Per-staff inference can't recover cross-staff coordination (ties spanning systems, voice-piano alignment). Confirmed empirically — a clean per-staff retrain in 2026-05 produced cleaner per-staff outputs (Stage D unknown_tokens dropped 8 → 1) but didn't move `onset_f1`. System-level is the architectural fix.
- **C-RADIOv4-H + custom decoder over Florence-2 fine-tuning:** General-purpose VLMs fail entirely at OMR transcription (Calvo-Zaragoza et al., WORMS 2024). Florence-2's 51K NL tokenizer conflicts with our 495-token music vocabulary. The SMT/SMIReT architecture (5.92% SER) validates the encoder + task-specific decoder pattern. RADIO was selected over the original DaViT for richer features (~700M params, hidden 1280) — the per-staff retrain confirmed cropping (not encoder capacity) was the bottleneck, motivating the move to system-level inputs.
- **Custom 495-token vocabulary over sub-word tokenization:** Music-aware encoding achieves 16.4% CER vs 62.3% character-level and 39.7% learned sub-word (Alfaro-Contreras et al., WORMS 2023).
- **Grammar FSA over statistical language models:** Statistical LMs cannot enforce structural constraints like beat consistency (Torras et al., WORMS 2022).
- **RoPE over learned positional embeddings:** Smooth extrapolation beyond training sequence length for unusually dense systems.
- **DoRA rank-64 over standard LoRA:** Weight-decomposed adaptation outperforms standard LoRA on fine-tuning benchmarks.
- **SVG-tree label derivation over Verovio bounding-box rect:** Verovio's bounding-box rect is occasionally undersized (the trailing brace fails to encompass its bottom staff); the SVG element tree is structurally correct in all cases. Verified across 15 rounds of overlay review on 50+ pages.

## References

- Calvo-Zaragoza et al., "Can multimodal LLMs read music score images?" WORMS 2024
- Ríos-Vila et al., "Towards Sheet Music Information Retrieval: SMIReT" WORMS 2024
- Dvořák et al., "Layout Analysis with YOLOv8" WORMS 2024
- Alfaro-Contreras et al., "Audio-Music Notation-Lyrics Transcription" WORMS 2023
- Torras et al., "Integration of Language Models into Seq2Seq for Handwritten Music Recognition" WORMS 2022
- Calvo-Zaragoza & Rizo, "End-to-End Neural OMR" Applied Sciences 2018
- Calvo-Zaragoza, Hajič & Pacha, "Understanding Optical Music Recognition" ACM Computing Surveys 2020

## License

GPL-3.0 — see [LICENSE](LICENSE).
