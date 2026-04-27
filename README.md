# Clarity-OMR-Train

Training code and pipeline for the [Clarity-OMR](https://github.com/clquwu/Clarity-OMR) optical music recognition model.

For **inference only** (PDF to MusicXML), see [Clarity-OMR](https://github.com/clquwu/Clarity-OMR).

## System Architecture

Clarity-OMR is a **4-stage pipeline** designed for clean, modern, typeset sheet music:

```
INPUT: Full-page score image (scan or PDF render)
  │
  ▼
STAGE A — Page Analysis (YOLOv8m)
  │  Detect: staves, system brackets, barlines, title/page regions
  │  Output: ordered list of staff bounding boxes grouped by system
  │
  ▼
STAGE B — Staff-Level Recognition (DaViT encoder + custom RoPE decoder)
  │  Input: individual cropped staff image (192px height, up to 2048px width)
  │  Output: token sequence per staff (~487-token music vocabulary)
  │
  ▼
STAGE C — Constrained Decoding + Assembly
  │  Grammar FSA validates each staff's token sequence during beam search
  │  Staves assembled into systems using Stage A spatial metadata
  │  Cross-staff attributes resolved (shared time/key signatures, barline alignment)
  │
  ▼
STAGE D — MusicXML Serialization
  │  Token sequences → music21 stream objects → MusicXML export
  │
  ▼
OUTPUT: Valid MusicXML file
```

### Why pipeline over end-to-end

- Staff-level recognition at 192px height preserves full detail per note — end-to-end full-page would require downsampling, losing the fine detail that distinguishes sharps from naturals or eighth notes from sixteenths.
- Staff detection on clean typeset notation is essentially solved (0.991 mAP50, Dvořák et al., WORMS 2024).
- Each stage can be debugged, evaluated, and improved independently.
- Scales to any number of staves without sequence length constraints.

## Model Architecture — Stage B (Core)

The core recognition model uses a **DaViT-pretrained encoder** paired with a **custom autoregressive Transformer decoder** with RoPE positional encoding.

### Encoder: DaViT (86M parameters)

[DaViT](https://github.com/dingmyu/davit) (Dual Attention Vision Transformer) alternates between spatial-window and channel-group self-attention:

- **Spatial attention** captures local glyph structure (noteheads, stems, beams).
- **Channel attention** captures global patterns (staff line positions, key signature context across the full width).
- **Pretrained on ImageNet** — general visual features transfer well to the high-contrast, geometric domain of printed notation.
- A **deformable attention layer** (3M params) is added before the final encoder output for handling dense notation (clustered accidentals, chords, grace notes).

Input: grayscale staff crop, height 192px, width preserved up to 2048px. Feature map: spatial downsampling factor 32 → 6 × (W/32) grid.

### Positional Bridge

1. **2D sinusoidal positional encoding** added to encoder feature map (separate x/y frequencies for horizontal time-position and vertical pitch-position).
2. **Flatten** in raster-scan order.
3. **Linear projection** from encoder dim 768 → decoder dim 768 + LayerNorm.

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
| Max decode length | 512 tokens |
| Vocabulary | ~487 custom music tokens |

Full cross-attention over encoder output (no windowing needed).

## Token Vocabulary (~487 tokens)

A custom domain-specific vocabulary. Music-aware encoding achieves ~4× lower error rate than character-level encoding (Alfaro-Contreras et al., WORMS 2023).

| Category | Count | Examples |
|---|---|---|
| Structural | 17 | `<bos>`, `<eos>`, `<measure_start>`, `<voice_1>`, `<chord_start>`, `<tuplet_3>` |
| Pitch | 87 | `note-C4`, `note-F#5`, `note-Bb3`, `rest` (C2–C7, 17 pitch classes per octave) |
| Grace notes | 35 | `gracenote-C4`, `gracenote-D5` (natural notes only, octaves 2–6) |
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

### Encoding Example

```
<staff_start> clef-G2 keySignature-DM timeSignature-4/4
<measure_start>
  <voice_1> note-F#5 _quarter note-E5 _quarter note-D5 _quarter note-C#5 _quarter
  <voice_2> <chord_start> note-D4 note-F#4 note-A4 <chord_end> _half
            <chord_start> note-A3 note-E4 note-G4 <chord_end> _half
<measure_end>
<staff_end>
```

## DoRA (Weight-Decomposed Low-Rank Adaptation)

All linear layers in both encoder and decoder are adapted with [DoRA](https://arxiv.org/abs/2402.09353) rank-64. DoRA decomposes weight updates into magnitude and direction, outperforming standard LoRA.

```python
adapter_config = {
    "method": "DoRA",
    "r": 64,
    "lora_alpha": 64,           # α/r = 1.0
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "out_proj",     # Self-attention
        "gate_proj", "up_proj", "down_proj",           # SwiGLU MLP
        "cross_attn_q", "cross_attn_k",                # Cross-attention
        "cross_attn_v", "cross_attn_out"
    ],
    "lora_dropout": 0.10,
    "bias": "none"
}
```

Fully trainable new modules (not adapted — trained from scratch): deformable attention, positional bridge, token embeddings, LM head, decoder norm, pitch contour head.

## Grammar FSA (Constrained Decoding)

A finite-state automaton runs during beam search, producing a binary mask over the vocabulary at each step to enforce structurally valid output.

**Hard constraints (invalid tokens masked to -inf):**

| Rule | Description |
|---|---|
| Token sequence validity | After `<measure_start>`, only note/rest/chord/voice/attribute tokens. After `<staff_end>`, only `<eos>`. |
| Beat consistency | Track cumulative duration per measure. Force `<measure_end>` when beats are full. |
| Chord well-formedness | Between `<chord_start>` and `<chord_end>`, only pitch tokens (duration follows after close). |
| Voice consistency | Voice tokens must alternate properly with explicit voice switching. |

**Soft constraints (logit penalties):**

| Rule | Penalty | Description |
|---|---|---|
| Pitch range plausibility | -5.0 | Pitches outside normal range for current clef. |
| Accidental propagation | -3.0 | Contradicting accidentals within a measure. |
| Measure balance | -2.5 × diff | Penalizes `<measure_end>` when beats don't sum correctly. |
| CV note count prior | -3.0 per excess | Penalizes note emissions exceeding computer vision detection count. |
| CV pitch prior | -0.45 to -4.5 | Multi-tiered penalty for pitch disagreement with CV detection (self-disabling if CV unreliable). |

## Datasets

| # | Dataset | Size | Purpose |
|---|---|---|---|
| 1 | **PrIMuS** | 87,678 incipits | Stage 1 — monophonic glyph→token mapping |
| 2 | **Camera-PrIMuS** | 87,678 incipits | Mixed into Stage 1 at 30% for scan robustness |
| 3 | **GrandStaff** | ~8,000 samples | Stage 2 — polyphonic piano grand staff pairs |
| 4 | **OpenScore Lieder** | ~1,200 pieces | Stage 3 + primary evaluation set |
| 5 | **Synthetic Full-Page** | ~20,000 pages (~120K staff crops) | Stage 3 — orchestral, chamber, all complexity levels |

Synthetic data is generated from MusicXML sources (MuseScore corpus, IMSLP public domain) rendered via Verovio with 3 different visual configs per score.

**Target distribution for synthetic data:**
- 40% piano solo and piano+voice
- 25% orchestral full scores (10-20+ staves)
- 20% chamber music (2-5 instruments)
- 10% choral (SATB + piano)
- 5% solo instrument + piano accompaniment


### Loss Function

- **Primary:** Token-level cross-entropy with label smoothing (ε=0.05)
- **Auxiliary:** Pitch contour consistency (λ=0.1) — a 2-layer MLP (768→128→3) predicting pitch direction (up/down/same) between adjacent notes, providing gradient signal to the encoder-decoder interface

**Total loss:** L = L_CE + 0.1 · L_contour

### Training Stability

- BF16 mixed precision
- Gradient clipping at max norm 1.0
- Checkpoint every 1,000 steps
- Validation every 500 steps on held-out 5% split
- Gradient norm monitoring per module group

### Total Training Budget

| Component | Time |
|---|---|
| Synthetic data generation | ~4 hours (CPU) |
| YOLO training | ~2 hours |
| Stage 1: Monophonic (15 epochs) | ~6 hours |
| Stage 2: Polyphonic (30 epochs) | ~6 hours |
| Stage 3: Full complexity (20 epochs) | ~16 hours |
| Evaluation | ~4 hours |

## Data Augmentation

Only augmentations simulating realistic variations of clean printed scores:

| Augmentation | Parameters | Probability | Purpose |
|---|---|---|---|
| Rotation + Scale | ±2.0°, scale 0.92–1.08 | 80% | Scanner misalignment, DPI variation |
| Brightness | ±10% | 65% | Scanner exposure |
| Contrast | ±12% | 65% | Paper/ink density |
| Gaussian blur | σ ∈ [0, 1.0], kernel 3×3 | 45% | Scanner defocus |
| JPEG compression | quality 70–95 | 25% | PDF re-render artifacts |
| Resolution downsample | 85–100% + resize | 25% | Low-DPI simulation |
| Salt-and-pepper noise | 0.1–0.25% of pixels | 30% | Minor scan artifacts |

Applied online during training (not pre-generated).

## Evaluation

### Metrics

- **Symbol Error Rate (SER):** Edit distance between predicted and ground-truth token sequences, normalized by ground-truth length.
- **Pitch accuracy:** % of notes with correct pitch (ignoring duration).
- **Rhythm accuracy:** % of notes with correct duration (ignoring pitch).
- **Key/time signature accuracy:** Exact-match rate.
- **Structural F1:** F1 on barlines, measure boundaries, voice assignments.
- **Musical similarity:** Note-level precision, recall, F1 using [mir_eval](https://github.com/mir-evaluation/mir_eval) transcription metrics (onset_tolerance=50ms, pitch_tolerance=50 cents).

## Repository Structure

```
├── configs/                          # Training YAML configs
│   ├── splits.yaml                   # Train/val/test split definitions
│   ├── train_stage1.yaml             # Stage 1 monophonic config
│   ├── train_stage2.yaml             # Stage 2 polyphonic config
│   ├── train_stage3.yaml             # Stage 3 full complexity config
│   └── train_stage2_*.yaml           # Stage 2 variants (rank-64, repair, finetune)
│
├── src/
│   ├── models/
│   │   ├── davit_stage_b.py          # DaViT encoder + RoPE decoder architecture
│   │   ├── florence_stage_b.py       # Florence-2 (deprecated, kept for reference)
│   │   └── yolo_stage_a.py           # YOLOv8 staff detection wrapper
│   │
│   ├── train/
│   │   ├── train.py                  # Main training loop + DoRA setup
│   │   ├── train_yolo_stage_a.py     # YOLO fine-tuning script
│   │   ├── model_factory.py          # Model instantiation + checkpoint loading
│   │   ├── build_focus_manifest.py   # Build focused training manifests
│   │   ├── check_training_data.py    # Data validation utilities
│   │   ├── monitor_training.py       # Training progress monitoring
│   │   └── monitor_dashboard.py      # Training dashboard
│   │
│   ├── tokenizer/
│   │   └── vocab.py                  # 487-token music vocabulary definition
│   │
│   ├── decoding/
│   │   ├── grammar_fsa.py            # Grammar FSA for constrained decoding
│   │   └── beam_search.py            # Beam search with FSA integration
│   │
│   ├── data/
│   │   ├── generate_synthetic.py     # Synthetic data generation pipeline
│   │   ├── convert_tokens.py         # MusicXML ↔ token sequence conversion
│   │   ├── index.py                  # Dataset indexing and manifest building
│   │   └── filter_low_ink_samples.py # Filter out low-quality training samples
│   │
│   ├── pipeline/
│   │   ├── assemble_score.py         # Cross-staff assembly (Stage C)
│   │   └── export_musicxml.py        # Token → music21 → MusicXML (Stage D)
│   │
│   ├── eval/
│   │   ├── evaluate_stage_b_checkpoint.py  # Checkpoint evaluation
│   │   ├── compare_musicxml.py       # MusicXML comparison (mir_eval metrics)
│   │   ├── metrics.py                # Training metrics computation
│   │   ├── run_eval.py               # Batch evaluation runner
│   │   ├── tune_penalties.py         # Grammar penalty tuning
│   │   └── summarize_stage_b_failures.py   # Error analysis
│   │
│   ├── cv/
│   │   ├── staff_analyzer.py         # Staff line detection and analysis
│   │   └── priors.py                 # Visual priors for notation
│   │
│   ├── pdf_to_musicxml.py            # End-to-end pipeline orchestrator
│   └── cli.py                        # CLI argument parsing
│
├── docs/
│   ├── omr-final-plan.md             # Full architecture design document
│   ├── TRAINING_COMMANDS.md          # Training command reference
│   └── TRAINING_COMMANDS_UBUNTU.md   # Ubuntu-specific training setup
│
├── analyze_data.py                   # Dataset analysis utilities
└── requirements.txt                  # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (H100/A100 recommended, any GPU with 48+ GB VRAM works)
- PyTorch with CUDA

### Setup

```bash
git clone https://github.com/clquwu/Clarity-OMR-Train.git
cd Clarity-OMR-Train

# Production setup (Windows): run scripts/setup_venv_cu132.ps1 — pulls torch nightly cu132,
# cuDNN 9.21.01, project deps, and drops the sitecustomize for DLL path resolution.

# Manual cu132 install (if not on Windows or scripting fails):
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132

# Rollback / reproducibility (cu128, kept on disk as venv/):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download datasets:
   - [PrIMuS](https://grfia.dlsi.ua.es/primus/) — place in `data/primus/`
   - [Camera-PrIMuS](https://grfia.dlsi.ua.es/primus/) — place in `data/camera-primus/`
   - [GrandStaff](https://github.com/multiscore/GrandStaff) — place in `data/grandstaff/`
   - [OpenScore Lieder](https://github.com/OpenScore/Lieder) — place in `data/openscore-lieder/`

2. Generate synthetic data:
   ```bash
   python src/data/generate_synthetic.py --output data/synthetic/ --num-pages 20000
   ```

3. Build dataset index:
   ```bash
   python src/data/index.py --data-root data/ --output data/manifest.jsonl
   ```

## Training

### Stage 1 — Monophonic

```bash
python src/train/train.py --config configs/train_stage1.yaml
```

### Stage 2 — Polyphonic

```bash
python src/train/train.py --config configs/train_stage2.yaml --resume checkpoints/stage1_best.pt
```

### Stage 3 — Full Complexity

```bash
python src/train/train.py --config configs/train_stage3.yaml --resume checkpoints/stage2_best.pt
```

### YOLO (Stage A)

```bash
python src/train/train_yolo_stage_a.py
```

See `docs/TRAINING_COMMANDS.md` for detailed training commands and options.

## Evaluation

```bash
# Evaluate a Stage B checkpoint
python src/eval/evaluate_stage_b_checkpoint.py --checkpoint checkpoints/stage3_best.pt

# Compare MusicXML outputs using mir_eval metrics
python src/eval/compare_musicxml.py ground_truth.musicxml predicted.musicxml
```

## Architecture Design Rationale

The full architecture decision record, including evidence-based justification for every design choice. 

Key decisions:

- **DaViT + custom decoder over Florence-2 fine-tuning:** General-purpose VLMs fail entirely at OMR transcription (Calvo-Zaragoza et al., WORMS 2024). Florence-2's 51K NL tokenizer conflicts with our 487-token music vocabulary. The SMT/SMIReT architecture (5.92% SER) validates the encoder + task-specific decoder pattern.
- **Custom 487-token vocabulary over sub-word tokenization:** Music-aware encoding achieves 16.4% CER vs 62.3% character-level and 39.7% learned sub-word (Alfaro-Contreras et al., WORMS 2023).
- **Grammar FSA over statistical language models:** Statistical LMs cannot enforce structural constraints like beat consistency (Torras et al., WORMS 2022).
- **RoPE over learned positional embeddings:** Smooth extrapolation beyond training sequence length for unusually dense staves.
- **DoRA rank-64 over standard LoRA:** Weight-decomposed adaptation outperforms standard LoRA on fine-tuning benchmarks.

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
