# Evaluation

This page covers running the lieder corpus evaluation end-to-end,
the metrics it reports, and the decision gates the project uses to
interpret the results.

## Prerequisites

- A CUDA-capable GPU (see [HARDWARE.md](HARDWARE.md))
- Activated cu132 venv (see [INSTALL.md](INSTALL.md))
- Stage A YOLO weights at `runs/detect/runs/yolo26m_systems/weights/best.pt`
  (or pass `--stage-a-weights`)
- A trained Stage B checkpoint (e.g. `checkpoints/full_radio_stage3_v2/_best.pt`)
- Lieder corpus data at `data/openscore_lieder/{scores,eval_pdfs}/`
  (see [paths.md](paths.md))

## Two-pass design

Lieder eval runs in two passes to keep music21/zss memory bounded:

1. **Inference pass.** `eval.run_lieder_eval` runs the RADIO/YOLO pipeline and
   writes `<piece_id>.musicxml` + `<piece_id>.musicxml.diagnostics.json` per
   piece. No metrics computed.
2. **Scoring pass.** `eval.score_lieder_eval` runs metric computation in a
   subprocess per piece (so music21 state is fully reclaimed between pieces)
   and writes `lieder_<name>_scores.csv`.

The `--run-scoring` flag on `run_lieder_eval` invokes the scorer subprocess
automatically after the inference pass completes.

## Running the corpus eval

```bash
python -m eval.run_lieder_eval \
    --checkpoint checkpoints/full_radio_stage3_v2/_best.pt \
    --config configs/train_stage2_radio_polyphonic.yaml \
    --name stage3v2_corpus \
    --max-pieces 50 \
    --run-scoring
```

Outputs (gitignored under `eval/results/`):

- `eval/results/lieder_stage3v2_corpus/<piece_id>.musicxml`
- `eval/results/lieder_stage3v2_corpus/<piece_id>.musicxml.diagnostics.json`
- `eval/results/lieder_stage3v2_corpus_inference_status.jsonl`
- `eval/results/lieder_stage3v2_corpus_scores.csv`

`--max-pieces 50` runs the deterministic 50-piece subset used for the
Subproject 4 corpus gate. Drop the flag for the full 145-piece eval split.

## Smoke run (single piece)

```bash
python -m src.cli.run_system_inference \
    --pdf data/openscore_lieder/eval_pdfs/lc6623145.pdf \
    --out smoke_lc6623145.musicxml \
    --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt \
    --stage-b-ckpt checkpoints/full_radio_stage3_v2/_best.pt
```

## Stratified scoring

`lieder_<name>_scores.csv` includes per-piece `onset_f1` and `linearized_ser`.
The `stage_d_skipped_systems` column flags pieces where one or more system
crops were dropped during MusicXML export — useful for triaging low scores.

The `tedn` column is omitted by default; pass `--tedn` to
`eval.score_lieder_eval` to enable (slow, memory-heavy).

## Metrics

- **Symbol Error Rate (SER):** Edit distance between predicted and ground-truth token sequences, normalized by ground-truth length.
- **Onset F1 (mir_eval):** Note-level precision/recall/F1 (onset_tolerance=50ms, pitch_tolerance=50 cents).
- **Pitch / rhythm accuracy:** Per-note correctness rates.
- **Key/time signature accuracy:** Exact-match.
- **Structural F1:** Barlines, measure boundaries, voice assignments.
- **Stratified by `staves_in_system`** (1, 2, 3, 4+) — checks system-level model holds up at increasing staff counts.

## Decision gates

| Outcome | Mean lieder onset_f1 | Action |
|---|---|---|
| Strong | ≥ 0.30 | Ship: PR, write-up, follow-ups for full-quality decode + HF release prep |
| Mixed | 0.241 ≤ x < 0.30 | Beats DaViT baseline but not transformative; investigate residual error before next major iteration |
| Flat / regressed | < 0.241 | System-level approach also failed; pivot to classical pipeline (Audiveris-style) |

## Canonical published results

See [RESULTS.md](RESULTS.md). Per-run CSVs from local boxes stay local
(gitignored).
