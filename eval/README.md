# Evaluation harness

Vendored from the score-ingest Phase 0 cross-check (validated bit-equivalent to
upstream Clarity-OMR's eval.py -- gist 9992cca340).

## Two-pass design (both demo and lieder evals)

All full evals use a **two-pass** design to prevent music21/zss state from
accumulating across pieces:

1. **Inference pass** — runs the RADIO/YOLO pipeline per piece, writes predicted
   MusicXML + Stage-D diagnostics sidecars to disk, then exits cleanly.
2. **Scoring pass** — subprocess-isolates per-piece metric computation (cheap
   metrics @60 s, tedn @300 s) so the OS reclaims all memory after each child
   exits.  The parent process stays small regardless of corpus size.

This design was adopted after Phase A profiling confirmed that in-process scoring
causes ~11 GB/min committed-memory growth.  A 20-piece lieder run hit 43 GB at
piece 6 and was killed before pagefile exhaustion (the demo eval OOM at 39 GB
was the same root cause — see PR #26).

### Shared infrastructure

`eval/_scoring_utils.py` contains the subprocess dispatch logic used by both
`score_demo_eval.py` and `score_lieder_eval.py`:

- `score_piece_subprocess()` — splits cheap vs tedn into separate child
  processes with independent timeouts
- `_run_subprocess()` — invokes `eval._score_one_piece`, parses JSON result
- `_read_stage_d_diag()` — reads the `.diagnostics.json` sidecar (in-process,
  no music21)
- `CSV_HEADER` — canonical column list shared by both CSV outputs

## Files

- `playback.py` — primary metric, mir_eval onset-only F1 with stripTies canonicalization
- `canonicalize.py` — MusicXML normalization (parts -> P1, P2, ...; stripTies; etc.)
- `upstream_eval.py` — author's eval.py vendored verbatim, used as the cross-check reference
- `lieder_split.py` — defines the OpenScore Lieder held-out 10% split (Task 9)
- `_scoring_utils.py` — shared subprocess-isolated scoring infrastructure (PR E)
- `_score_one_piece.py` — subprocess worker: loads music21/zss, scores one piece, prints JSON
- `run_baseline_reproduction.py` — runs author's DaViT checkpoint through our pipeline; must match published numbers within +/-0.05 (Task 10)
- `run_clarity_demo_eval.py` — **inference pass** for the 4-piece public comparison table (PR #26)
- `score_demo_eval.py` — **scoring pass** for the 4-piece results; subprocess-isolated per piece
- `run_lieder_eval.py` — **inference pass** for the Lieder eval split (PR E)
- `score_lieder_eval.py` — **scoring pass** for the Lieder results; subprocess-isolated per piece (PR E)

## Running the evals

For running the lieder corpus evaluation (and the demo / baseline evals),
see [`docs/EVALUATION.md`](../docs/EVALUATION.md).

For repo-relative paths to checkpoints and corpus data, see
[`docs/paths.md`](../docs/paths.md).
