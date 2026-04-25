# Evaluation harness

Vendored from the score-ingest Phase 0 cross-check (validated bit-equivalent to
upstream Clarity-OMR's eval.py -- gist 9992cca340).

## Files

- `playback.py` -- primary metric, mir_eval onset-only F1 with stripTies canonicalization
- `canonicalize.py` -- MusicXML normalization (parts -> P1, P2, ...; stripTies; etc.)
- `upstream_eval.py` -- author's eval.py vendored verbatim, used as the cross-check reference
- `lieder_split.py` -- defines the OpenScore Lieder held-out 10% split (Task 9)
- `run_baseline_reproduction.py` -- runs author's DaViT checkpoint through our pipeline; must match published numbers within +/-0.05 (Task 10)
- `run_lieder_eval.py` -- full Lieder eval, both encoders (Task 11)
- `run_4piece_eval.py` -- public 4-piece comparison table (Task 11)

## To re-run

```bash
python -m eval.run_baseline_reproduction
python -m eval.run_lieder_eval --checkpoint <path>
python -m eval.run_4piece_eval --checkpoint <path>
```

Results land in `eval/results/`.
