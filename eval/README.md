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
- `run_clarity_demo_eval.py` -- inference pass for the 4-piece public comparison table (PR #26)
- `score_demo_eval.py` -- scoring pass for the 4-piece results; subprocess-isolated per piece

## To re-run

```bash
python -m eval.run_baseline_reproduction
python -m eval.run_lieder_eval --checkpoint <path>
# 4-piece demo eval uses a two-pass design (inference then scoring):
python -m eval.run_clarity_demo_eval --checkpoint <path> --out-dir eval/results/demo
python -m eval.score_demo_eval --results-dir eval/results/demo
```

Results land in `eval/results/`.
