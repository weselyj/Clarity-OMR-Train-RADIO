# Results

Per-run eval outputs are not committed to this repository (they are
gitignored under `eval/results/`). Canonical published results — model
weights, scoring artifacts, and the lieder eval CSV bundle — will be
released to Hugging Face once Stage 3 inference reaches the ship gate.

**Hugging Face:** _Release pending. Link will be published here when
artifacts are uploaded._

To regenerate results locally on a CUDA-capable GPU box, see
[EVALUATION.md](EVALUATION.md).

## Reference baselines

Historical baselines (DaViT pre-rebuild, MVP smokes, pre-Subproject 4 lieder
runs) are preserved in [`archive/results/`](../archive/results/) for
reproducibility audits but should not be used as targets for current work.
