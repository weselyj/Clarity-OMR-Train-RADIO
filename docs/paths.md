# Repository-Relative Paths

Canonical paths for build artifacts, training outputs, and corpus data
that aren't committed to the repo. Paths are relative to the repo root
unless noted.

## Stage A (system YOLO) weights

| Path | Source |
|---|---|
| `runs/detect/runs/yolo26m_systems/weights/best.pt` | Produced by `scripts/train_yolo.py` (see [TRAINING.md](TRAINING.md)) |
| `runs/detect/runs/yolo26m_systems/weights/last.pt` | Sibling final-epoch weights |

This is the default `--stage-a-weights` for `eval.run_lieder_eval`. The
`runs/` tree is gitignored.

## Stage B (RADIO) checkpoints

| Path | Notes |
|---|---|
| `checkpoints/full_radio_stage3_v2/_best.pt` | Production Stage 3 v2 checkpoint (val_loss 0.148 at step 4000) |
| `checkpoints/full_radio_stage3_v2/_final.pt` | Last step of the run |
| `checkpoints/full_radio_stage3_v2/_step_<N>.pt` | Every 500 steps for ablations |

The `checkpoints/` tree is gitignored. Other Stage B variant directories
(`full_radio_stage1_v2`, `full_radio_stage2_systems_v2`, `baseline_davit`,
…) follow the same pattern.

## Lieder corpus (eval data)

| Path | Contents |
|---|---|
| `data/openscore_lieder/scores/` | `.mxl` ground-truth files, nested `<Composer>/<Opus>/<Song>/<id>.mxl` |
| `data/openscore_lieder/eval_pdfs/` | PDF render of the deterministic 10% eval split |

Source: [OpenScore Lieder](https://github.com/OpenScore/Lieder). The 50-piece
subset used for Subproject 4 evals is a deterministic sub-slice of `eval_pdfs/`
(the eval driver enforces the slice).

## Synthetic data

| Path | Notes |
|---|---|
| `data/synthetic_v2/` | Verovio-rendered grand-staff pages + per-system labels |

Built by `src/data/generate_synthetic.py`.

## Encoder cache (Stage 3 training)

| Path | Notes |
|---|---|
| `data/encoder_cache/` | Pre-encoded RADIO features keyed by sample hash |

Built by `scripts/build_encoder_cache.py`.

## Eval outputs (per-run, gitignored)

| Path | Producer |
|---|---|
| `eval/results/lieder_<name>/<piece_id>.musicxml` | `eval.run_lieder_eval` |
| `eval/results/lieder_<name>/<piece_id>.musicxml.diagnostics.json` | `eval.run_lieder_eval` |
| `eval/results/lieder_<name>_inference_status.jsonl` | `eval.run_lieder_eval` |
| `eval/results/lieder_<name>_scores.csv` | `eval.score_lieder_eval` (run via `--run-scoring`) |

See [`eval/results/README.md`](../eval/results/README.md) and
[`docs/RESULTS.md`](RESULTS.md).

## Token manifest reference

`src/data/manifests/token_manifest_stage3_audit.json` — committed audit
summary of the Stage 3 combined token manifest. Records total entry count,
per-dataset breakdown (synthetic_systems / grandstaff_systems /
primus_systems / cameraprimus_systems), per-split counts, and source
manifest paths. The full per-system `.jsonl` manifests it summarizes are
gitignored build artifacts.

## Decision-gate config

`eval/decision_gate.py` — gates 3 static system floors
(`synthetic_systems` ≥ 90, `grandstaff_systems` ≥ 95, `primus_systems` ≥ 80)
and 2 dynamic regression tripwires (`cameraprimus_systems`, `cameraprimus`,
`primus`).

`grandstaff` (single-staff split) is intentionally NOT gated — it's invalid
eval data per the product rule (single-staff inputs must come from naturally
single-staff sources).
