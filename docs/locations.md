# Locations Reference

Canonical paths for assets that aren't in this repo's working tree, so future sessions don't have to re-discover them.

Last verified: 2026-05-10. Update when a path moves; this doc is the source of truth so other docs and memories can link here instead of duplicating paths.

## Repos

| Box | Path | Notes |
|---|---|---|
| Local (laptop / workstation) | `/home/ari/work/Clarity-OMR-Train-RADIO` | Linux, Python 3.14. CPU torch + ultralytics + pymupdf + mir_eval installed user-level via `pip install --user --break-system-packages` so unit tests in `tests/inference`, `tests/pipeline`, `tests/cli`, `eval/tests` run end-to-end. GPU/CUDA workloads still run on GPU box only. |
| GPU box (`seder` / `10.10.1.29`) | `C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO` | Windows, cmd.exe; ssh with `cd /d "<path>"` for spaces. Use `venv-cu132\Scripts\python.exe` (torch 2.13 dev cu132 + ultralytics + pymupdf) for inference and scoring; default system Python lacks torch. WSL Ubuntu installed but stopped. |
| Origin | `https://github.com/weselyj/Clarity-OMR-Train-RADIO.git` | `main` is the integration branch |

## Stage A (system YOLO) weights

| Path | Where | Size / Date |
|---|---|---|
| `runs/detect/runs/yolo26m_systems/weights/best.pt` | GPU box only | 44 MB, 2026-05-04 |
| `runs/detect/runs/yolo26m_systems/weights/last.pt` | GPU box only | sibling final-epoch weights |

YOLO config + training metadata live alongside the weights dir. Local box has no `runs/` directory; pull from GPU box if needed.

## Stage B (Stage 3 v2) checkpoints

GPU box: `checkpoints/full_radio_stage3_v2/`

| File | Notes |
|---|---|
| `stage3-radio-systems-frozen-encoder_best.pt` | val-loss 0.148 @ step 4000 — production checkpoint for Subproject 4 inference |
| `stage3-radio-systems-frozen-encoder_final.pt` | last step |
| `stage3-radio-systems-frozen-encoder_step_NNNNNNN.pt` | every 500 steps, useful for ablations |

Other Stage B variants exist on the GPU box at `checkpoints/`:
`baseline_davit`, `full_radio_stage1`, `full_radio_stage1_v2`, `full_radio_stage2`, `full_radio_stage2_systems`, `full_radio_stage2_systems_v2` (v2 is the polyphonic-mix anchor), `full_radio_stage3`, `full_radio_stage3_v1`, `full_radio_stage3_v2`, `full_radio_stage3_yolo_v1`, `mvp_radio_stage2`, `smoke_davit_stage1`, `v1_phase1_safe`.

## Lieder corpus (eval data)

GPU box only, at `data/openscore_lieder/`:

| Subdir | Contents |
|---|---|
| `scores/` | 1,462 `.mxl` (compressed MusicXML) ground-truth files, nested `<Composer>/<Opus>/<Song>/<id>.mxl` |
| `eval_pdfs/` | 145 PDFs — deterministic 10% eval split (seed 20260425) |

Source: cloned from `https://github.com/OpenScore/Lieder` at commit `6b2dc542c…`, downloaded 2026-04-25.

The 50-piece subset used for Phase 2 / Subproject 4 evals is a deterministic sub-slice of `eval_pdfs/` (driver picks the first 50 by ID; the eval driver enforces the slice).

**Smoke piece**: `lc6623145`
- PDF: `data/openscore_lieder/eval_pdfs/lc6623145.pdf`
- Ground truth: `data/openscore_lieder/scores/Barnard,_Charlotte_Alington/_/Five_o'clock_in_the_morning/lc6623145.mxl`
- Per-system smoke output (Subproject 4): `smoke_lc6623145.musicxml` + `.musicxml.diagnostics.json` at the GPU box repo root; staged copy at `smoke_predictions/lc6623145.musicxml` for the scorer. Smoke score CSV: `smoke_scores.csv`. Last result 2026-05-10: `onset_f1=0.0946`, `linearized_ser=0.5526` (well above the per-staff `0.067` baseline).
- Predecessor (per-staff, wrong format) output: `eval/results/lieder_stage3_v2_smoke/lc6623145.musicxml` — superseded.

## Subproject 4 (per-system inference) artifacts

GPU box, `C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\` relative:

| Path | Notes |
|---|---|
| `eval/results/lieder_subproject4/<piece_id>.musicxml` (+ `.musicxml.diagnostics.json`) | 50-piece corpus predictions written by `python -m eval.run_lieder_eval --name subproject4` |
| `eval/results/lieder_subproject4_inference_status.jsonl` | Phase 1 per-piece status (one JSON record per piece, flushed on each completion) |
| `eval/results/lieder_subproject4_scores.csv` | Phase 2 per-piece scores (cheap metrics: onset_f1, linearized_ser; TEDN off by default) |
| `subproject4_run.log` | Combined stdout/stderr from the corpus run; useful for post-mortem on per-piece failures |

The per-system inference entrypoints in this repo (Subproject 4):
- One-off: `python -m src.cli.run_system_inference --pdf … --out … --yolo-weights … --stage-b-ckpt …`
- Corpus: `python -m eval.run_lieder_eval --checkpoint … --config … --name … --max-pieces 50 --run-scoring`

## Encoder cache (Stage 3 training)

GPU box: encoder cache at hash `ac8948ae4b5be3e9`, 1.38 TB, 215,985 samples (per memory `project_radio_stage3_design.md`). Path TBD when next confirmed; not directly relevant for inference.

## Synthetic data

GPU box: `data/synthetic_v2/` (rendered grand-staff pages + per-system labels). Not needed for inference; relevant for retraining / re-rendering only.

## Token manifest

`src/data/manifests/token_manifest_stage3.jsonl` (in repo) — first line is canonical example of expected per-system manifest format. `staves_in_system`, `staff_indices` show multi-staff stacked layout.

## Decision-gate

`eval/decision_gate.py` (in repo) — gates 3 static system floors (`synthetic_systems` ≥ 90, `grandstaff_systems` ≥ 95, `primus_systems` ≥ 80) and 2 dynamic regression tripwires (`cameraprimus_systems`, `cameraprimus`, `primus`).

`grandstaff` (single-staff split) is intentionally NOT gated — invalid eval data per the product rule (single-staff inputs must come from naturally single-staff sources).

## Auto-doc sources to keep current

- This file (`docs/locations.md`) — repo-internal canonical paths
- `/mnt/nas/wiki/index.md` — homelab-wide wiki, separate concern
- `/home/ari/.claude/projects/-home-ari/memory/MEMORY.md` — Claude per-session memory index

When a checkpoint moves, a corpus is re-downloaded, or a weights file is regenerated, update this file in the same commit. Other docs and memories should link here rather than copy paths.
