# Subproject 4 — Shipped (2026-05-10)

> Per-system end-to-end inference pipeline (PDF → MusicXML) plus the 50-piece lieder corpus eval. All 16 plan tasks complete; ship-gate **PASS**.

## State at end of session

| Box | Branch | HEAD | Pushed? |
|---|---|---|---|
| Local (`/home/ari/work/Clarity-OMR-Train-RADIO`) | `feat/subproject4-system-inference` | `7793351` (+ run-artifacts commit) | yes |
| GPU box (`seder` / `10.10.1.29`) | `feat/subproject4-system-inference` | `7793351` (artifacts produced) | n/a |
| Origin | `feat/subproject4-system-inference` | `7793351` | yes |

Working tree will be clean once the run-artifacts commit lands.

## Smoke result (`lc6623145`)

- **`onset_f1 = 0.0946`** — well above the per-staff `0.067` baseline (≈41% improvement)
- `linearized_ser = 0.5526`
- 1 of N systems on the piece had truncated decoder tokens (no `<staff_end>`); `assemble_score_from_system_predictions` skipped just that system rather than failing the piece. The Task-12 + post-smoke fix (`5ce3fbb`) catches `ValueError` from the splitter and continues.
- Predicted MusicXML: 295 KB, 1274 notes, 211 measures, 5 parts. Diagnostics sidecar clean (0 skipped notes, 0 unknown tokens).

## 50-piece corpus result

- **Phase 1 status: 50/50 `done`, 0 failures.** Wall time **82 min** (mean 99 s/piece, median 81 s, max 348 s) — well under the spec's 2-3 h estimate.
- **Phase 2 scoring: 50/50 scored, 0 failures.** No missing references, no scoring failures.
- **`mean(onset_f1) = 0.0819`** (median 0.0782, range 0.0000–0.2504)
- **`mean(linearized_ser) = 0.6557`** (median 0.6511)
- **33/50 pieces above the 0.067 baseline** (66%).
- TEDN was off by default per spec (cheap-metrics-only run).
- Per-piece distribution: see [`eval/results/lieder_subproject4_scores.csv`](../../../eval/results/lieder_subproject4_scores.csv) and [`eval/results/lieder_subproject4_inference_status.jsonl`](../../../eval/results/lieder_subproject4_inference_status.jsonl).

## Ship-gate verdict

| # | Criterion | Result |
|---|---|---|
| 1 | Smoke produces parseable MusicXML + numeric F1 | ✅ |
| 2 | Smoke F1 well above 0.067 baseline | ✅ (0.0946 vs 0.067) |
| 3 | Full Phase 1 + Phase 2 run completes — every piece has a row | ✅ (50/50 inference, 50/50 scored) |
| 4 | ≥80% Phase 1 ok AND ≥80% Phase 2 scored | ✅ (100% / 100%) |
| 5 | Aggregate `mean(onset_f1)` reported | ✅ (0.0819) |

**Verdict: SHIP.** All five criteria pass. Per the spec, "the **architectural** ship-gate is 'the eval ran cleanly and produced a real number,' not 'the number is good.'" The aggregate (0.0819) clears the per-staff baseline (0.067) but is modest; that's a finding for the next phase, not a Subproject 4 blocker.

## Notes for next phase

- **Onset_f1 floor on a subset.** 17 of 50 pieces score below the 0.067 baseline. Likely failure modes to investigate (in order of suspected prevalence):
  1. Decoder truncation mid-staff under `--max-decode-steps 2048` for very dense systems — visible in the smoke as one truncated system. Increasing the budget or switching to length-aware decoding may recover signal.
  2. Stage A under-detecting systems on multi-system pages with tight margins (cameraprimus-style scans look the worst-affected per `project_radio_stage1_v2`).
  3. Even-split y-coords are crude; multi-staff systems where the brace doesn't divide the page evenly may misalign assemble_score's per-staff regions.
- **Linearized SER (0.6557 median) is more useful than onset_f1 right now.** The token-level metric is cleaner because it doesn't depend on time-alignment of dense piano content. Worth reporting alongside onset_f1 in future shipping decisions.
- **Decoder timeout protection deferred.** The plan flagged "subprocess-per-piece inference fallback flag (only if observed OOMs warrant it)" — this run hit no OOMs and no piece took longer than ~6 minutes (max 347 s). Keep it deferred unless a future run breaks the trend.

## What landed this session

Commits on top of the prior tasks-1-11-shipped handoff (`27e17b9`):

| SHA | Task / Purpose |
|---|---|
| `55c6d28` | Task 12 — rewire `run_inference` to in-process `SystemInferencePipeline` |
| `ca4057c` | Task 13 — `--run-scoring` + `--tedn` flags + `invoke_scoring_phase` subprocess helper |
| `f35f674` | Task 14 — `--tedn` flag in `score_lieder_eval` (default off); `--ground-truth-dir` / `--out-csv` aliases |
| `2996826` | drop two `score_demo_eval` tests scoped out of Subproject 4; install `mir_eval` locally |
| `5ce3fbb` | fix: skip malformed system tokens (decoder truncation) in `assemble_score_from_system_predictions` instead of failing the piece |
| `d70fc73` | lift `eval/_score_one_piece.py` from archive — Task 11 missed it; required by the scorer subprocess |
| `7793351` | docs: locations.md updated with Subproject 4 artifact paths and the GPU-box `venv-cu132` requirement |
| (next) | Phase 1 status JSONL + Phase 2 scores CSV from the 50-piece run |

REVIEW CHECKPOINT 2 verified: 113 of 113 new-functionality tests green locally; 2 pre-existing failures unrelated to this work (`tests/data/test_multi_dpi.py` ImageMagick path, `tests/data/test_encoder_cache.py` torch perf paths) confirmed in the prior handoff.

## Spec deviations worth knowing

1. **Lazy YOLO import** in `src/models/yolo_stage_a_systems.py` (carried over from Tasks 1-11; necessary on CPU-only torch installs).
2. **`AssembledStaff` does not carry `system_index_hint`** to the output type; the input `StaffRecognitionResult` does. The Task-4 test was rewritten to match this rather than mutate `AssembledStaff`. Carried over from Tasks 1-11.
3. **Malformed system tokens are skipped silently** rather than recovered or counted. The eval driver's per-piece status row records "ok" as long as inference completed; the diagnostics sidecar does not currently surface skipped-system counts. Future improvement: thread skipped systems into `StageDExportDiagnostics` so the scorer can correlate "low onset_f1" with "many skipped systems."
4. **Score driver flag aliases** (Task 14): `--ground-truth-dir` aliases `--reference-dir`, `--out-csv` is a new override Path. The original `--name`-based dir convention still works; the new flags exist so `invoke_scoring_phase` (Task 13) can call the scorer with the canonical Subproject 4 contract.

## Operational notes

- **GPU box environment**: `venv-cu132/Scripts/python.exe` is the correct interpreter (torch 2.13 dev cu132 + ultralytics + pymupdf). Default system Python on the box lacks torch and crashes module imports. Documented in [`docs/locations.md`](../../locations.md).
- **PyMuPDF on first GPU-box run**: `pymupdf` was not previously installed in `venv-cu132` and had to be `pip install`ed before the smoke could run. Future rebuilds of the venv should pin all of `requirements.txt` including `pymupdf`.
- **`tee` is Unix-only**; on Windows use `> file 2>&1` for log capture. The first corpus-eval launch failed instantly with `'tee' is not recognized as an internal or external command, operable program or batch file.` and was relaunched with the Windows redirect.
- **LM Studio was down** for this session; `delegate_to_local_agent` was unavailable. All subagents were sonnet `general-purpose` per the user's session-start instruction.

## Open follow-ups (not blockers)

Per the spec's "Deferred follow-ups" section:

- TorchAO inference quantization experiment (`/home/ari/docs/clarity_omr_radio_torchao_evaluation.md`).
- Refactor `src/eval/evaluate_stage_b_checkpoint.py` to call the new `load_stage_b_for_inference` helper instead of inlining the 8-step sequence.
- Subprocess-per-piece inference fallback flag — keep deferred; this 50-piece run hit no OOMs.

New (post-shipping) follow-ups:

- Investigate the 17 below-baseline pieces; cross-reference with `cameraprimus`-style scans on the eval split.
- Consider exposing skipped-system counts in `StageDExportDiagnostics` so future runs can correlate decoder-truncation with score regressions.

## References

- Spec: [`docs/superpowers/specs/2026-05-10-radio-subproject4-design.md`](../specs/2026-05-10-radio-subproject4-design.md)
- Plan: [`docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`](../plans/2026-05-10-radio-subproject4-implementation.md)
- Predecessor handoff: [`2026-05-10-subproject4-tasks-1-11-shipped.md`](2026-05-10-subproject4-tasks-1-11-shipped.md)
- Locations / paths: [`docs/locations.md`](../../locations.md)
- Smoke artifacts (GPU box): `smoke_lc6623145.musicxml` + `.musicxml.diagnostics.json`, `smoke_scores.csv`
- Corpus artifacts: `eval/results/lieder_subproject4/<piece_id>.musicxml` (GPU box only — large), `eval/results/lieder_subproject4_inference_status.jsonl` (committed), `eval/results/lieder_subproject4_scores.csv` (committed)
