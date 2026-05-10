# Subproject 4 — Shipped (2026-05-10)

> Per-system end-to-end inference pipeline (PDF → MusicXML) plus the 50-piece lieder corpus eval. All 16 plan tasks complete; ship-gate **PASS**. Merged to `main` as PR #44.

## State at end of session

| Box | Branch | HEAD |
|---|---|---|
| Local (`/home/ari/work/Clarity-OMR-Train-RADIO`) | `main` | `3b6ab4e` (merge of PR #44) |
| GPU box (`seder` / `10.10.1.29`) | `main` | `3b6ab4e` (fast-forward synced) |
| Origin | `main` | `3b6ab4e` |

Working trees clean on both boxes. Feature branch `feat/subproject4-system-inference` deleted both locally and on origin.

PR: https://github.com/weselyj/Clarity-OMR-Train-RADIO/pull/44

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
| `d99d795` | data: subproject4 50-piece corpus run results (status JSONL + scores CSV) + this shipping handoff |
| `674d0a2` | review fixes: revert lazy YOLO import to spec form; add `skipped_systems` counter to `StageDExportDiagnostics` (see "Pre-merge review fixes" below) |
| `3b6ab4e` | merge of PR #44 to `main` |

REVIEW CHECKPOINT 2 verified: 113 of 113 new-functionality tests green locally; post-`674d0a2` the 5 `tests/inference/test_system_pipeline*.py` cases no longer collect on the local CPU-only torch install (expected — those run on the GPU box's `venv-cu132`, where 108/108 pass). 2 pre-existing failures unrelated to this work (`tests/data/test_multi_dpi.py` ImageMagick path, `tests/data/test_encoder_cache.py` torch perf paths) confirmed in the prior handoff.

## Pre-merge review fixes (commit `674d0a2`)

The user reviewed the PR and flagged two of the items called out under "Spec deviations worth knowing." Both addressed before merge:

1. **Reverted the lazy YOLO import.** The previous `yolo_stage_a_systems.py` carried a `YOLO = None` placeholder + `_get_yolo()` helper to dodge a `RuntimeError: operator torchvision::nms does not exist` on local CPU-only torch installs. The right answer was to stop installing CPU torch locally and run torch-dep tests on `seder`'s `venv-cu132` (CUDA 13 nightly) instead. Module-top `from ultralytics import YOLO` is now back, matching the spec. Saved as user feedback memory `feedback_use_gpu_box_for_torch_tests.md` so future sessions don't recreate the workaround.
2. **Added `skipped_systems` field to `StageDExportDiagnostics`.** `assemble_score_from_system_predictions` now accepts an optional `diagnostics` argument and increments the counter every time the splitter raises on a malformed (decoder-truncated) system. `SystemInferencePipeline.run_pdf` forwards the caller's diagnostics through, so the per-piece `.musicxml.diagnostics.json` sidecar now exposes `skipped_systems`. This unlocks correlating low `onset_f1` with decoder-truncation rate during post-mortems on the 17 below-baseline pieces. Two new tests (`test_diagnostics_skipped_systems_counter_tracks_each_drop`, `test_diagnostics_default_none_does_not_crash`) cover the counter and the `None` default.

After these fixes, on the GPU box: 108 passed / 1 skipped across `tests/pipeline tests/inference tests/cli eval/tests` (excluding the unrelated `eval/tests/test_playback.py` collection).

## Spec deviations worth knowing

1. ~~**Lazy YOLO import** in `src/models/yolo_stage_a_systems.py`~~ — **resolved in `674d0a2`**: reverted to spec-form module-top import. Local CPU-only torch installs are no longer a target; torch-dep tests run on the GPU box.
2. **`AssembledStaff` does not carry `system_index_hint`** to the output type; the input `StaffRecognitionResult` does. The Task-4 test was rewritten to match this rather than mutate `AssembledStaff`. Carried over from Tasks 1-11.
3. ~~**Malformed system tokens are skipped silently**~~ — **partially resolved in `674d0a2`**: `StageDExportDiagnostics.skipped_systems` is now incremented on each drop and surfaced in the `.musicxml.diagnostics.json` sidecar. The eval driver's per-piece status row still reports "ok" regardless (status JSONL contract unchanged); the count is in the sidecar, not the JSONL.
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

- Investigate the 17 below-baseline pieces; cross-reference with `cameraprimus`-style scans on the eval split. The new `skipped_systems` field in each piece's `.musicxml.diagnostics.json` sidecar (added in `674d0a2`) tells you which low scores are driven by decoder truncation vs genuine model weakness — re-run the corpus eval on `main` once and aggregate the sidecar counts to triage.
- Consider also exposing `skipped_systems` in the per-piece status JSONL so a Phase-1 post-mortem doesn't have to re-open every sidecar.

## References

- Spec: [`docs/superpowers/specs/2026-05-10-radio-subproject4-design.md`](../specs/2026-05-10-radio-subproject4-design.md)
- Plan: [`docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`](../plans/2026-05-10-radio-subproject4-implementation.md)
- Predecessor handoff: [`2026-05-10-subproject4-tasks-1-11-shipped.md`](2026-05-10-subproject4-tasks-1-11-shipped.md)
- Locations / paths: [`docs/locations.md`](../../locations.md)
- Smoke artifacts (GPU box): `smoke_lc6623145.musicxml` + `.musicxml.diagnostics.json`, `smoke_scores.csv`
- Corpus artifacts: `eval/results/lieder_subproject4/<piece_id>.musicxml` (GPU box only — large), `eval/results/lieder_subproject4_inference_status.jsonl` (committed), `eval/results/lieder_subproject4_scores.csv` (committed)
