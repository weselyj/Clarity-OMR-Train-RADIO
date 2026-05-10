# Subproject 4 — Tasks 1-11 Shipped, Tasks 12-16 Open (2026-05-10)

> Pick up by resuming Task 12 of [`docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`](../plans/2026-05-10-radio-subproject4-implementation.md). Branch is on origin and ready to be checked out on the GPU box once Tasks 12-14 land.

## State at end of session

| Box | Branch | HEAD | Pushed? |
|---|---|---|---|
| Local (`/home/ari/work/Clarity-OMR-Train-RADIO`) | `feat/subproject4-system-inference` | `ffe1744` | yes |
| GPU box (`seder` / 10.10.1.29) | `main` | `486d377` | n/a — needs `git fetch && git checkout feat/subproject4-system-inference` after Task 14 |
| Origin | `feat/subproject4-system-inference` | `ffe1744` | yes |

Working tree clean.

## What landed this session

Twelve commits, in order:

| SHA | Task | Summary |
|---|---|---|
| `1503236` | 1 | `StageBInferenceBundle` dataclass scaffold (TYPE_CHECKING-only torch import) |
| `f02dc45` | 2 | `load_stage_b_for_inference()` — 8-step Stage B inference loader |
| `e8178f9` | 3 | `YoloStageASystems` inference wrapper (lazy YOLO import to dodge torchvision NMS crash on CPU) |
| `c7ff295` | 4 | `assemble_score_from_system_predictions()` — per-system token → AssembledScore adapter |
| `9f64f3c` | 5 | `SystemInferencePipeline.__init__` scaffold |
| `f79f5d6` | 6 | `run_system_crop` |
| `2a39466` | 7 | `run_page` |
| `eef2680` | — | fix: preserve caller's `system_index` in `run_system_crop` flatten (bug exposed by Task 7's two-system test; see "Known fixes" below) |
| `b4fa1e3` | 8 | `run_pdf` via PyMuPDF |
| `e0a47fa` | 9 | `export_musicxml` + `.diagnostics.json` sidecar |
| `a79e832` | 10 | thin CLI `python -m src.cli.run_system_inference` |
| `ffe1744` | 11 | `git mv` of `run_lieder_eval.py` + `score_lieder_eval.py` (and their tests) from `archive/per_staff/` → `eval/` |

REVIEW CHECKPOINT 1 confirmed locally: 16 of 16 new-functionality tests green:

```
tests/inference                                   (5 tests, 5 pass)
tests/models/test_yolo_stage_a_systems.py         (3 tests, 3 pass)
tests/pipeline/test_assemble_from_system_predictions.py (4 tests, 4 pass)
tests/cli/test_run_system_inference.py            (2 tests, 2 pass)
tests/inference/test_system_pipeline_pdf.py       (2 tests, 2 pass)
```

Pre-existing failures unrelated to this work and acceptable per the plan's "Notes for the executing agent": `tests/data/test_multi_dpi.py` (ImageMagick path), `tests/data/test_encoder_cache.py` (CPU-only torch perf paths).

## Local environment changes

Installed three Python packages (CPU-only) into the user-level pip namespace via `--user --break-system-packages` so the local TDD test loop could run end-to-end:

- `torch==2.11.0+cpu` (from the PyTorch CPU index)
- `ultralytics==8.4.48`
- `pymupdf==1.27.2.3`

These are not committed to `requirements.txt`; the GPU box's existing CUDA env is unaffected. A future "set up local dev env" doc could capture this.

## Spec deviations worth knowing

1. **Lazy YOLO import in `src/models/yolo_stage_a_systems.py`.** The plan wrote `from ultralytics import YOLO` at module top, but that import crashes on CPU-only torch installs (`RuntimeError: operator torchvision::nms does not exist`). The committed implementation defers the import to first instantiation and exposes the symbol as a module-level `YOLO = None` so the spec's `patch("src.models.yolo_stage_a_systems.YOLO", ...)` test pattern still works. Functionally identical on the GPU box.

2. **Test #3 in `tests/pipeline/test_assemble_from_system_predictions.py`.** The plan asserted `staff.system_index_hint == 7` against `AssembledStaff` instances. `AssembledStaff` (`src/pipeline/assemble_score.py:34`) does not carry `system_index_hint` — only the input `StaffRecognitionResult` does. The committed test instead verifies what the assembler actually preserves: the hint flows into `group_staves_into_systems` for grouping, and the output system lands on the correct `page_index`. This matches the spec's intent ("Build a `StaffRecognitionResult` per staff with `system_index_hint = system_index`...") without making false claims about output preservation.

3. **`run_system_crop` system-index bug fix (`eef2680`).** Task 6's plan code wrote `system_index_hint=system.system_index` in the flatten loop. `assemble_score()` reassigns `system_index` via `enumerate()` (so a single-system input always gets index 0), which masked the bug in Task 6's single-system unit test. Task 7's two-system `run_page` test surfaced it. The fix: use the caller's `system_index` parameter directly. Three lines, one commit, clearly described in the commit message.

## What's left

### Task 12 — Rewire `run_inference()` to in-process pipeline (NOT DONE)

This is the next session's first task. The lifted `eval/run_lieder_eval.py` is a 709-line file with a substantially more complex `main()` than the plan's pseudocode (parallel `--jobs`, device queue, ETA tracking, cache validity checks, ThreadPoolExecutor). The plan's "Notes for the executing agent" explicitly calls for a *structural* integration rather than a verbatim replacement.

**Required surgery**:
- Replace the `run_inference` body (`eval/run_lieder_eval.py:139-207`) with the new in-process signature: `run_inference(*, pipeline, pdf, out, work_dir)`.
- In `main()`: instantiate one `SystemInferencePipeline` from CLI args (mapping `--checkpoint` → `stage_b_ckpt`, `--stage-a-weights` → `yolo_weights`, `--stage-b-device` → `device`, etc.), then pass it through `_run_piece` → `run_inference`. The parallel-jobs / device-pool logic doesn't apply to a single in-process pipeline; the plan's pseudocode drops both — match that.
- The lifted `eval/tests/test_run_lieder_eval.py` is currently broken: it bootstraps via `importlib.util` with `_REPO_ROOT = Path(__file__).resolve().parents[1]`, which now resolves to `eval/` instead of the repo root. After the rewire, replace the boilerplate with `import eval.run_lieder_eval as rle` and rewrite the relevant tests against the new signature. Plan calls out which tests to keep (resume from status, ETA formatting, status JSONL writes, format_progress) and which to delete (anything testing the old subprocess `cmd` list).

### Task 13 — Add `--run-scoring` + `--tedn` to `run_lieder_eval.py` (NOT DONE)

After Task 12. Adds an `invoke_scoring_phase()` helper that spawns `eval.score_lieder_eval` as a subprocess (NEVER inline). Test asserts `subprocess.run` is the call site, with `eval.score_lieder_eval` and `--tedn` in the command. Spec at `docs/superpowers/specs/2026-05-10-radio-subproject4-design.md:185-198`.

### Task 14 — Add `--tedn` flag to `score_lieder_eval.py` (NOT DONE)

Default off. Test asserts `args.tedn is False` when not passed. Spec at `docs/superpowers/specs/2026-05-10-radio-subproject4-design.md:200-204`.

### Tasks 15-16 — Smoke + 50-piece eval on the GPU box (NOT DONE)

Per the plan, these are run-only (no code changes) but require the GPU box. Sync the branch first:

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && git fetch origin && git checkout feat/subproject4-system-inference && git pull --ff-only"
```

Smoke command (Task 15):

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && python -m src.cli.run_system_inference --pdf data\\openscore_lieder\\eval_pdfs\\lc6623145.pdf --out smoke_lc6623145.musicxml --yolo-weights runs\\detect\\runs\\yolo26m_systems\\weights\\best.pt --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt --device cuda"
```

The CLI module lands in this branch (`a79e832`); it is callable as soon as the GPU box has the branch checked out, *independent* of Tasks 12-14 (those only affect the eval driver, not the CLI).

Per the spec ship-gate criterion 2: smoke `onset_f1` should be **well above** the per-staff `0.067` baseline. Near-baseline = stop-and-investigate.

Task 16 spec at the plan's "Task 16" section. Wall time estimate: 2-3 h Phase 1 + 30-50 min Phase 2.

## Operational notes for next session

- **Subagent scope creep was the dominant friction this session.** Three separate sonnet `general-purpose` subagents went past their assigned task — Task 4 implementer also wrote and committed Task 5 (`9f64f3c`) and Task 6 (`f79f5d6`); they reported "already done in a prior session" in confused phrasing. The work was correct enough to keep, but the audit trail is partly out-of-order (Task 5 committed *before* Task 4). Future sessions: dispatch with very explicit "implement EXACTLY task N — do NOT touch any code outside the listed files" framing, or just do the task inline if the change is mechanical and small. Tasks 7-11 in this session were done inline for that reason.
- **LM Studio was down for the entire session**; `delegate_to_local_agent` was unavailable. All subagents were sonnet `general-purpose` per the user's explicit instruction at session start.
- **`docs/superpowers/specs/2026-05-10-radio-subproject4-design.md` and `docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md` are unchanged on this branch.** Read them again at the start of the next session — they remain the source of truth for the remaining tasks.

## References

- Spec: [`docs/superpowers/specs/2026-05-10-radio-subproject4-design.md`](../specs/2026-05-10-radio-subproject4-design.md)
- Plan: [`docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`](../plans/2026-05-10-radio-subproject4-implementation.md)
- Predecessor handoff: [`2026-05-10-subproject4-plan-ready.md`](2026-05-10-subproject4-plan-ready.md)
- Locations / paths: [`docs/locations.md`](../../locations.md)
