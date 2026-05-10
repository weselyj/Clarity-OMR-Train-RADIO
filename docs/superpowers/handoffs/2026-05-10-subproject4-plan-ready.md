# Subproject 4 — Plan Ready, Awaiting Execution (2026-05-10)

> Pick this handoff up at the start of the next session. The work it covers is **all on `main`**; there is no in-flight branch yet.

## TL;DR

Three docs and one small refactor landed today. Subproject 4 is fully designed, externally reviewed, and turned into a step-by-step implementation plan. Next session's job is to execute the plan.

## State at end of session

| Box | Branch | HEAD |
|---|---|---|
| Local (`/home/ari/work/Clarity-OMR-Train-RADIO`) | `main` | `4e55231` |
| GPU box (`seder` / 10.10.1.29, `C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO`) | `main` | `4e55231` |
| Origin | `main` | `4e55231` |

Working tree clean on both boxes.

## What landed today

1. **`83ffcea` — refactor(data): split yolo_aligned_crops; lift shared utils to yolo_common**
   - Item #3 from the per-system-cleanup handoff: shared YOLO geometry/adapter helpers (`iou_xyxy`, `_yolo_predict_to_boxes`) extracted to `src/data/yolo_common.py`; per-staff-only functions moved to `archive/per_staff/src/data/yolo_aligned_crops.py`.
   - Item #4 (squash two single-staff-floor commits): skipped — already pushed, would need force-push, marginal value.

2. **`06aa4be` — docs: Subproject 4 design + overview + canonical locations reference**
   - [`docs/superpowers/specs/2026-05-10-radio-subproject4-design.md`](../specs/2026-05-10-radio-subproject4-design.md) — technical spec.
   - [`docs/superpowers/specs/2026-05-10-radio-subproject4-overview.md`](../specs/2026-05-10-radio-subproject4-overview.md) — plain-English companion.
   - [`docs/locations.md`](../../locations.md) — canonical paths reference (corpus, weights, GPU box paths).
   - The spec went through one external review round that caught 5 real issues. All five fixes are in the committed spec; see `## What was the first review round, and what did it change?` in the overview doc.

3. **`4e55231` — docs(plans): Subproject 4 implementation plan**
   - [`docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`](../plans/2026-05-10-radio-subproject4-implementation.md) — 17 tasks, 5 phases, 2 review checkpoints, ~2130 lines.
   - TDD-paced (test → fail → impl → pass → commit per step).

## Next session: pick this up by executing the plan

### Step 1 — read the plan + spec

The plan references the spec; the spec references the overview and the locations doc. Read in this order:

1. [`docs/superpowers/specs/2026-05-10-radio-subproject4-design.md`](../specs/2026-05-10-radio-subproject4-design.md) — what we're building and why
2. [`docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`](../plans/2026-05-10-radio-subproject4-implementation.md) — how, step by step
3. [`docs/locations.md`](../../locations.md) — where things are

### Step 2 — pick an execution mode

The plan offers two:

- **Subagent-driven** (recommended): one fresh subagent per task, review between tasks. Good for keeping main-context lean and reviewing diffs in isolation. Use the `superpowers:subagent-driven-development` skill.
- **Inline**: execute tasks in the same session using the `superpowers:executing-plans` skill. Batch with checkpoints.

### Step 3 — branch off main and start with Task 0

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git fetch origin
git checkout main && git pull --ff-only origin main   # should already be at 4e55231
git checkout -b feat/subproject4-system-inference
```

Then work through Task 1 (`StageBInferenceBundle`), Task 2 (`load_stage_b_for_inference`), etc.

### Estimated effort

- Tasks 1-10 (foundation + library + CLI): ~3-4 hours of TDD-paced engineering on the local box (no GPU required for the unit tests).
- Tasks 11-14 (lift + rewire eval): ~1-2 hours.
- **REVIEW CHECKPOINT 2** before Task 15 — push branch, sync GPU box.
- Tasks 15-16 (smoke + 50-piece eval on GPU box): ~3-5 hours wall (smoke ~5 min, corpus ~2-3h inference + ~30-50min scoring).

Total: ~7-12 hours including the GPU-box runs.

## Important things to know

- **LM Studio was down at the start of today's session.** Local-LLM delegation went to a sonnet `general-purpose` / `Explore` subagent instead. If LM Studio is up next session, you can route diagnostic queries through `delegate_to_local_agent` per the user's CLAUDE.md preference.
- **The spec's two-pass eval split is load-bearing**, not stylistic. The archived `eval/run_lieder_eval.py` documents a 43 GB OOM at piece 6/20 from in-process metric scoring. Phase 1 (inference) stays in-process; Phase 2 (scoring) stays subprocess-isolated. Don't merge them.
- **The smoke piece on `lc6623145` must score well above the per-staff `0.067` baseline** before running the corpus eval. Per spec ship-gate criterion 2, near-baseline smoke is a stop-and-investigate signal — likely Stage A format mismatch or sidecar loss.
- **Pre-existing test failures locally**: `tests/data/test_multi_dpi.py` (ImageMagick on Windows path), `tests/data/test_encoder_cache.py` (torch missing locally). Not regressions; ignore when verifying.
- **The plan committed to `eval/lieder_split.py` already being live** — it's at `eval/lieder_split.py` (74 lines), do not lift it from archive.

## Deferred — explicit non-goals for this subproject

These are listed in the spec's "Deferred follow-ups" section. Do **not** start them in the same session:

- TorchAO inference quantization experiment (separate doc at `/home/ari/docs/clarity_omr_radio_torchao_evaluation.md`).
- Refactor `src/eval/evaluate_stage_b_checkpoint.py` to call the new `load_stage_b_for_inference` helper.
- Subprocess-per-piece inference fallback flag (only if observed OOMs warrant it).

## References

- External review (drove 5 spec fixes): `/home/ari/docs/clarity_omr_radio_subproject4_review.md`
- TorchAO follow-up evaluation: `/home/ari/docs/clarity_omr_radio_torchao_evaluation.md`
- Predecessor handoff: [`2026-05-10-per-system-cleanup-wrap.md`](2026-05-10-per-system-cleanup-wrap.md) (the cleanup that opened the door to Subproject 4)
