# Sub-plan A (strict eval & gate harness) — execution notes & Sub-plan-D pre-integration caveats

> Written at end of Sub-plan A execution. Plan: `2026-05-17-robust-stage-a-eval-gate-harness.md`.
> Spec: `../specs/2026-05-17-robust-stage-a-clutter-detection-design.md`.
> **Status: COMPLETE & PASS.** All 6 tasks shipped + 4 review-driven follow-ups.
> 32/32 CPU tests pass (`python3 -m pytest tests/robust_stage_a/ -v`), none skipped.
> `HEAD` of the chain = `322004f`; all on `main`, pushed.

## Deliverables

- `eval/robust_stage_a/manifest.py` — pure schema + validating loader.
- `eval/robust_stage_a/gate.py` — pure scoring/gate (no torch/ultralytics; CPU-tested).
- `eval/robust_stage_a/run_gate.py` — thin CLI/GPU seam (NOT unit-tested in A, by design — Sub-plan D exercises inference on seder).
- `tests/robust_stage_a/{test_manifest.py,test_gate.py}` — 32 CPU tests.

## Commit chain (all on main)

```
06d3035 feat  manifest schema + validator              (Task 1)
bb3fded fix   reject has_lyrics=true + empty lyric_bands (review follow-up A)
6fb9547 feat  IoU + greedy pred/GT matching             (Task 2)
9ca27aa feat  per-scenario scoring                      (Task 3)
5021fa0 fix   reject is_non_music=false + empty systems  (review follow-up B)
1562fa3 feat  lyric-system recall + combined gate        (Task 4)
a7abb91 feat  lieder-recall CSV reader                   (Task 5)
dd0add5 feat  CLI orchestrator + verdict_to_report       (Task 6)
68f61dc perf  load YOLO once in run_gate, not per scan   (review follow-up)
322004f refac drop dead import, type _geometry_ok, eps note (cleanup)
```

## Intentional deviations from the verbatim plan (all reviewed, spec-aligned)

1. **`match_predictions` tie-break** (`6fb9547`): plan's `if v >= best_iou` with
   `best_iou=match_iou` made the *last* of two equal-IoU GTs win, failing the
   plan's own `test_match_one_pred_cannot_take_two_gt`. Changed to
   `best_iou = match_iou - 1e-9` + `if v >= match_iou and v > best_iou`
   (standard lowest-index greedy tie-break). Tests were NOT weakened.
2. **Loader hardening A** (`bb3fded`): reject `has_lyrics=true` + empty/null
   `lyric_bands` — otherwise `_geometry_ok`'s `any(... for lb in [])` is
   vacuously true and silently inflates lyric-system recall (the spec's
   central risk).
3. **Loader hardening B** (`5021fa0`): reject `is_non_music=false` + empty
   `gt_systems` — otherwise such a scenario takes the music branch with zero
   GT and vacuously passes the strict per-scenario gate.
4. **YOLO load hoisted** (`68f61dc`): `_infer` reloaded the checkpoint every
   scenario (15–50× redundant loads on the held-out run). Now loaded once in
   `main()`; `from ultralytics import YOLO` stays inside `main()` *after*
   `parse_args()` so `--help` / CPU-import still works with no ultralytics.

## Accepted conventions (decided, do NOT re-litigate without new evidence)

- **Frozen dataclasses use `list[...]` fields harness-wide** (`GtSystem`,
  `Scenario`, `MatchResult`, `GateVerdict`). `frozen=True` is shallow in
  Python (blocks rebinding, not in-place mutation). Accepted because: these
  are load-once read-only value objects in a single-threaded one-shot CLI,
  nothing mutates them anywhere in the 6 tasks, and the tuple "fix" would
  break the plan's verbatim `== [...]` equality tests. If true immutability
  is ever wanted it must be an all-or-nothing pass that also updates those
  tests deliberately.

## Sub-plan-D pre-integration caveats (READ before the first seder gate run)

`run_gate.py` is intentionally un-unit-tested. Before/while wiring Sub-plan D:

1. **Lieder CSV must have integer-formatted counts.** `recall_from_stagea_csv`
   does `int(row["expected_p1_staves"])`; `int("3.0")` raises `ValueError`.
   Confirm `eval/score_stage_a_only.py` writes integers (it does today — spot
   check the committed `eval/results/stagea_baseline_pre_faintink.csv` before
   the run). Fail-loud on a malformed/short CSV is intentional (a corrupt
   baseline must not silently pass the gate).
2. **Missing image paths give context-free errors.** `load_manifest` does not
   check `sc.image` existence; a missing file makes ultralytics raise without
   the `scenario_id`. Pre-validate all `sc.image` are readable before the
   inference loop, or wrap `_infer` to re-raise with `scenario_id`.
3. **Do not point `--lyric-baseline` and `--lyric-baseline-out` at the same
   file.** The out-snapshot is written before the baseline is read, so a
   shared path silently self-baselines (always "OK", no regression detection).
4. **`verdict_to_report` OK/REGRESSION uses bare `>=`** (ignores `eps`). It
   agrees with `combined_gate` only while `NO_REGRESSION_EPS == 0.0` (current).
   If a non-zero eps is ever introduced, thread it into `verdict_to_report`
   (or store per-metric pass booleans on `GateVerdict`) or the human-readable
   label can contradict the `GATE:` header.
5. **Lyric-baseline re-snapshot:** if the held-out manifest initially has no
   lyric-bearing scenarios, `lyric_system_recall` snapshots 1.0; adding lyric
   scenarios later will read as a spurious regression against that 1.0.
   Re-snapshot `--lyric-baseline-out` after expanding the manifest with lyric
   archetypes.

## Not in scope (per the decomposition — each its own spec→plan cycle)

Sub-plan B (retrain-hardening / epoch-~34 NaN), C (data engine /
clutter+hard-neg), D (iterative robustness loop, integrating A+B+C, abstention,
the explicit-text-class contingency). Phase 3 (bass-clef / Stage-B clef bias)
stays parked until robust-Stage-A lands.
