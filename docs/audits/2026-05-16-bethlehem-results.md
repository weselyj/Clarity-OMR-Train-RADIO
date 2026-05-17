# Bethlehem clean-transcription — Phase 2 results & verdict (2026-05-16)

**Plan:** [docs/superpowers/plans/2026-05-16-bethlehem-clean-transcription-plan.md](../superpowers/plans/2026-05-16-bethlehem-clean-transcription-plan.md)
**Spec:** [docs/superpowers/specs/2026-05-16-bethlehem-clean-transcription-design.md](../superpowers/specs/2026-05-16-bethlehem-clean-transcription-design.md)

## Verdict: **Phase 2 PASS**. Defect 1 resolved end-to-end. Defect 2 persists → Phase 3 triggered (separate spec).

## What was done

- **Task 1** GT scoring harness (`scripts/audit/score_against_gt.py`) — committed `6dfe84f`/`77d4963`.
- **Task 2** faint-ink Stage-A augmentation arm — committed `d6e00f9`; the `A.Morphological(erosion)` form crashed detection training (applied to bboxes → `cv2.erode (-215)`), refactored to an image-only `ImageOnlyErosion` (`1ffe7df`→`fd3f903`, TDD + spec & code-quality reviews; `e025c65` py_random determinism).
- **Task 3** Stage-A regression baseline — scorer nested-GT bug fixed (`a4387c1`, recursive `{stem:path}` index), pre-faint-ink baseline snapshot `eval/results/stagea_baseline_pre_faintink.csv` (`ce30da7`): **recall 0.930** (145 pieces, 425/457).
- **Task 4** faint-ink Stage-A YOLO retrain on seder. Operational fixes en route: `--batch 4` pinned (autoselect=8 overflowed VRAM, `90f7b49`); worker run under `$ErrorActionPreference=Continue` (PowerShell `Stop` made git/libpng stderr fatal, `18f8f2d`); worker UTF-8 logging. Retrain stopped at epoch 53 via `EarlyStopping(patience=20)`; an EMA NaN at ~epoch 34 did **not** compromise the deliverable — Ultralytics skips checkpoint saves while EMA is NaN, so `best.pt` is the pre-NaN **epoch-33** snapshot (validated: mAP50 0.995, mAP50-95 0.930, 0/768 non-finite weights). Provenance-validation requirement added to plan Step 4.4 (`54326c2`).

New Stage-A weights (seder): `runs/detect/runs/yolo26m_systems_faintink/weights/best.pt`.

## Phase 2 gate evidence

| Gate | Criterion | Result | Verdict |
|---|---|---|---|
| **5.1 Bethlehem recovery** | 4 systems @ conf 0.25, clean geometry, a detection in y≈1319–2035 | **4/4 systems**; faint leading system at **y=1321–2042, conf 0.950**; all confs 0.94–0.95, full-width, no overlap/junk | **PASS** |
| **5.2 No-regression** | new aggregate Stage-A recall ≥ baseline − 0.01 | new **0.9344** vs baseline **0.9300** (Δ **+0.0044**); 145 pieces, **0 regressed**, 0 worsened | **PASS** |
| **5.3 E2E re-score** | measure_recall ≥ 0.95 (all 4 systems reach decoder) | **measure_recall 1.000 (17/17 both parts)** vs v4 0.765 (13/17), Δ **+0.235** | **PASS** |

E2E detail (new Stage-A + current Stage-B default `full_radio_stage3_v3`, greedy, conf 0.25):

```
measure_recall   1.0      (v4: 0.7647)   parts 2/2, measures [17,17]==GT [17,17]
clef_accuracy    0.75     (v4: 0.8333)   pred_clefs P0=[G2,G2,G2,G2]  P1=[F4,G2,F4,G2]
note_onset_f1    0.1228   (v4: 0.08)     tracked, not gated
```

## Interpretation

- **Defect 1 (Stage-A missed the faint leading system) — RESOLVED.** The faint-ink augmentation arm + retrain makes Stage-A detect all 4 Bethlehem systems at default conf, and end-to-end this delivers the previously-missing measures: 17/17 vs 13/17. No detection regression on the 145-piece lieder eval (recall slightly up).
- **Defect 2 (rest-heavy bass staff clef-flips to treble G2) — STILL PRESENT.** Part-1 clef sequence `[F4, G2, F4, G2]`: 2 of 4 bass systems emitted as G2. The clef_accuracy "drop" 0.833→0.75 is a denominator artifact (recovering the 4th system added 2 clef occurrences); the per-system flip pattern is unchanged from v4. This is a Stage-B behavior, explicitly deferred by the spec.

## Phase 3 trigger decision (plan Step 5.5)

Phase 2 PASS **and** the Step-5.3 re-score still shows the rest-heavy bass flipping to G2 → per the plan and spec sequencing, **Phase 3 (Stage-B clef bias) is triggered and gets its own spec + plan via brainstorming** (it was deferred by design because its task shape was empirically contingent on this re-score). Phase 3 is **not** started here. Phase-3 baseline reference: v3 lieder mean clef metric 0.2398 (per plan).

This plan's Phase 1–2 scope is **complete**.
