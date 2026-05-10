# Per-System Cleanup Wrap-Up (2026-05-10)

> Supersedes [`2026-05-10-radio-stage3-phase2-mid-handoff.md`](2026-05-10-radio-stage3-phase2-mid-handoff.md). Phase 2's lieder ship-gate could not be measured because the lieder driver feeds per-staff crops to a system-trained Stage 3 v2 — fundamental format mismatch. The session pivoted to a full per-staff archival + re-alignment of the decision-gate to the per-system product rule.

## What was already on the branch (pre-session)

- `e87cd94` — Phase 2 mid-handoff doc
- `877e172` — per-dataset eval matrix (4 JSONs at 500 samples each, S2v2 vs S3v2 on `_systems` and `_full` manifests)
- 3 earlier commits adding the eval-driver patches + decision-gate + stratified analyzer

## Discovery

The Phase 2 plan was to run `eval/run_lieder_eval` (50 pieces, beam=5) as the architectural ship-gate. While the synthetic eval was running mid-session, the user pointed out the lieder smoke workdir crops were per-staff (`page_NNNN__sysXX__staffYY.png`), but Stage 3 v2 was trained on per-system crops (multi-staff stacked, `__sys00.png`). Three lines of evidence confirmed the mismatch:

1. Lieder smoke workdir at `eval/results/lieder_stage3_v2_smoke_workdirs/lc6623145/crops/` — every crop is one staff
2. `src/data/manifests/token_manifest_stage3.jsonl` first-line — `image_path` points at `system_crops/.../sys00.png`, with `staves_in_system=3` and `staff_indices=[0,1,2]` confirming multi-staff stacked
3. `src/pdf_to_musicxml.py` Stage A path emits both `system_index` AND `staff_index` ([src/cli.py:62-63](archive/per_staff/src/cli.py#L62-L63), [src/pdf_to_musicxml.py:349-350](archive/per_staff/src/pdf_to_musicxml.py#L349-L350)) — per-staff cropping is the only mode

The lieder smoke onset_f1 of 0.067 was therefore measuring Stage 3 v2's response to wrong-format inputs, not its true capability. The synthetic eval at S3v2=39.7 (well below the 90 floor) is OOD-generalization (dpi150 vs training dpi300), not the same problem.

## Product rule, clarified

The user clarified the model's intended capability:

- **Inference**: a system bbox (1+ staves stacked) goes in, tokens come out. Naturally single-staff scores (violin solos, monophonic primus) are 1-staff systems
- **Training**: single-staff training data must come from naturally single-staff *sources*. Don't take a 2-staff piano grand-staff and feed each staff separately — that pollutes single-staff exposure with split-from-system data

Per this rule, the `grandstaff` (single-staff) variant — split from a 2-staff source — is NOT a valid product input. `grandstaff_systems` (the 2-staff system format) IS. The `primus` and `cameraprimus` single-staff variants are valid because their sources are naturally 1-staff.

## Outcomes (commits on this branch)

| Commit | Subject |
|---|---|
| `e05ca05` | drop single-staff floors initial (superseded by `8b8da66`) |
| `549c40f` | rename `train_stage2_radio_systems.yaml` → `train_stage2_radio_polyphonic.yaml` (it's an 80% systems / 20% single-staff mix, not pure-per-system) |
| `8b8da66` | restore primus single-staff as dynamic regression tripwire; grandstaff (split) stays ungated; `_resolve_cameraprimus_floor` → `_resolve_dynamic_floor` |
| `f9606bc` | Phase A — extract decoder-runtime helpers from `src/cli.py` to `src/inference/decoder_runtime.py`; rewire 2 import sites + 1 test patch |
| `c54a6a0` | Phase B — git mv 24 per-staff files (source + tests + scripts) to `archive/per_staff/`; drop dead `build_stage_a_model` from `src/train/model_factory.py` |
| `8b72564` | Phase C — README documents the per-staff archival + product rule |

### Decision-gate state after this session

5 floors gated:
- **Static system floors**: `synthetic_systems` ≥ 90, `grandstaff_systems` ≥ 95, `primus_systems` ≥ 80
- **Dynamic regression tripwires** `max(75, baseline-5)`: `cameraprimus_systems`, `cameraprimus`, `primus`

Not gated: `grandstaff` (single-staff) — invalid eval data per the rule.

### Risks resolved

- **Risk #1** (Stage 2 v2 mix): Option (d) — rename + document. No retraining; `train_stage2_radio_polyphonic.yaml` carries the 80/20 mix forward as the v2 anchor
- **Risk #2** (single-staff floors): aligned to the product rule per above
- **Risk #5** (system YOLO weights path): confirmed at `runs/detect/runs/yolo26m_systems/weights/best.pt` on the GPU box (44 MB, 2026-05-04)

## Sync state at end of session

- Local: `/home/ari/work/Clarity-OMR-Train-RADIO` at HEAD `8b72564`
- Origin: `feat/per-system-cleanup` (renamed from `feat/stage3-phase2-evaluation`) at HEAD `8b72564`
- GPU box: `10.10.1.29:Clarity-OMR-Train-RADIO` at HEAD `8b72564`

## Open work — Subproject 4

The replacement: a per-system end-to-end inference pipeline. Sketch (per audit):

1. New Stage A using `runs/detect/runs/yolo26m_systems/weights/best.pt` + `src/models/system_postprocess.py:extend_left_for_brace` — produce one PNG per system bbox, `__sysXX.png`, no `staff_index`
2. Stage B = Stage 3 v2 unchanged — already accepts system crops
3. Token splitter — reuse `src/data/convert_tokens.py:_split_staff_sequences_for_validation` (line 168) to chunk system token output at `<staff_end>` boundaries
4. Assembly — extend `src/pipeline/assemble_score.py` (don't rewrite) with `assemble_score_from_system_predictions(...)` that consumes the chunked sequences via the existing `StaffRecognitionResult` API
5. End-to-end smoke on lc6623145 before any corpus eval

Estimated: 4–8 h engineering, then re-running the lieder eval becomes a 2–3 h run that produces a real onset_f1.

## Things NOT done that the next session might want

1. **Subproject 4 spec** (the per-system inference pipeline). Branch off main once this PR merges
2. **Re-run synthetic eval** — synthetic_systems S3v2 = 39.7 indicates dpi150 OOD weakness, but S2v2 number was never collected (e6 killed at 49% to pivot to cleanup). If the verdict ever needs the architectural-bet floor, re-run
3. **Audit `src/data/yolo_aligned_crops.py`** — kept in `src/` because `yolo_aligned_systems.py` imports from it; arguably the shared utilities should be lifted out of a per-staff-named module
4. **Squash `e05ca05` + `8b8da66`** — they describe the same decision (single-staff floors) at two stages of clarity. Already-pushed; squash would need force-push. Skipped here

## Reference

- Audit: see the Plan agent's report at the top of this branch's session transcript (not committed; lives in conversation history)
- Per-staff archive: [`archive/per_staff/`](../../../archive/per_staff/)
- New decoder runtime: [`src/inference/decoder_runtime.py`](../../../src/inference/decoder_runtime.py)
- Decision-gate: [`eval/decision_gate.py`](../../../eval/decision_gate.py)
- Renamed Stage 2 config: [`configs/train_stage2_radio_polyphonic.yaml`](../../../configs/train_stage2_radio_polyphonic.yaml)
