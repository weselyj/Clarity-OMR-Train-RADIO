# Stage D part-alignment fix + 4-piece demo eval (2026-05-11)

> Reproduced the upstream Clarity-OMR 4-piece MIR evaluation on the Stage 3 v2 RADIO checkpoint and fixed a Stage D MusicXML export bug that caused every prediction to be rejected by MuseScore as "corrupt." End-to-end verified.

## State at end of session

| Box | Branch | HEAD |
|---|---|---|
| Local (`/home/ari/work/Clarity-OMR-Train-RADIO`) | `fix/stage-d-part-alignment` | local commit |
| GPU box (`seder` / `10.10.1.29`) | `main` (synced earlier this session) | `6d38d26` (pre-fix) |
| Origin | `main` | `6d38d26` (pre-fix), feature branch pushed |

The GPU box still has the running `running_total` patch applied to `src/pipeline/export_musicxml.py` from the in-session iteration, plus the driver and tests under untracked paths. After PR review/merge, `seder` should `git fetch && git checkout main && git pull` to drop those local edits in favor of the merged copy.

## What was broken

All 4 MusicXML outputs from the demo eval opened with **"corrupt MusicXML"** errors in MuseScore 4. `music21` parsed them happily and scored them, which masked the issue from any python-only verification path.

Direct measurement on the 4 produced files:

| File | part 1 measures | part 2 measures | extra at |
|---|---:|---:|---|
| `clair-de-lune-debussy.musicxml` | 96 | 97 | tail of part 2: `[97]` |
| `fugue-no-2-bwv-847-in-c-minor.musicxml` | 21 | 19 | tail of part 1: `[20, 21]` |
| `gnossienne-no-1.musicxml` | 28 | 27 | tail of part 1: `[28]` |
| `prelude-in-d-flat-major-op31-no1-scriabin.musicxml` | 38 | 37 | tail of part 1: `[38]` |

Every divergence was at the tail with no middle gaps and no duplicate measure numbers — consistent with one or more orphan single-staff systems contributing measures to only one part.

## Root cause

Partwise MusicXML requires every `<part>` to share the same measure-index timeline. `music21` is permissive about violations; MuseScore is strict.

Pipeline flow that hit the bug:

1. [src/pipeline/assemble_score.py:339-347](../../src/pipeline/assemble_score.py#L339-L347) — within each system, per-staff measure counts are normalized to `canonical_measure_count = max(staff_measure_counts)`. This correctly aligns staves *within* a system.
2. [src/pipeline/assemble_score.py:192-237](../../src/pipeline/assemble_score.py#L192-L237) — `_merge_undersized_systems` merges *consecutive* undersized systems on the same page, but **leaves a trailing/orphaned single-staff system as a 1-staff group when it has no neighbor to merge with**.
3. [src/pipeline/assemble_score.py:108-126](../../src/pipeline/assemble_score.py#L108-L126) — `_resolve_part_label` assigns a lone undersized-system staff to a part by *clef family* (treble → RH, bass → LH). The other part receives no staff from this system.
4. [src/pipeline/export_musicxml.py:925-934 (pre-fix)](../../src/pipeline/export_musicxml.py) — the music21 export loop iterates `systems → staves` and appends each staff's tokens to its part. **No padding was added to parts that didn't receive a staff from a given system.** Net effect: that part fell behind by `canonical_measure_count` measures.

The same shape can also be produced by:
- `append_tokens_to_part_with_diagnostics` silently dropping a malformed measure (decoder-emitted bad tokens), making the served part shorter than `canonical_measure_count`.
- A `ValueError` caught in the lenient try/except path (partial append).
- `post_process_tokens` collapsing measures in principle (unobserved in this run).

## Fix

[src/pipeline/export_musicxml.py](../../src/pipeline/export_musicxml.py) — added per-system **running_total padding** to both `assembled_score_to_music21` and `assembled_score_to_music21_with_diagnostics`:

```python
running_total = 0
for system in systems:
    for staff in system.staves:
        # ... append staff tokens to its part (try/except in diagnostics variant)
    running_total += system.canonical_measure_count
    for label in score.part_order:
        part = parts.setdefault(label, stream.Part(id=label))
        deficit = running_total - len(part.getElementsByClass(stream.Measure))
        if deficit > 0:
            _pad_part_with_empty_measures(
                part, deficit, diagnostics=diagnostics, strict=strict,
            )
```

`_pad_part_with_empty_measures` synthesizes `[<measure_start>, rest, _whole, <measure_end>]` token sequences (the same shape `_normalize_measure_count` uses internally) and routes them through the existing `append_tokens_to_part(_with_diagnostics)` machinery — keeps measure-numbering consistent and reuses one code path for measure construction.

Why running_total rather than a per-served-set check: the per-served approach handled the orphan-system case but missed silent token drops (a staff appended fewer measures than its declared `canonical_measure_count`). Running_total is computed from actual `len(part.measures)` after each system, so it self-corrects for every cause of measure-count divergence — missing staves, silent drops, partial appends, anything.

A new diagnostics counter — **`StageDExportDiagnostics.padded_measures`** — records how many pad measures were inserted (visible in the `.musicxml.diagnostics.json` sidecar).

## Verification

| Source | Result |
|---|---|
| Local 4 padded-by-hand copies | All open cleanly in MuseScore 4 (user confirmed) — establishes that measure-count alignment was the sole structural cause. |
| Regression tests on GPU box | 14/14 pipeline tests pass — `tests/pipeline/test_export_musicxml_part_alignment.py` (4 tests, covers balanced + trailing-orphan + leading-orphan + diagnostics-counter) and `tests/pipeline/test_export_part_padding.py` (3 tests, parallel coverage). |
| Fresh 4-piece run on Stage 3 v2 RADIO | All 4 outputs have aligned parts: clair `[99, 99]`, fugue `[24, 24]`, gnoss `[31, 31]`, prelude `[39, 39]`. |
| Diagnostics sidecars | `padded_measures` counter populated: 5 / 8 / 7 / 3 across the 4 pieces. |

## 4-piece demo eval result (Stage 3 v2 RADIO, greedy, 300 dpi)

| Piece | onset_f1 | note_f1 | overlap | quality | padded | skipped_sys |
|---|---:|---:|---:|---:|---:|---:|
| Clair de Lune (Debussy) | 0.0315 | 0.0119 | 1.000 | 26.3 | 5 | 0 |
| Fugue No. 2 BWV 847 (Bach) | 0.0631 | 0.0299 | 1.000 | 32.8 | 8 | 0 |
| Gnossienne No. 1 (Satie) | 0.1032 | 0.0372 | 1.000 | 34.2 | 7 | 2 |
| Prelude Op. 31 No. 1 (Scriabin) | 0.0246 | 0.0035 | 1.000 | 25.8 | 3 | 0 |
| **mean** | **0.0556** | **0.0206** | **1.000** | **29.8** | | |

mir_eval config matches upstream's `eval/upstream_eval.py` verbatim: 50 ms onset tolerance, 50 cents pitch tolerance, `stripTies` canonicalization, default `offset_ratio=0.2` for full-note F1. Scoring runs in a subprocess per piece to keep `music21`/zss state bounded.

**Score impact of the padding fix is negligible** (pre-fix mean onset_f1 0.0544 → post-fix 0.0556). Expected — whole-rest pad measures contribute no note events to mir_eval matching.

**Where this sits in project baselines:** mean 0.0556 is below the lieder corpus mean for the same checkpoint (0.0819) and far below the project's "mixed" decision gate (0.241).

## What landed this session

Working-tree changes committed on `fix/stage-d-part-alignment`:

| Path | Status | Purpose |
|---|---|---|
| `src/pipeline/export_musicxml.py` | M | `_pad_part_with_empty_measures` helper; running_total padding in both export variants; `padded_measures` diagnostics field. |
| `tests/pipeline/test_export_musicxml_part_alignment.py` | A | 4 regression tests (balanced, trailing-orphan, leading-orphan, diagnostics variant + counter assertion). |
| `tests/pipeline/test_export_part_padding.py` | A | 3 parallel regression tests added by hook automation in-session; kept for additional coverage. |
| `eval/run_clarity_demo_radio_eval.py` | A | New driver — system-level reimplementation of the archived per-staff `archive/per_staff/eval/run_clarity_demo_eval.py`. Loops `SystemInferencePipeline` once over the 4 canonical stems, subprocesses `eval.upstream_eval` for scoring, writes per-piece JSON + `summary.json`. Supports `--stems` for debug-time subsetting. |

The `eval/run_clarity_demo_radio_eval.py` driver retains a small block of debug logging gated on `STAGE_D_DEBUG_LOG` env var; harmless when unset.

The `archive/per_staff/eval/run_clarity_demo_eval.py` (legacy) and `archive/per_staff/eval/score_demo_eval.py` (legacy) remain archived; not touched.

## Operational notes

- **SSH stability into seder was intermittent.** Multiple inference runs (5–7 min/piece) were interrupted by SSH drops, which propagate via process tree to kill the python worker. The user-side `run_demo_eval_logged.cmd` (a separate launcher created mid-session by user automation) works around this by detaching from the SSH session; it's still present on `seder` and re-runs survive SSH disconnects.
- **GPU memory contention** was the root cause of one mid-session failure cluster — zombie python workers from earlier disconnected runs held 5-15 GB each. `taskkill /F /IM python.exe` cleared them. Future runs should kill stragglers before kicking off a new attempt.
- **Local CPU has no torch.** All pipeline tests (`tests/pipeline/`, `tests/inference/`, etc.) are CUDA-gated at collection via `tests/conftest.py`; they pass on `seder`'s `venv-cu132` but are skipped locally. The pure-data tests in `tests/data/` still run locally where applicable.

## Why transcription quality is unchanged

The fix is structural only — it makes MusicXML output schema-conformant for MuseScore. **mir_eval scores barely changed** (Δ ≤ 0.02 per piece) because whole-rest pad measures contribute no note events.

The underlying transcription quality issue is **decoder under-generation**: each piece shows 173-490 fewer notes than its reference. Likely causes (in priority order, unverified):

1. **Stage B `--max-decode-steps 2048` cap.** Clair de Lune's reference has 1623 notes and is the largest; the decoder may be hitting the cap mid-system on dense piano content.
2. **Stage A under-detecting systems** on the demo PDFs (each pre-fix Stage-D diagnostic showed 0 or 2 `skipped_systems`, so this is at most a partial explanation).
3. **Token-grammar FSA biasing** toward `rest`/short outputs when uncertain.

None of these are touched by this PR.

## Open follow-ups

- Investigate the under-generation. Easiest first cut: re-run with `--max-decode-steps 4096` (or higher) on Clair de Lune alone and see whether note count rises. Larger experiment: log per-system token-emission stop reason from `_decode_stage_b_tokens` to distinguish "hit `<eos>`" from "hit step cap."
- Consider also reproducing the upstream's DaViT checkpoint through this same harness on the 4 pieces for an apples-to-apples comparison. The archived `archive/per_staff/eval/run_baseline_reproduction.py` exists for that path; it'd need a small port to the system-level CLI similar to what `run_clarity_demo_radio_eval.py` did.
- After PR merge, `seder` should sync: `git fetch && git checkout main && git pull` to drop the in-session uncommitted state.
- The `STAGE_D_DEBUG_LOG` instrumentation in `assembled_score_to_music21_with_diagnostics` is currently in-tree for one-off triage. Consider removing or moving behind a `pytest`-only switch in a follow-up if it shouldn't ship.

## References

- Driver: [`eval/run_clarity_demo_radio_eval.py`](../../eval/run_clarity_demo_radio_eval.py)
- Tests: [`tests/pipeline/test_export_musicxml_part_alignment.py`](../../tests/pipeline/test_export_musicxml_part_alignment.py), [`tests/pipeline/test_export_part_padding.py`](../../tests/pipeline/test_export_part_padding.py)
- Upstream eval (vendored verbatim): [`eval/upstream_eval.py`](../../eval/upstream_eval.py)
- Upstream model card (HF): https://huggingface.co/clquwu/Clarity-OMR — 4 canonical demo pieces showcased there
- Archived per-staff demo eval (replaced by `run_clarity_demo_radio_eval.py`): [`archive/per_staff/eval/run_clarity_demo_eval.py`](../../archive/per_staff/eval/run_clarity_demo_eval.py)
- Stage 3 v2 checkpoint: `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt` (3.2 GB; on `seder` only)
- Demo data: `data/clarity_demo/{pdf,mxl}/` (4 PDFs + 4 reference `.mxl` files; on `seder` only, gitignored)
- Predecessor handoff: [`2026-05-10-subproject4-shipped.md`](2026-05-10-subproject4-shipped.md)
