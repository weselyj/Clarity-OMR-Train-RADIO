# Stage 3 v2 Failure-Mode Investigation

**Status:** design approved 2026-05-11 — ready for implementation plan
**Branch off:** `feat/stage3-v3-retrain` at `88592df` (continuation, not a new branch)
**Predecessor:** [`docs/audits/2026-05-11-stage3-v3-retrain-results.md`](../../audits/2026-05-11-stage3-v3-retrain-results.md) — frankenstein diagnostic disproved encoder-drift hypothesis (mean onset_f1 +0.001, within noise)

## Goal

Find the dominant failure mode(s) that the Stage 3 v2 audit's encoder-drift hypothesis did not account for. The frankenstein experiment showed that even pairing Stage 3 v2's decoder with Stage 2 v2's encoder (the features the decoder was supposedly trained against) does not move demo onset_f1 measurably. Something else is breaking the model. This investigation runs two parallel diagnostics to identify the failure mode before any retrain decision.

## Non-goals

- **Not fixing the failure mode.** This investigation diagnoses; the fix is a separate sub-project once we know what's broken.
- **Not retraining.** No GPU training time committed until the actual failure mode is identified.
- **Not running A2 or Phase B (full process audit).** A2 was already run (FAIL) and Phase B is a broader sweep — both are out of scope here.
- **Not running the DaViT upstream baseline.** That's a parallel architectural-comparison thread; this investigation stays scoped to RADIO + Stage 3 v2's existing checkpoint.
- **Not refactoring the inference pipeline.** Stream B *instruments* the pipeline (read-only counters); any code changes implied by findings live in a follow-up sub-project.

## Why now

The frankenstein diagnostic gave a definitive negative result — encoder-cache staleness is not the dominant failure mode. Before any retrain commits GPU time, we need to identify what IS. The audit explicitly flagged A3 (decoder-on-training round-trip) and Phase B as "relevant only if Phase A passes but downstream still disappoints." That's now where we are. Stream B (pipeline-stage note loss) extends the audit's scope to cover assembly / post-processing, which the audit didn't.

## Scope

Two parallel diagnostic streams, run sequentially in a single sub-project. Both operate on the existing Stage 3 v2 checkpoint (`checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt`) on seder; neither modifies the model.

| Stream | What it tests | Cost |
|---|---|---|
| A — Decoder-on-training round-trip | Does Stage B reproduce its own training labels at inference? | ~30 min GPU + 1h human |
| B — Pipeline-stage note loss | Where in the inference pipeline do notes get lost? | ~2-4h human + ~30 min GPU |

## Architecture

### Stream A — A3 (decoder round-trip on training data)

Implementation was sketched in the original audit plan (Task 4) but skipped under fast-exit. Reuse that sketch:

- Pick 20 training samples (5 per corpus × 4 corpora) via the existing `scripts/audit/_sample_picker.py`
- For each, run Stage B inference (same code path as the demo eval — encoder + decoder via `_load_stage_b_crop_tensor` → `_encode_staff_image` → `_decode_stage_b_tokens`)
- Compare predicted tokens to ground-truth labels from the manifest
- Metrics:
  - Mean token-position accuracy (per-position match rate, computed over `min(len(pred), len(target))`)
  - Exact-match sequence rate
  - Per-class accuracy for `note-*`, `rest`, `timeSignature-*`, `keySignature-*` token classes
  - Per-sample exact predicted-vs-target first 50 tokens (for inspection)

**Triage thresholds (from the audit plan, intended as advisory, not certification):**
- Mean token accuracy ≥ 80% → decoder reproduces what it was trained on; gap is elsewhere (assembly or domain/methodology)
- Mean token accuracy 50-80% → partial convergence or significant noise; needs interpretation
- Mean token accuracy < 50% → decoder is genuinely broken or undertrained

### Stream B — Pipeline-stage note loss

Wraps the existing inference pipeline (`SystemInferencePipeline` + Stage D export) with note-counting instrumentation at six observable boundaries. Operates on a single piece at a time (start with Clair de Lune — largest, most under-generation). One JSON output per piece with stage-by-stage counts.

**Pipeline stages instrumented:**

| Stage | Source of count | What's counted |
|---|---|---|
| 1. Raw decoder output (per system) | output of `_decode_stage_b_tokens` | tokens matching `^note-` |
| 2. After staff split | output of `_split_staff_sequences_for_validation` | tokens matching `^note-` per staff |
| 3. After `_enforce_global_key_time` + `post_process_tokens` | normalized tokens after `assemble_score` line ~348 | `^note-` per staff |
| 4. AssembledScore tokens | `AssembledScore.systems[*].staves[*].tokens` | `^note-` per staff |
| 5. music21 Notes (in-memory) | `assembled_score_to_music21_with_diagnostics` return value | `len(score.flatten().getElementsByClass(music21.note.Note))` (chord-internal notes counted via `Chord.notes`) |
| 6. MusicXML file | re-parse the output `.musicxml` via music21 | same count as stage 5, post-write |
| ref | reference `.mxl` parsed via music21 | same count |

The most informative comparison is the *drop between stages*. If stage 1 reports 1000 notes and stage 5 reports 200, the assembly pipeline is losing 800 notes — that's a concrete bug to fix. If stage 1 already reports 200 against a reference of 1000, the decoder is the bottleneck.

**Implementation pattern:** add a thin wrapper around the existing inference path that hooks each boundary, rather than modifying `system_pipeline.py` directly. Keeps the diagnostic out of production code.

## Deliverable

Extend `docs/audits/2026-05-11-stage3-v3-retrain-results.md` with three new sections:

1. **Stream A — Decoder round-trip results**: token accuracy table per corpus + per-class accuracy + interpretation
2. **Stream B — Pipeline-stage note loss results**: per-piece stage-by-stage table + identified bottleneck stage(s)
3. **Revised verdict**: replaces the placeholder verdict. Identifies the dominant failure mode and recommends a follow-up sub-project (decoder retrain with corrected config, assembly fix, etc.).

## File structure

**New files (`feat/stage3-v3-retrain` branch):**
- `scripts/audit/a3_decoder_on_training.py` — Stream A script (was sketched in original audit plan, never written)
- `scripts/audit/pipeline_note_loss.py` — Stream B script

**Generated artifacts (not in git, on seder):**
- `audit_results/a3_decoder_stage3_v2.json` — Stream A output
- `audit_results/pipeline_note_loss_clair_de_lune.json` — Stream B output (one per piece run)

## Risks and decisions made

| Risk | Mitigation |
|---|---|
| Stream A on Stage 3 v2 might be confounded by the encoder drift (decoder was trained against cached features, not live encoder output) | Acknowledge this in the report. The interpretation matters: if Stream A shows low accuracy with the LIVE encoder, but the frankenstein eval (S2 enc + S3 dec) was also low, then encoder-features-the-decoder-was-trained-against aren't being reproduced either. That points further at the decoder itself. |
| Stream B might find note loss at multiple stages, making interpretation messy | Report stage-by-stage counts honestly. If multiple stages lose notes, that's still a useful finding — fix the biggest drop first. |
| Reference note count may include music21-internal artifacts (e.g., tied notes counted once vs twice) that don't match the kern-token concept of "note" | Use a consistent counting convention in the report ("music21 Note objects after stripTies"). The exact count matters less than the cross-stage delta. |
| Stream A and Stream B might give conflicting signals | Honestly possible. If A says decoder reproduces labels well but B shows notes lost in assembly, the conflict resolves cleanly (decoder is fine, fix the assembly). If A says decoder is broken AND B shows notes lost in assembly, fix the bigger drop first. |

## Effort estimate

| Stream | Wall-clock |
|---|---|
| Stream A | ~1 hour (script implementation reuse from audit plan + ~10 min on seder) |
| Stream B | ~3-4 hours (new instrumentation logic + ~10 min per piece on seder) |
| Report writing | ~1 hour |
| **Total** | **~half a day** |

## What follows this sub-project

The investigation's revised verdict will steer one of:

- **Decoder is the bottleneck** → retrain (with proper freeze, possibly other tuning) becomes justified, OR the architecture is genuinely insufficient for piano OMR
- **Assembly is the bottleneck** → small code-fix sub-project to plug the leak, then re-evaluate. No retrain needed.
- **Both contribute, decoder more so** → retrain first; assembly fix later
- **Neither is the bottleneck** (model output is fine, problem is mir_eval methodology / domain gap) → reconsider the eval criterion or the demo-pieces choice; project's "is RADIO viable" question needs different evidence

In all branches, the new diagnostic scripts (`a3_*`, `pipeline_note_loss`) stay in tree as reusable post-training audit tools.
