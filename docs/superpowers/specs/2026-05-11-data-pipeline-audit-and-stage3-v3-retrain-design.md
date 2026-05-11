# Data-Pipeline + Capacity Audit and Stage 3 v3 Retrain

**Status:** design approved 2026-05-11 — ready for implementation plan + handoff to fresh session
**Branch off:** `main` at `b2ed4a2` (post-merge of PR #49)
**Supersedes:** [`2026-05-11-stage3-v3-retrain-design.md`](2026-05-11-stage3-v3-retrain-design.md) (halted at Phase 1 gate; encoder-freeze fix is incorporated here, but the v3 retrain assumptions are no longer the dominant variables)

## Goal

Diagnose whether the synthetic_systems corpus is structurally sound, fix the data-pipeline and (probable) capacity issues identified in this session's audits, and retrain Stage 3 as v3 with the corrected setup. Produce a definitive answer on whether RADIO can reach the project's quality bar when trained correctly with a balanced corpus, proper val coverage, frozen encoder, and adequate model capacity.

## Why this work, now

This session produced a complete diagnostic arc (PRs #46-49). Summary:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Encoder DoRA was unfrozen during S3v2 training | TRUE | PR #48: 384/384 encoder LoRA keys drifted; A2 max_abs_diff 85.15 |
| Encoder drift is the dominant failure mode | FALSE | PR #49: frankenstein (S2v2 enc + S3v2 dec) +0.001 mean onset_f1 vs S3v2 |
| Decoder can reproduce its own training labels | NO on multi-staff | PR #49 Stream A: synthetic 11%, grandstaff 40% vs primus 75%, cameraprimus 87% |
| Assembly pipeline drops notes | NO | PR #49 Stream B: raw decoder is 46-80% of reference; pipeline contributes minor additional loss |
| Training-data setup is the actual root cause | YES | Per-corpus val_loss + manifest counts: synthetic was 77.8% of cached-tier weight on the SMALLEST corpus (20.5k entries) with ZERO val split |

The training of S3v2 was effectively flying blind on its largest gradient source (synthetic_systems), heavily over-weighting a small corpus with no validation feedback. Multi-staff piano content (which is what the demo eval cares about) was simultaneously under-weighted (grandstaff at 10% of gradient on 97k entries) and learned poorly. The fix is at the data-pipeline level, not the model architecture (probably) — but the audit confirms the latter.

Detailed background in memory:
- `project_radio_stage3_v2_audit.md` (encoder DoRA finding)
- `project_radio_frankenstein_diagnostic.md` (audit conclusion disproved)
- `project_radio_training_data_imbalance.md` (actual root cause)

## Non-goals

- **DaViT baseline reproduction** — independent parallel thread; would test architecture-family-wide viability, but isn't the immediate question.
- **Audiveris benchmark** — independent thread; the long-term target but not a near-term gate.
- **Generating a brand-new synthetic corpus from scratch.** If Phase 1 finds synthetic is structurally broken, regeneration is flagged as a separate sub-project rather than bundled here. This sub-project drops or down-weights synthetic if needed.
- **Retraining Stage 1 or Stage 2.** The bug is in Stage 3 data-setup; upstream stages remain fine.

## In-scope changes (decided before Phase 1)

- Add val split to synthetic_systems (one of the two confirmed bugs)
- Rebalance dataset_mix away from synthetic dominance (the other confirmed bug)
- Apply the encoder-freeze fix from the v3 retrain spec (Option B, cache-derived auto-detect) — regression-prevention; not presumed dispositive but should ship in any retrain
- Retrain Stage 3 as v3 with the corrected config (9000 steps; v2 was still descending at 6000)

## Conditionally-in-scope (Phase 1 may add)

- `max_sequence_length` bump from 512 if grand-staff sequences are being truncated
- Decoder size bump if capacity analysis shows the current decoder is too small for multi-staff content
- Drop synthetic_systems entirely (if Phase 1 finds the generator is producing broken data)

## Architecture

Three sequential phases with an explicit gate after Phase 1. The Phase 1 audit drives the Phase 2 implementation; user reviews the audit findings before Phase 2 starts.

### Phase 1 — Data-pipeline + capacity audit (~2-3 days, no GPU training)

Five investigation streams, each contributing a section to a single audit report.

**Stream 1 — Synthetic-corpus inspection.** Sample 20 synthetic_systems training entries spanning all 3 font styles (`bravura-compact`, `gootville-wide`, `leipzig-default`) × 2 sub-corpora (`synthetic_fullpage`, `synthetic_polyphonic`). For each: open the staff-crop image (cache path `data/processed/synthetic_v2/staff_crops/<font>/<sample_id>.png`), inspect the label `token_sequence` from `data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl`, compare against the visible content. Look for token-image mismatch, truncated content, mis-tokenized accidentals, degenerate labels. Aggregate per-font and per-sub-corpus quality stats. Output: per-sample inspection notes + a quality-summary table.

**Stream 2 — Reproduce synthetic generation pipeline.** Locate the synthetic generator (`src/data/generate_synthetic.py` or similar; pattern from existing manifests). Run it on a fresh sample of 10 source MusicXML files from `data/openscore_lieder/` (the source for synthetic_v2). Compare freshly-generated images and labels against the cached training data for the same source pieces. Look for non-determinism, silent verovio rendering failures, drift since the corpus was built. Output: did the generator drift? Does it match what's in the cache?

**Stream 3 — Sequence-length + complexity distributions.** Compute per-corpus stats over the full manifest:
- Token-sequence length percentiles: P50, P90, P95, P99, max
- Note-density per system (count of `^note-` tokens / count of `^<measure_start>` tokens)
- Multi-staff ratio (fraction of samples with >1 `<staff_start>` token)
- Cross-reference against `max_sequence_length: 512` in the v2 config — count how many samples per corpus exceed 512 tokens (silently truncated during training)

Output: per-corpus distribution table + truncation count.

**Stream 4 — Val-split structural audit.** Verify val splits per corpus. Already known:
- grandstaff_systems: 5340 val (5.5% of 102k)
- primus_systems: 4280 val (5.4% of 79k)
- cameraprimus_systems: 4280 val (5.4% of 79k)
- synthetic_systems: 0 val (0% of 20.5k)

For synthetic, determine the right unit to split by. Most likely: split by `source_path` (the source MusicXML file) so all crops derived from one source go to one split — avoids leakage between fonts/sub-corpora that share the same source piece. Document the recommended split methodology.

Also verify the other corpora's val splits are non-leaking. Spot-check for `source_path` overlap between train and val.

Output: split methodology recommendation + leakage spot-check result.

**Stream 5 — Capacity / model-size analysis.** Compute the current decoder's parameter count and per-layer breakdown:
- Decoder layers (`config.decoder_layers`, `config.decoder_heads`, `config.decoder_dim`)
- Vocabulary size and embedding dim
- Cross-attention and feedforward block sizes

Compare against a reference: the upstream Clarity-OMR DaViT decoder (the project diverged from this; weights publicly available on HuggingFace). Cross-reference with the Stream A finding: 11% token accuracy on synthetic_systems training data is suspicious for a transformer that "should" fit its training data — could be capacity, could be data corruption, could be optimization. Capacity check rules in or out.

Output: current vs reference parameter counts + recommendation: "current is N% of reference, recommend size bump to X" OR "current capacity is fine; bottleneck is elsewhere."

**Phase 1 deliverable.** Single report at `docs/audits/<DATE>-data-pipeline-and-capacity-audit.md` with five Stream sections + a synthesized findings table + a concrete Phase 2 fix proposal. The proposal lists every change to make (data, code, possibly architecture) with rationale + estimated effort.

**Phase 1 gate.** User reviews the report and approves the Phase 2 fix proposal before any code is written or training is started. If the audit recommends regenerating synthetic, the regeneration is escalated as a follow-up sub-project rather than bundled.

### Phase 2 — Implement fixes + retrain (~1 week wall-clock incl. 8h GPU)

Whatever Phase 1 recommends, plus the baseline fixes below.

**Baseline fixes (always applied):**

1. **Encoder freeze fix (Option B from the v3 retrain spec).** In `src/train/train.py` `_prepare_model_for_dora` at lines ~1327-1331:
   - Add `*, stage_config: Optional[StageTrainingConfig] = None` parameter
   - Derive `uses_cache = bool(stage_config and stage_config.cache_root and stage_config.cache_hash16)`
   - When `uses_cache`, keep encoder-side params frozen (`if uses_cache and "encoder" in name: continue` before the lora-activation logic)
   - Update the call site at line ~2181 to pass `stage_config=stage`
   - Add pre-flight assertion immediately after: count trainable encoder params and `raise RuntimeError` if non-zero when `uses_cache`
   - Add regression tests in `tests/train/test_freeze_encoder.py` (3 tests: cache freezes, no-cache unfreezes, decoder always trainable)

2. **Val split for synthetic_systems.** Implement the split methodology recommended by Stream 4 (likely by-source-piece). Modify the manifest writer/builder for synthetic_v2 (likely in `src/data/generate_synthetic.py` or a new `scripts/data/add_val_split_synthetic.py`). Verify no source_path overlap between train and val after the split. Target ~5% val (~1000 entries from 20,583).

3. **dataset_mix rebalancing.** New weights from Phase 1 Stream 1+3 findings. Most likely outcome: synthetic at ~10-30%, grandstaff at ~40-50%, primus at ~10-20%. Update `configs/train_stage3_radio_systems_v3.yaml`. (If Phase 1 recommends dropping synthetic entirely, the dataset_mix omits it.)

**Conditional fixes (only if Phase 1 recommends):**

4. `max_sequence_length` bump from 512 to 1024 or 2048. Affects memory; verify the cached encoder feature shapes don't depend on this directly.
5. Decoder size bump per Stream 5 recommendation. Affects checkpoint shape; cannot resume from Stage 2 v2 weights if the dim differs — would require starting from random init (or a re-fit Stage 2 with the new size, which is out of scope).

**New v3 config file.** `configs/train_stage3_radio_systems_v3.yaml`:
- Copy v2 as a starting point
- Update header docstring to describe supersession of v2 and reference this spec
- Apply baseline fixes (new dataset_mix; output dir `checkpoints/full_radio_stage3_v3/`; step log `logs/full_radio_stage3_v3_steps.jsonl`)
- Apply conditional fixes if any
- `effective_samples_per_epoch: 9000` (carried from prior v3 spec; v2 was still descending at step 6000)

**Resume verification smoke test (~30 min before main run).** Start the run, stop at step 100, restart with `--resume-checkpoint`, verify global_step continues from 101 and loss curve looks continuous. Catches resume regression before the 8h main run is exposed to it.

**Main retrain run (~8h on seder).** Standard launch command, same shape as v2's. Monitor `val_loss_per_dataset` for all 4 corpora — confirm synthetic now appears (it didn't in v2) and that all corpora descend.

**Phase 2 deliverable.** `checkpoints/full_radio_stage3_v3/_best.pt` on seder, with the freeze fix verified by both the pre-flight assertion (catches at training start) and the encoder-LoRA-keys-diff check (catches at training end — should show all encoder LoRA keys unchanged from Stage 2 v2's checkpoint).

### Phase 3 — Re-evaluate + verdict (~1 day)

Four eval passes against `checkpoints/full_radio_stage3_v3/_best.pt`:

1. **A2 encoder parity** (`scripts/audit/a2_encoder_parity.py`). Expected PASS (max_abs_diff << 0.01 — encoder is now actually frozen).
2. **A3 decoder-on-training** (`scripts/audit/a3_decoder_on_training.py`). The most direct test of whether the data fix took. Expected: per-corpus token accuracy ≥ 80% across all 4 corpora. If synthetic is still < 50%, the fix didn't take and we need to revisit Phase 1 conclusions.
3. **4-piece HF demo eval** (`eval/run_clarity_demo_radio_eval.py --name stage3_v3_best`). Compare per-piece + mean onset_f1 against:
   - v2 baseline: mean 0.0589
   - Frankenstein baseline: mean 0.0599
4. **50-piece lieder ship-gate** (Subproject 4 eval). Compare against v2 corpus mean 0.0819.

**Decision matrix:**

| A2 | A3 multi-staff accuracy | Demo mean onset_f1 | Verdict |
|---|---|---|---|
| PASS | ≥ 0.80 | ≥ 0.241 | **SHIP.** Foundation + data + (capacity?) were the only issues. Close the RADIO-viability chapter. |
| PASS | ≥ 0.80 | 0.10 - 0.241 | Foundation + data sound, residual gap is architecture/capacity. Trigger DaViT baseline + architecture sub-project. |
| PASS | < 0.80 | any | Data fix didn't take. Investigate which corpus regressed; revisit Phase 1 conclusions. May require regenerating synthetic. |
| FAIL | any | any | Encoder freeze didn't stick despite the pre-flight. Debug Phase 2 implementation. |

**Phase 3 deliverable.** `docs/audits/<DATE>-stage3-v3-data-rebalance-results.md` with all four eval results, the A3 retest table, and the verdict.

## File structure

**New files (single feature branch, off `main` at `b2ed4a2`):**

| Path | Phase | Purpose |
|---|---|---|
| `docs/audits/<DATE>-data-pipeline-and-capacity-audit.md` | 1 | Audit report |
| `scripts/audit/inspect_synthetic_samples.py` | 1 | Stream 1 sample inspection tool |
| `scripts/audit/reproduce_synthetic_generation.py` | 1 | Stream 2 generator-reproduction check |
| `scripts/audit/corpus_distributions.py` | 1 | Stream 3 length/complexity stats |
| `scripts/audit/val_split_audit.py` | 1 | Stream 4 split-leakage check |
| `scripts/audit/decoder_capacity_analysis.py` | 1 | Stream 5 param-count analysis |
| `scripts/data/add_val_split_synthetic.py` | 2 | Carve val split from synthetic |
| `configs/train_stage3_radio_systems_v3.yaml` | 2 | New training config |
| `tests/train/test_freeze_encoder.py` | 2 | Regression tests (from v3 retrain spec) |
| `docs/audits/<DATE>-stage3-v3-data-rebalance-results.md` | 3 | Final results report |

**Modified files:**
- `src/train/train.py` — encoder freeze fix in `_prepare_model_for_dora` + pre-flight assertion at the call site (from v3 retrain spec, never landed)
- `data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl` — split column updated by add_val_split_synthetic.py (or new manifest file written alongside)

**Generated artifacts (not in git, on seder):**
- `audit_results/synthetic_inspection.json`, `corpus_distributions.json`, etc. — Phase 1 outputs
- `checkpoints/full_radio_stage3_v3/*.pt` — Phase 2 retrain output
- `eval/results/clarity_demo_stage3_v3_best/*`, `eval/results/lieder_stage3_v3/*` — Phase 3 eval output

## Risks and decisions made

| Risk | Mitigation |
|---|---|
| Phase 1 audit may surface that synthetic is broken AND the corpus needs regeneration. That's a big addition to scope. | Spec excludes regeneration. If audit recommends it, regeneration becomes a separate sub-project (likely 1-2 week additional). User decides at the Phase 1 gate. |
| Stream 5 capacity analysis may recommend a decoder size bump, which would invalidate the ability to resume from Stage 2 v2 weights. | Documented in Phase 2: a size bump means starting from random decoder init. If audit recommends a size that requires fresh init, user can decide whether that's worth the additional compute or to defer. |
| Synthetic val split methodology may need to handle the `synthetic_polyphonic` vs `synthetic_fullpage` sub-corpora carefully — they share source pieces but produce different label sequences. | Stream 4's job is to figure this out. The split should respect sub-corpus identity if the labels differ structurally. |
| Phase 2 fixes may inadvertently break the existing inference pipeline (e.g., loader changes). | A2 + smoke demo eval against the *v3* checkpoint at Phase 3 catches this. No regression test against v2 baseline needed because we're not modifying v2's files. |
| The v3 retrain plan from earlier had tasks 4-9 that didn't execute (Phase 1 gate halted). Those tasks are superseded by this spec but the plan file is still in the repo. | The handoff doc explicitly tells the new session: ignore `2026-05-11-stage3-v3-retrain-plan.md` Tasks 4-9; this spec replaces them. |

## Effort estimate

| Phase | Wall-clock |
|---|---|
| Phase 1 audit + report | 2-3 days |
| Phase 2 implementation + retrain | ~1 week (including 8h GPU training) |
| Phase 3 re-eval + verdict | 1 day |
| **Total** | **1.5-2 weeks** |

## What follows this sub-project

The Phase 3 verdict steers exactly one of:

- **SHIP** (A2 PASS, A3 ≥ 0.80, demo ≥ 0.241): foundation question is answered yes. Close RADIO-viability chapter. Decide on deployment, scope-down, or further productionization.
- **Architecture / capacity sub-project** (A2 PASS, A3 ≥ 0.80, demo 0.10-0.241): data is sound but residual gap remains. Triggers DaViT baseline reproduction + architecture investigation.
- **Re-audit / regenerate synthetic** (A3 < 0.80): the data fix didn't take. Most likely regenerate synthetic, possibly with broader changes. Becomes the next sub-project.
- **Debug freeze regression** (A2 FAIL): unexpected; trainer-construction bug. Small fix sub-project.

All four branches preserve the Phase 1 audit scripts and Phase 2 freeze fix in tree as durable improvements.

## Handoff to fresh session

A separate handoff document at `archive/handoffs/2026-05-11-session-end-handoff.md` provides:
- One-paragraph project status with PR hashes
- Pointer to this spec + the upcoming plan
- Memory entries to read first
- Starting branch / SHA
- Explicit "ignore these" pointers for stale artifacts from this session
- A "first 10 minutes" checklist so the fresh session can verify context before diving in

The fresh session's first action: read the handoff doc, then this spec, then write the implementation plan via the writing-plans skill.
