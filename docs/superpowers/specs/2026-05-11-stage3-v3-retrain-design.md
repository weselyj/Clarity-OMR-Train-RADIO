# Stage 3 v3 — Encoder-Freeze Fix + Retrain

**Status:** design approved 2026-05-11 — ready for implementation plan
**Branch off:** `main` at `c8e8dc8` (post-merge of PR #47 and PR #48)
**Predecessor:** [`docs/audits/2026-05-11-stage3-v2-training-audit.md`](../../audits/2026-05-11-stage3-v2-training-audit.md) — diagnosed encoder DoRA not frozen during Stage 3 v2

## Goal

Resolve the encoder-DoRA-not-frozen bug identified in the audit and produce a Stage 3 v3 checkpoint whose A2 (encoder output parity) passes and whose downstream metrics meet or exceed the project's 0.241 decision gate on the 4-piece HF demo. Get a definitive answer on whether the RADIO architecture works when trained correctly.

## Non-goals

- **Not closing the decision gate by other means.** This sub-project tests one hypothesis (foundation was broken; fix unlocks the architecture). It does NOT explore alternative architectures, alternative training data, or alternative loss functions.
- **Not running A3 / Phase B from the audit.** Those become relevant only if A2 passes on v3 AND downstream metrics still disappoint — separate sub-project.
- **Not running DaViT baseline or Audiveris benchmark.** Independent parallel threads; out of scope for this work.
- **Not refactoring the trainer beyond the freeze fix.** Audit was scoped to Stage 3; the freeze bug is isolated. Other trainer tuning (per PyTorch perf guide) is a separate sub-project if it ever becomes warranted.
- **Not retraining Stage 1 or Stage 2.** The bug is in Stage 3 only; upstream stages are downstream of nothing.

## Scope (single sub-project, 4 phases with explicit gates)

| Phase | Effort | Gate to next phase |
|---|---|---|
| 1. Diagnostic (frankenstein checkpoint + 4-piece demo eval) | ~30 min compute + ~30 min code | Demo mean onset_f1 thresholds (see Phase 1 §) |
| 2. Fix code (encoder-side freeze guard via cache-derived auto-detect, Option B) | a few hours | Regression tests pass + defensive assertion in place |
| 3. Retrain Stage 3 v3 (9000 steps from Stage 2 v2 init) | ~8h on seder | Training completes + final A2 PASS on `_best.pt` |
| 4. Re-evaluation (A2 + 4-piece demo + 50-piece lieder) | ~30 min compute | Decision matrix below |

Each phase ends with an explicit go/no-go decision. The phases are sequential — Phase 2 doesn't start until Phase 1 passes its gate; Phase 3 doesn't start until Phase 2 lands.

## Architecture

### Phase 1 — Diagnostic (frankenstein checkpoint)

**Hypothesis being tested:** if the decoder were fed encoder features from the encoder it was *trained against* (Stage 2 v2's encoder, which produced the cache), demo onset_f1 should rise meaningfully. If it doesn't, there are failure modes beyond encoder drift and committing to a retrain is premature.

**Method:** new one-off script `scripts/audit/build_frankenstein_checkpoint.py` that:
1. Loads `checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt`
2. Loads `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt`
3. Builds a merged state_dict: encoder params from Stage 2 v2, everything else from Stage 3 v2
4. Saves to `checkpoints/_frankenstein_s2enc_s3dec.pt`
5. Logs key counts: how many encoder keys taken from S2v2, how many remaining keys taken from S3v2, any keys missing or extra in either source (mismatched keys are an experimental confound to investigate before relying on the result)

Then run `eval/run_clarity_demo_radio_eval.py --stage-b-ckpt checkpoints/_frankenstein_s2enc_s3dec.pt --name frankenstein_s2enc_s3dec`. No code changes to the eval driver.

**Gate decision:**

| Frankenstein mean onset_f1 | Decision |
|---|---|
| **≥ 0.15** | Diagnosis confirmed. Proceed to Phase 2. |
| **0.10 – 0.15** | Partial signal. Pause and discuss with user before launching the 8h retrain. |
| **< 0.10** | Diagnosis incomplete. Do NOT retrain yet. Open a follow-up investigation sub-project. |

Rationale: v2 demo mean is 0.0589; gate is 0.241; halfway is ~0.15. A meaningful-but-incomplete jump (halfway+) suggests the retrain will close the rest. No meaningful jump means encoder drift was not the dominant failure mode.

### Phase 2 — Fix (Option B: cache-derived freeze)

**Single load-bearing code change** in `_prepare_model_for_dora` at [`src/train/train.py:1327-1331`](../../src/train/train.py#L1327-L1331). The function gains awareness of whether the stage config uses cached encoder features. If yes, all parameters whose name contains `encoder` stay frozen — even if their name also contains `lora_`. Rationale for auto-detection rather than an explicit kwarg: cache use and encoder-update are physically incompatible (the cache becomes stale after the first encoder gradient step), so coupling them in code eliminates the foot-gun that produced this bug. The narrow "live encoder + frozen encoder" case (which auto-detect doesn't support) isn't a use case we've ever needed; if it becomes needed later, an explicit override kwarg can be added then.

**Caller change:** wherever `_prepare_model_for_dora` is invoked from the train loop, pass the stage config so the function can read `cache_root` and `cache_hash16`. Keep the function signature minimal — pass the existing `StageTrainingConfig` object rather than adding multiple new kwargs.

**Two regression tests** in new file `tests/train/test_freeze_encoder.py` (CUDA-gated):
1. `test_cache_config_freezes_encoder` — build a Stage B model, run `_prepare_model_for_dora` with a stage config that has `cache_root` + `cache_hash16` set, assert no encoder param has `requires_grad=True`.
2. `test_no_cache_unfreezes_encoder` — same model, stage config WITHOUT cache, assert encoder LoRA params DO get `requires_grad=True`. This proves the gate is doing real work.

**Defensive runtime assertion** added near the existing model-prep call site: log `trainable encoder params = <N>` and assert `N == 0` when `uses_cache`. Fails loudly within seconds of training start rather than after 5.6h of wasted compute.

**Out of scope for the fix:** updating the checkpoint filename or config docstring to remove the now-redundant "frozen-encoder" phrasing. The freeze is now structural, so the documentation is no longer load-bearing. Leave it as-is and let a future cleanup pass handle it.

### Phase 3 — Retrain Stage 3 v3

**Config:** new file `configs/train_stage3_radio_systems_v3.yaml`, copied from `_v2` with the following changes:
- `effective_samples_per_epoch: 9000` (up from 6000 — v2 was still descending at step 5500/6000; cached training is cheap enough to absorb the extra 3000 steps as insurance)
- Output dir: `checkpoints/full_radio_stage3_v3/`
- Step log: `logs/full_radio_stage3_v3_steps.jsonl`
- Same `cache_hash16: ac8948ae4b5be3e9` (the existing cache, Stage 2 v2 derived per Phase 0)
- Same data manifest (`src/data/manifests/token_manifest_stage3.jsonl`)
- Same optimizer/schedule (only training-behavior change is the freeze fix itself)
- Update docstring at top: "Retrained with encoder freeze enforced via Option B (cache-derived auto-detect). Supersedes v2; v2's `_best.pt` had encoder DoRA unfrozen due to a missing freeze in `_prepare_model_for_dora`."

**Resume verification (your concern from the conversation):** before launching the 9000-step run, smoke-test the resume code path with a deliberate stop+resume cycle. ~30 minutes of insurance. If resume is broken, the 9000-step run is exposed to mid-flight termination (network, GPU hang, OS update). Sequence:
1. Start the v3 training run with the v3 config
2. Stop it cleanly after step 100 (Ctrl-C through SSH or kill on seder)
3. Restart with `--resume-checkpoint`
4. Confirm: same per-step train_loss curve from step 100 onward as if we'd never stopped
5. Continue the run to completion

**Pre-flight assertion (audit-recommended):** added in Phase 2, but worth restating: immediately after `_prepare_model_for_dora`, log and assert:
```
trainable encoder params: 0   (sentinel — would have been 384 in v2)
trainable decoder params: <N>
```
Hard failure if encoder count > 0 when `uses_cache=True`.

**Checkpoint retention strategy:** keep `_step_NNNNNNN.pt` files at every 500-step boundary (matches v2 behavior). This is a belt-and-suspenders backup against `_best.pt` tracking the wrong step (see "Best-checkpoint accounting" below).

**Best-checkpoint accounting:** the v2 step log shows lowest val_loss at step 5500 (0.164), but the audit's metadata read claimed step 4000 / val_loss 0.148. Before launching v3, spot-check what `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt`'s internal metadata reports and reconcile with the step log. If `_best.pt` actually points at the wrong step's weights, that's a separate small bug worth fixing in the trainer's checkpoint-save logic before v3 launches. If it's only the metadata that's stale (the weights match step 5500), it's cosmetic. Resolve this before v3 so v3's `_best.pt` is trustworthy.

**Step-count rationale:** with the encoder frozen, cached features stay valid the entire run. Each step's gradient signal is consistent (unlike v2, where the encoder kept drifting away from the cache). Expect convergence to be at least as deep as v2's best (val 0.164 @ step 5500), and possibly meaningfully better. Allow up to 9000 steps to let it find a real plateau rather than be artificially cut off.

### Phase 4 — Re-evaluation

Three eval passes against `checkpoints/full_radio_stage3_v3/_best.pt`:

1. **A2 (encoder parity)** — re-run `scripts/audit/a2_encoder_parity.py` with the v3 checkpoint and the same cache. Expected: PASS (max_abs_diff well below 0.01 — the encoder didn't move). If FAIL, the freeze didn't actually stick; debug before doing anything else.
2. **4-piece HF demo eval** — `eval/run_clarity_demo_radio_eval.py --name stage3_v3_best`. Compare per-piece + mean onset_f1 against v2's post-first-emission-patch baseline (mean 0.0589).
3. **50-piece lieder ship-gate** — Subproject 4 eval. Compare mean onset_f1 against v2's corpus mean (0.0819).

**Decision matrix:**

| A2 | Demo mean onset_f1 | Verdict |
|---|---|---|
| PASS | ≥ 0.241 | **SHIP.** RADIO is viable when trained correctly. Close the audit chapter. |
| PASS | 0.10 – 0.241 | Foundation now sound; remaining gap is architecture/data. Next: A3 + Phase B from the audit, or pivot to DaViT baseline. |
| PASS | < 0.10 | Foundation sound but model doesn't generalize. Urgent: architecture investigation. |
| FAIL | — | Freeze didn't stick. Debug Phase 2 implementation. |

**Reporting:** write `docs/audits/2026-05-11-stage3-v3-retrain-results.md` documenting the three eval numbers, the A2 result, and the chosen verdict. Reference back to the audit so future-you can trace the full diagnosis-fix-validate arc in one place.

## File structure

**New files:**
- `scripts/audit/build_frankenstein_checkpoint.py` — Phase 1 merge script
- `tests/train/test_freeze_encoder.py` — Phase 2 regression tests
- `configs/train_stage3_radio_systems_v3.yaml` — Phase 3 config
- `docs/audits/2026-05-11-stage3-v3-retrain-results.md` — Phase 4 report

**Modified files:**
- `src/train/train.py` — Phase 2 fix in `_prepare_model_for_dora` + pre-flight assertion at the call site

**Generated artifacts (not in git):**
- `checkpoints/_frankenstein_s2enc_s3dec.pt` — Phase 1 diagnostic checkpoint
- `checkpoints/full_radio_stage3_v3/*.pt` — Phase 3 training output (including `_best.pt` and `_step_NNNNNNN.pt` series)
- `logs/full_radio_stage3_v3_steps.jsonl` — Phase 3 step log
- `eval/results/clarity_demo_stage3_v3_best/*` — Phase 4 demo eval output
- `eval/results/lieder_stage3_v3/*` — Phase 4 lieder eval output
- `audit_results/a2_encoder_stage3_v3.json` — Phase 4 A2 re-run output

## Risks and decisions made

| Risk | Mitigation |
|---|---|
| Frankenstein merge has mismatched keys, invalidating Phase 1's signal | Merge script logs per-source key counts and missing/extra keys. If >1% of keys mismatch, escalate before drawing a conclusion. |
| Phase 1 returns ambiguous result (0.10-0.15) | Discuss with user before launching the 8h retrain. Don't auto-proceed. |
| Phase 2 fix doesn't stick (freeze still leaky) | Phase 3 pre-flight assertion catches at step 0. Regression test catches at PR time. Final A2 in Phase 4 catches if everything else missed. |
| Phase 3 training crashes mid-run after step ~5000 | Resume verification step (30 min before main run) confirms recovery works. `_step_NNNNNNN.pt` checkpoints at 500-step intervals make any restart point recoverable. |
| `_best.pt` accounting bug carries over from v2 | Before v3 launch, spot-check v2's `_best.pt` metadata vs. step log and reconcile. Keep per-step checkpoints so best can be reconstructed from re-evaluating each. |
| 9000 steps may overfit on the cached data and val_loss starts climbing late | We're capturing `_best.pt` regardless of where it occurs in the run. Final-step checkpoint may not be the best but that's OK — we ship the best. |
| Phase 4 A2 PASSES but demo eval still bad | Decision matrix branch "PASS / < 0.241" handles this — triggers a separate next-sub-project, doesn't block closing this one. |
| Phase 4 A2 FAILS unexpectedly | Same model code paths are being exercised as Phase 2's regression tests, so this would mean the runtime model construction differs from the test setup somehow. Debug Phase 2 first. |

## Effort estimate

| Phase | Calendar time |
|---|---|
| Phase 1 (diagnostic) | 1-2 hours |
| Phase 2 (fix code + tests + assertion) | 2-4 hours |
| Phase 3 (retrain + resume verification) | ~9 hours (8h training + 1h setup/verification) |
| Phase 4 (re-eval + report) | ~1 hour compute + 1-2 hours report writeup |
| **Total** | **roughly half a day of human work + 8h of GPU time** |

If Phase 1 gates fail, Phase 3's GPU time is saved. Worst case is ~3h human work to confirm an architectural problem before spending the GPU budget.

## What follows this sub-project

If Phase 4 verdicts **SHIP**: project's foundational question is answered. Close out RADIO-viability chapter, decide what's next (deploy, scope down, etc.).

If Phase 4 verdicts in the **0.10 – 0.241** band: foundation sound but architecture/data limit. Next sub-project options:
- Run A3 + Phase B audit to look for secondary training-loop bugs
- DaViT baseline reproduction (parallel-thread option from the audit)
- Architecture/data experiments

If Phase 4 verdicts **< 0.10**: architecture is the bottleneck. Next sub-project should be architecture-focused (DaViT comparison, alternative backbone, etc.).

In all branches, this sub-project's deliverables (the freeze fix, regression tests, the runtime assertion, the audit infrastructure scripts) stay in tree as durable improvements.
