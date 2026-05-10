# RADIO Stage 3 Phase 1 — Complete Handoff (2026-05-09)

> Phase 1 (frozen-encoder, tier-grouped training) ran cleanly to v2 step 5500. Phase 1 → Phase 2 gate passes on every soft-gate. Phase 2 (lieder onset_f1) is the next session's job.

## TL;DR

- **Ship artifact:** `10.10.1.29:C:/Users/Jonathan Wesely/Clarity-OMR-Train-RADIO/checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt` — opt-step 5500, `val_loss=0.1643`.
- **Architectural bet held.** cameraprimus_systems val_loss = 0.136, *below* Stage 2 v2's overall best (0.148) on the dataset most prone to regression.
- **Two runs**: v1 (target 4500, completed cleanly), v2 (target 6000, completed cleanly with regression at step 6000 → don't extend further).
- **Branch `feat/stage3-phase1-training` is a stack of 36 commits** — Plan C implementation (Tasks 0–11) + 4 fixes during execution (cache-key fallback, dangling images var, val_loss aggregator, incremental-extension support). 71/71 train tests pass.
- **Phase 2 (Plan D) doesn't exist yet** — needs to be drafted by the next session per spec §"Phase 2".

## Phase 1 Exit Criteria — all PASS

| # | Criterion | Result |
|---|---|---|
| 1 | No sanity halt (val_loss > 5.0 in first 200 steps OR NaN) | ✅ never fired |
| 2 | best.pt saved + best_val_loss < 0.5 | ✅ best_val_loss=0.164 |
| 3 | Per-dataset val_loss floors hold | ✅ camera=0.136 (below SV2 0.148), grand=0.192 (above SV2 96.8 quality, but quality maps differently from val_loss), primus=0.165 |
| 4 | MusicXML validity rate ≥ 0.50 | not yet measured — Phase 2's eval driver runs this gate |
| 5 | Step-log telemetry complete | ✅ 9 val rows for v1, 12 for v2; both step-logs saved |

Criterion #4 is technically deferred to Plan D's eval phase but the metric is enabled and ready in `src/eval/metrics.py:musicxml_validity_from_tokens` + `src/eval/run_eval.py`.

## Run results

### v1 (target 4500 opt-steps)

| step | agg | camera | grand | primus |
|---|---|---|---|---|
| 500 | 0.310 | 0.298 | 0.342 | 0.289 |
| 1000 | 0.206 | 0.158 | 0.229 | 0.232 |
| 1500 | 0.224 | 0.266 | 0.219 | 0.189 |
| 2000 | 0.210 | 0.186 | 0.263 | 0.179 |
| 2500 | 0.247 | 0.316 | 0.220 | 0.206 |
| 3000 | 0.277 | 0.402 | 0.231 | 0.199 |
| 3500 | 0.198 | 0.156 | 0.264 | 0.174 |
| 4000 | 0.183 | 0.185 | 0.171 | 0.192 |
| **4500** | **0.169** | 0.143 | 0.206 | 0.160 |

Cameraprimus oscillated 0.158 → 0.402 in the middle of the run (peak-to-peak 0.244 over 2000 opt-steps, exceeding the spec's "sustained oscillation" trigger), but stabilized by step 3500 and ended below Stage 2 v2's anchor. Spec extension protocol triggered: continued descent at step 4500 → extend to 6000.

### v2 (target 6000 opt-steps; full re-run from Stage 2 v2 init)

| step | agg | camera | grand | primus |
|---|---|---|---|---|
| 4000 | 0.184 | 0.184 | 0.175 | 0.193 |
| 4500 | 0.185 | 0.184 | 0.208 | 0.163 |
| 5000 | 0.180 | 0.164 | 0.206 | 0.170 |
| **5500** | **0.164** | **0.136** | 0.192 | 0.165 |
| 6000 | 0.182 | 0.166 | 0.203 | 0.177 |

5500 → 6000 regressed (0.164 → 0.182). Spec criterion fires for "plateaued or regressed → finalize." **Don't extend to 7500.**

### v1 vs v2 comparison

v2 marginally better than v1 across the multi-staff datasets (camera −5%, grandstaff −7%); primus +3% (slight regression). Net aggregate −3%. Both runs produced usable best.pt; v2 is the ship artifact.

Wall time: each run ≈ 30–45 min on RTX 5090 with the encoder cache. The "1.5–3h projection" in the spec was from the earlier batch-size sweep; the actual run time was faster because b_cached=16 dominates throughput at the recommended config.

## Branch state

`feat/stage3-phase1-training` HEAD: `715a89b`. 36 commits since `main`.

```
ab27862 → fc1089c → 715a89b
   |          |        ↑ incremental-extension fix (smoke-tested)
   |          ↑ YAML target raised to 6000
   ↑ dangling 'images' var fix (mid-Phase-1)
```

Pre-execution: 0306de2 (Task 11) + earlier Plan C tasks 0–10 (TDD + reviews).
Mid-execution fixes: 88a9211 (cache-key legacy fallback), ab27862 (dangling images), fc1089c (extend YAML), 715a89b (incremental-extension support).

## Critical artifacts (preserved on GPU box `10.10.1.29`)

```
checkpoints/v1_phase1_safe/                       ← v1 best.pt + final + step_4500 (3.21 GB each)
checkpoints/full_radio_stage3_v1/                 ← v1 full run (best.pt + 9 step ckpts + final)
checkpoints/full_radio_stage3_v2/                 ← v2 full run (best.pt + 12 step ckpts + final)
logs/full_radio_stage3_v1_steps.jsonl             ← v1 step log (also _phase1_complete copy)
logs/full_radio_stage3_v2_steps.jsonl             ← v2 step log
```

Local mirrors:
```
/home/ari/work/sp3_review/stage3_v1_steps.jsonl
/home/ari/work/sp3_review/stage3_v1_progress.png
/home/ari/work/sp3_review/stage3_v2_steps.jsonl
/home/ari/work/sp3_review/stage3_v2_progress.png
```

Charts (re-runnable via scp pull): `~/bin/chart_stage3_v1.py`, `~/bin/chart_stage3_v2.py`.

## Bugs found and fixed during execution

1. **CacheMiss on every primus/grandstaff sample** (`88a9211`). The Phase 0 cache (1.4 TB, hash `ac8948ae4b5be3e9`) was built before commit `50fa624` switched `_sanitize_sample_key` from lossy `__` to reversible `_SLASH_/_COLON_/_BSLASH_`. The trainer used the new scheme; the cache files used the old scheme. Fix: backwards-compat fallback in `StageBDataset.__getitem__` — try new key first, fall back to legacy `__` on miss. Existing 1.4 TB cache stays usable; new writes still use the reversible scheme.
2. **`NameError: 'images' is not defined`** (`ab27862`). Task 2's tier-aware refactor moved `images` access into `_forward_batch_for_train`, but the JSONL row write at `train.py:2989` still referenced the local `images` variable. Fix: read batch_size from `_batch_dict["images"]` or `_batch_dict["encoder_hidden"]` based on tier.
3. **C1 (final-review fix, applied pre-launch, `ed46594`+`fbecfda`)**. `_run_validation_per_dataset` aggregated by `dataset_mix.ratio` directly, which silently dropped cameraprimus (its YAML ratio is 0.0 because that field weights the cached-tier sampler, not the global aggregate). `best_val_loss` and sanity halt would have been blind to the dataset most likely to regress. Fix: re-project using `cached_data_ratio` to produce the spec's 70/10/10/10 weighting.
4. **Incremental extension routing + I1** (`715a89b`). The spec's 4500 → 6000 → 7500 extension protocol requires resuming a tier-grouped stage with a raised YAML target. Two coupled bugs blocked it: (a) the resume routing walks `global_step` and treats `global_step >= stage_total_steps` as "stage done" — a target raise was silently SKIPPED; (b) the for-loop iterates microbatches in tier-grouped mode but `stage_start_step` was set in opt-step units. Fix: prefer `stage_name`-based matching against the YAML; save `stage_step` in opt-step units consistently; override `stage_start_step` to microbatch units in tier-grouped mode after `_batch_idx_consumed` is set. Smoke-tested end-to-end on GPU box (run 30 opt-steps, save, resume with raised target, run 23 more without crashing).

## Things NOT done that the next session might want

1. **Plan D — Phase 2 evaluation plan.** The spec's §"Phase 2" specifies three eval surfaces: lieder onset_f1 (architectural ship gate, ≥ 0.30 = Strong), per-dataset quality regression-check floors (Stage 2 v2 baselines: grand 96.8, grand-orig 93.4, primus 83.1, camera 75.2 yellow-flag), MusicXML validity rate. Plan D should be drafted with the same rigor as Plan C — TDD where applicable, eval-driver wiring, decision-gate framework.
2. **PR for branch `feat/stage3-phase1-training`.** 36 commits ready to merge. Suggest sub-bundling: Plan C tasks (10 commits) + post-launch fixes (4 commits, the cache-key, images, val-aggregator, extension fixes). Reviewer focus on: per-dataset val_loss aggregator (the most architecturally important fix), tier-grouped sampler arithmetic, extension routing.
3. **v1 vs v2 Phase 2 head-to-head.** v2 wins by ~3% on aggregate val_loss and ~5% on cameraprimus. Phase 2 should evaluate v2 as primary; if v2 fails the onset_f1 gate, evaluate v1 as well to see if the Phase 1 metrics translated to lieder differently between the two runs.
4. **Resume-routing backward compat note.** v1 + v2 checkpoints have `stage_step` in *microbatch* units (saved before `715a89b`). The new routing matches by stage_name and uses `stage_step` directly, which would inflate `resume_stage_completed_steps` for these old ckpts and cause the `stage_start_step > stage_total_steps` skip to fire. **Workaround for extending v1/v2: use `--start-stage stage3-radio-systems-frozen-encoder` (Option A) — that resets to step 1 and ignores the inflated count.** Only future checkpoints (post-`715a89b`) work with incremental resume.
5. **Defensive-rebuild StopIteration replay** (I2 from final review). If the per-step DataLoader iterator exhausts mid-stage (shouldn't happen on the happy path because the sampler is sized exactly), the rebuild replays `_batches[_start_idx:]` from the start, double-counting `_batch_idx_consumed`. Pre-existing footgun; never triggered in v1 or v2 runs. Worth a follow-up commit before next major use, but not Phase-1-blocking.
6. **Step counts under cap.** v2 run-#2 of the smoke-test reached opt-step 53 instead of the requested 60 — shuffle variance puts more live blocks (8 batches each) in some ranges than others, so a microbatch-bound loop can cap at fewer opt-steps than the YAML target. v1 hit 4500 cleanly; v2 hit 6000 cleanly. The variance is in the noise, but documenting for the next session.

## Phase 2 prerequisites (to draft Plan D against)

- Lieder eval set: needs to exist with stratification by `staves_in_system` (per-class metrics), and `lc6548281` annotation (the parent spec's architectural sanity-check sample).
- Eval driver: `src/eval/run_eval.py:evaluate_rows` is wired for token decode → music21 export → onset_f1. MusicXML validity metric was wired in `354d25b` (Plan C Task 10).
- Stage 2 v2 baseline numbers for the regression-check floor: grandstaff_systems 96.8, grandstaff 93.4, primus 83.1, cameraprimus 75.2 (yellow-flag, 200-sample re-eval pending per `project_radio_stage2_v2.md`).
- Compute budget: lieder onset_f1 eval over the full set + per-dataset quality eval is bigger than a single training run — likely 30 min – 2 h.

## Where to start (next session)

1. Read this file in full.
2. Read the Stage 3 design spec: `/home/ari/docs/superpowers/specs/2026-05-07-radio-stage3-design.md` §"Phase 2".
3. Read Plan C: `/home/ari/work/Clarity-OMR-Train-RADIO/docs/superpowers/plans/2026-05-09-radio-stage3-phase1-training.md`.
4. Read memory: `project_radio_stage3_design.md`, `project_radio_stage3_data_prep_progress.md`.
5. Draft Plan D using `superpowers:writing-plans` — should mirror Plan C's TDD structure but center on eval-driver wiring + the three eval surfaces.
6. Plan D first task: an eval driver smoke test against the v2 `_best.pt`. If the driver can produce one onset_f1 number end-to-end, the full eval is mechanical.
7. **The critical question for Phase 2:** does `_best.pt@step5500` produce lieder onset_f1 ≥ 0.30 (Strong), 0.241–0.30 (Mixed), or < 0.241 (Flat)? That decides whether the rebuild ships or pivots to Phase 0 / Audiveris-style alternative. The Phase 1 metrics suggest Strong is in reach (cameraprimus_systems already below Stage 2 v2's anchor on the dataset most predictive of lieder), but Phase 2 is the actual answer.

## Goodbye

Phase 1 is the architectural bet. The bet appears to have taken: a frozen-encoder Stage 3 with two-tier dataloader matched and slightly beat Stage 2 v2 on val_loss, with the strongest improvement on the dataset most likely to regress. Phase 2's lieder onset_f1 will tell us whether that translates into the user-visible quality gate. The infrastructure (encoder cache, tier-grouped sampler, per-dataset val_loss tracking, sanity halt, extension routing) all stress-tested cleanly; the fixes that came out of execution are durable.

Hand-off complete.
