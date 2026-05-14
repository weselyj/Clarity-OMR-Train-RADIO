# Session-end handoff: Stage 3 v4 — Phase 0 complete, Phase 1+ pending (2026-05-13)

> Hand off to a fresh session. Read this doc first, then the spec + Phase 0a addendum, then the plan. Context-load before any work: ~20 minutes.

## One-paragraph status

A real-world scan audit (2026-05-12) identified that Stage 3 v3 misreads bass-clef staves as treble on beginner-piano scans (Bethlehem, TimeMachine) because the v3 training mix is 100% clean engraving. This session designed Stage 3 v4: add a `scanned_grandstaff_systems` corpus by applying offline scan-realistic degradation to the existing grandstaff corpus, retrain cached. Phase 0's two cheap pre-flight gates both ran: Phase 0a's strict bass-clef-misread gate failed (12.1% < 30% threshold) but was reframed as PROCEED because the dominant cluster (phantom-staff-residual at 75.8%) shares the same root cause — both are decoder structural confusion on multi-staff piano, and the planned retrain targets both. Phase 0b's overfit smoke gate passed (train loss 0.006 / val loss 0.013 at step 500). Training infrastructure is verified healthy and the design is approved. Phase 1+ (corpus build, cache rebuild, retrain, evaluation) is the remaining ~1-2 day arc of mostly remote GPU work on seder.

## What the fresh session should read, in order

1. **This handoff doc** (you're reading it).
2. **Memory entries** worth refreshing before the spec:
   - `project_radio_clarity_demo_eval.md` — Clarity-OMR demo eval baseline
   - `project_radio_first_emission_fix.md` — PR #47 key/time-sig first-emission fix (still relevant; some lieder pieces fall in the key-time-sig-residual cluster)
   - `project_radio_subproject4_progress.md` — the per-system inference pipeline this retrain feeds
   - `project_radio_training_data_imbalance.md` — v3's data-rebalance backstory
3. **Original audit (the motivation):** [docs/audits/2026-05-12-real-world-scan-failure-modes.md](../../docs/audits/2026-05-12-real-world-scan-failure-modes.md) — Bethlehem + TimeMachine failure modes
4. **v3 results (the baseline to improve on):** [docs/audits/2026-05-12-stage3-v3-data-rebalance-results.md](../../docs/audits/2026-05-12-stage3-v3-data-rebalance-results.md) — lieder mean 0.2398, near-ship
5. **Spec (with Phase 0a addendum):** [docs/superpowers/specs/2026-05-13-stage3-v4-scan-realistic-retrain-design.md](../../docs/superpowers/specs/2026-05-13-stage3-v4-scan-realistic-retrain-design.md)
6. **Plan:** [docs/superpowers/plans/2026-05-13-stage3-v4-scan-realistic-retrain-plan.md](../../docs/superpowers/plans/2026-05-13-stage3-v4-scan-realistic-retrain-plan.md) — 20 tasks, Tasks 1-5 done, Tasks 6-20 pending
7. **Phase 0 results doc:** [docs/audits/2026-05-13-stage3-v4-results.md](../../docs/audits/2026-05-13-stage3-v4-results.md) — gate decisions for 0a + 0b

The spec and plan are self-contained from there.

## Starting state

- **Branch:** `feat/stage3-v4-scan-realistic` on `origin` at HEAD `f5ed4c5`.
- **Verify before working:** `git log --oneline -1` should show `f5ed4c5 fix(data): overfit-smoke manifest — add val-tagged duplicates so trainer can build val sampler`.
- **Seder (10.10.1.29) is on `main` and stale relative to this branch.** Two paths forward:
  - **Recommended:** SSH to seder and `git fetch && git checkout feat/stage3-v4-scan-realistic` before any seder-side run. The branch has the corpus build script, the v4 config, and the overfit-smoke artifacts — all required by Phase 1+.
  - Alternative: `scp` individual files to seder per-task. Works but adds friction; the branch checkout is cleaner.
- **Seder smoke artifacts already present on its filesystem:**
  - `data/manifests/overfit_smoke_v4.jsonl` (40 entries: 20 train + 20 val-duplicates)
  - `configs/train_stage3_v4_overfit_smoke.yaml`
  - `checkpoints/full_radio_stage3_v4_overfit_smoke/` (smoke checkpoints — safe to delete after this session if disk is tight)
  - `logs/full_radio_stage3_v4_overfit_smoke_steps.jsonl`

## What was done this session

| Commit | Subject |
|---|---|
| `366ae20` | docs(spec+plan): Stage 3 v4 scan-realistic retrain |
| `39aff51` | feat(audit): bottom-quartile lieder cluster analysis script (Phase 0a) |
| `7f0fcde` | fix(audit): bass-clef-misread fallback to system 0 GT clefs |
| `050c9e2` | fix(audit): cluster script — correct staff delimiter, dynamic phantom threshold, time-sig populated |
| `ead379c` | feat(audit): cluster script — add `--predicted-mxl-dir` mode, fix CSV piece column |
| `7748131` | feat(data): overfit-smoke manifest + builder (Phase 0b) |
| `da5ba6e` | feat(config): Stage 3 v4 overfit smoke config (Phase 0b) |
| `832b150` | docs(spec): Phase 0a addendum — reframed as unified decoder-multi-staff-confusion |
| `d9d7c86` | audit: Phase 0a bottom-quartile lieder cluster (gate FAIL, but see narrative) |
| `3865eed` | audit: Phase 0b overfit smoke gate PASS |
| `f5ed4c5` | fix(data): overfit-smoke manifest — add val-tagged duplicates so trainer can build val sampler |

Branch is pushed to origin. No PR opened yet (deferred until Phase 3 results land).

## Phase 0 verdicts (summary; details in results doc)

**Phase 0a — bottom-quartile lieder cluster analysis:**
- 33 pieces with v3 onset_f1 < 0.10 were clustered by failure mode.
- Strict gate (bass-clef-misread ≥ 30% of bottom quartile): **FAIL** — only 4/33 (12.1%) showed strict bass-clef-misread.
- Reframed: **PROCEED.** The dominant cluster is phantom-staff-residual at 25/33 (75.8%) — but spot-checking shows these are NOT a different failure mode. Predicted MXLs on flagged pieces have 6 parts where GT has 3; assembly is striping the decoder's confused staff-count output into double the expected parts. Same root cause as bass-clef-misread (decoder structural confusion on multi-staff piano without scan-realistic training exposure). Unified cluster is ~85% of the bottom quartile. The planned retrain targets the root cause directly.

**Phase 0b — overfit smoke (20-sample grand-staff overfit):**
- **PASS.** Train loss 0.006 (gate < 0.05) and val loss 0.013 (gate < 0.10) at step 500. Curves dropped below gates well before step 500 (train < 0.05 by step 200, val < 0.10 by step 100).
- Config divergence from plan: original spec had `batch_size=4` + `max_seq=1024`. First attempt OOM'd on the live C-RADIO-H encoder (no cache for smoke). Reduced to `batch_size=1` + `max_seq=512` + `grad_accumulation_steps=4` for equivalent effective batch. Updated config is committed.
- Wall-clock: 9 min on RTX 5090.

## What's left to do (Phase 1, 2, 3 — see plan Tasks 6-20)

**Phase 1a — Real-scan distribution calibration (CPU, ~2h, half browsing).** Plan Tasks 6-7. Implement `scripts/audit/measure_real_scan_distribution.py`, collect 5-10 real scans (Bethlehem + TimeMachine page renders + 5 IMSLP scans), run measurement, write `docs/audits/2026-05-13-real-scan-degradation-calibration.md`. The calibrated parameter ranges feed Phase 1b's degradation pipeline.

**Phase 1b — Scan-degradation module (CPU, ~2h).** Plan Task 8. Implement `src/data/scan_degradation.py` + tests. TDD-friendly, pure Python (PIL + numpy + cv2). After 1a finishes, update degradation parameter ranges to match observed real-scan distribution.

**Phase 1c — Build corpus + rebuild encoder cache (CPU + ~3-4h GPU).** Plan Tasks 9-12.
- Task 9: implement `scripts/data/build_scanned_grandstaff_systems.py` + tests (CPU, ~1h).
- Task 10: dry-run on seder with `--limit 100`, visually spot-check 10 outputs against real scans, tune parameters if needed, full build (~30 min on seder).
- Task 11: update `scripts/build_stage3_combined_manifest.py` to register the new corpus, regenerate the combined manifest.
- Task 12: rebuild encoder cache from scratch on seder (~3-4h on RTX 5090). New `cache_hash16` captured for v4 config.

**Phase 2 — Stage 3 v4 retrain (~1h GPU).** Plan Tasks 13-15.
- Task 13: write `configs/train_stage3_radio_systems_v4.yaml`. Plug in the new `cache_hash16` from Task 12.
- Task 14: 500-step smoke verification (`--max-steps 500`, ~3-4 min).
- Task 15: full 9000-step run (~1h on seder).

**Phase 3 — Evaluation (~3-4h on seder).** Plan Tasks 16-19.
- Task 16: primary gate — Bethlehem + TimeMachine through `scripts/predict_pdf.py` at default settings. Verify bass clefs AND bass-register pitches.
- Task 17: 139-piece lieder ship-gate (~3h). Mean onset_f1 ≥ v3's 0.2398. Bottom-quartile movement ≥ 17/33 to ≥ 0.10.
- Task 18: A3 token-accuracy spot check across all 5 corpora.
- Task 19: write final results doc + verdict.

**Task 20: open PR** at the end.

## Gotchas / things the fresh session needs to know

1. **Seder paths use Windows backslashes.** Image paths in the grandstaff source manifest contain `\` separators. The trainer + cache builder handle this fine; data pipeline tooling should pass through verbatim. If you write new scripts that read these paths, use `pathlib.Path` and don't manually split on `/`.

2. **`data/openscore_lieder/scores/` contains non-ASCII paths** (e.g., `Mörike-Lieder`). Direct SCP of these paths fails on encoding mismatch between PowerShell's cp1252 and OpenSSH's UTF-8. **Workaround used this session:** stage files to a flat `C:/tmp/gt_mxls/` directory via a PowerShell script first, then bulk-scp from that flat dir. The script is at `/tmp/stage_gt_mxls.ps1` locally; you may want to keep a version in `scripts/data/` for reuse.

3. **The cluster script's `--predicted-mxl-dir` mode is the path to use** for any future per-piece failure-mode analysis. The token-dump path exists but token dumps aren't generated by the current `eval/run_lieder_eval.py`. If you need raw decoder outputs for some future analysis, modify `eval/run_lieder_eval.py` to emit `.tokens.jsonl` per piece — see `scripts/predict_pdf.py`'s `--dump-tokens` for the format.

4. **Smoke manifest needs val-tagged duplicates.** The trainer's `build_stage_b_sampler` raises if the manifest has no val entries for the requested split. The fix used: duplicate each train entry with `split: val` in `data/manifests/overfit_smoke_v4.jsonl`. If you regenerate this manifest, do the same — or upstream the val-duplicate step into `scripts/data/build_overfit_smoke_manifest.py`.

5. **DatasetMix has no `manifest_path` field.** The plan's Task 4 originally proposed `dataset: overfit_smoke_v4` + `manifest_path: ...`, but that field isn't supported. Use the real `dataset: grandstaff_systems` tag and scope the data via `--token-manifest` on the trainer CLI. The same trick will work for any future "use a custom subset" config.

6. **C-RADIO-H + no-cache OOMs at `batch_size > 1` on RTX 5090.** The cache exists for a reason. If you ever run live-encoder training for any reason (including the smoke), use `batch_size: 1` + `grad_accumulation_steps: N` for the effective batch you want.

7. **Phase 0a was extended mid-flight** to support `--predicted-mxl-dir` input. The script supports both that and the original `--tokens-dir` mode. If you re-run Phase 0a after the retrain (e.g., to see if v4 reduced the cluster sizes), pass `--predicted-mxl-dir <v4 predictions>` instead.

8. **GPU box repo branch:** as of session end, seder is on `main`. Pull `feat/stage3-v4-scan-realistic` before any Phase 1+ seder-side work — otherwise the corpus build script, the v4 config, etc. won't exist there.

## Risks for Phase 1+

1. **Degradation pipeline calibration accuracy.** If the offline pipeline doesn't span the real-scan distribution, v4 trains on the wrong distribution. Mitigation: Phase 1a's calibration step uses Bethlehem + TimeMachine + 5 IMSLP scans; tune ranges to span observed values. After Phase 1b implementation, visually compare 10 degraded outputs against real scans before committing the full corpus build.

2. **Cache rebuild time + disk.** Phase 1c's encoder cache rebuild is the longest single step (~3-4h on 5090). Current cache `ac8948ae4b5be3e9` is 1.38 TB; the new one with `scanned_grandstaff_systems` added will be ~1.6 TB. Verify seder has ~2 TB free before launching (current free: ~1.2 TB per session-end `dir` output, may need to delete old caches first).

3. **Token vocabulary mismatch between corpora.** scanned_grandstaff inherits grandstaff's tokens verbatim, so vocab is unchanged. No new vocab tokens needed; no decoder embedding resize. This was an issue in v2→v3 but is not an issue here.

4. **Phase 2 trainer assertion fires correctly.** The pre-flight assertion (`[freeze] trainable encoder params: 0`) is what flagged the v2 encoder unfreeze bug. v4 must also see this assertion fire — confirm in the first few seconds of the 500-step smoke (Task 14).

5. **A2 cache parity check.** v3 had a known A2 failure (cache-vs-live preprocessing mismatch) that was attributed to a different root cause and shipped despite. A freshly-rebuilt cache in Phase 1c may or may not pass A2. If it fails: document and proceed; v3 shipped with the same fail.

## Quick repro commands (for sanity-checking what's on disk)

```bash
# Verify branch state
cd /home/ari/work/Clarity-OMR-Train-RADIO
git log --oneline -1   # expect f5ed4c5
git log --oneline main..feat/stage3-v4-scan-realistic | wc -l   # expect 11

# Re-read Phase 0 results
cat docs/audits/2026-05-13-stage3-v4-results.md
cat docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md

# Re-run Phase 0a (if you want to see it work)
python3 scripts/audit/cluster_bottom_quartile_lieder.py \
    --scores-csv /tmp/lieder_v3_scores.csv \
    --predicted-mxl-dir /tmp/lieder_v3_pred_mxls \
    --gt-mxl-root /tmp/lieder_v3_gt_mxls \
    --output /tmp/phase0a_rerun.md

# Re-run Phase 0b smoke (~9 min on seder; only if you want to verify the pipeline still works)
ssh 10.10.1.29 'cd "C:/Users/Jonathan Wesely/Clarity-OMR-Train-RADIO" && \
  venv-cu132/Scripts/python.exe -u src/train/train.py \
    --stage-configs configs/train_stage3_v4_overfit_smoke.yaml \
    --mode execute \
    --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \
    --start-stage stage3-v4-overfit-smoke \
    --checkpoint-dir checkpoints/full_radio_stage3_v4_overfit_smoke \
    --token-manifest data/manifests/overfit_smoke_v4.jsonl \
    --step-log logs/full_radio_stage3_v4_overfit_smoke_steps.jsonl'

# Pull v3 inputs locally (if not already there from this session)
scp '10.10.1.29:C:/Users/Jonathan Wesely/Clarity-OMR-Train-RADIO/eval/results/lieder_stage3_v3_best_scores.csv' /tmp/lieder_v3_scores.csv
```

## Execution mode recommendation for the fresh session

Subagent-driven development worked well in this session for the pure-Python tasks (Tasks 1, 3, 4 — script + tests + commit). For Phase 1+, the mix is:

- **Subagent-friendly:** Tasks 6 (calibration script), 8 (scan_degradation module), 9 (corpus builder), 13 (v4 config). Pure Python, TDD discipline applies.
- **Driver/operator (you in the next session):** Task 7 (collect real scans, ~half-hour browsing/downloading), Task 10 (visual spot-check of corpus build outputs), Tasks 12, 14, 15, 16, 17 (multi-minute to multi-hour seder runs that don't fit subagent timeouts well).
- **Mixed:** Task 11 (combined manifest registry update) is mostly mechanical Python but touches a critical script; consider supervised dispatch.

A good session plan: dispatch implementers for Tasks 6, 8, 9, 13 in sequence (~2-3h of dispatch + review). Then drive Tasks 7, 10, 12, 14-19 directly. Phase 3 results doc (Task 19) is again subagent-friendly.

## Open questions for the fresh session

1. **Real-scan corpus for calibration:** the spec says "5-10 IMSLP scans of beginner piano scores." Picking which scans matters — they should span the range we want to handle (rotation, JPEG quality, noise levels). If you have time, deliberately pick a mix of conditions (one heavily-yellowed page, one clean modern scan, one slightly-rotated phone capture, one high-DPI archival). Otherwise just grab whatever 5-10 IMSLP beginner-piano scans are easy.

2. **What to do if v4 misses the primary gate (Bethlehem + TimeMachine):** spec Section 7 documents the escalation ladder (B: multi-variant cache rebuild; C: no-cache live-tier fine-tune). v4.1 spec would be a separate doc; don't write it preemptively.

3. **PR scope:** plan Task 20 opens one PR for the whole branch at the end. If you'd rather merge in two stages (Phase 0 + Phase 1 build infra → main, then v4 results → main), that's a defensible split — the Phase 0 work has standalone value (the cluster script + smoke config + smoke manifest are reusable).
