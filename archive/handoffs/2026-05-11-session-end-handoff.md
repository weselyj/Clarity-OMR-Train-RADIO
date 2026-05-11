# Session-end handoff: Stage 3 v2 audit arc → v3 data-rebalance retrain (2026-05-11)

> Hand off to a fresh session. Read this doc first, then the spec, then the plan. Total context-load before doing any work: ~15-20 minutes.

## One-paragraph status

The Stage 3 v2 RADIO checkpoint scores mean onset_f1 ≈ 0.06 on the 4-piece HF demo eval (vs the project's 0.241 decision gate). This session ran a complete diagnostic arc that disproved an early hypothesis (encoder drift was a real bug but not the dominant cause), then identified the actual root cause: the Stage 3 v2 training had `synthetic_systems` (the smallest corpus, 20.5k entries) weighted at 77.8% of cached-tier gradient share, with zero val split — flying blind on its largest gradient source while under-weighting the multi-staff piano content (`grandstaff_systems`, 97k entries at 11.1%) that the demo eval actually tests. The fix is at the data-pipeline level, possibly with a decoder capacity bump. The spec + plan for the fix are in tree at the paths below. The fresh session executes the plan via subagent-driven development.

## What the fresh session should read, in order

1. **This handoff doc** (you're reading it)
2. **Memory entries** (run these in your head as a checklist before reading the spec):
   - `project_radio_stage3_v2_audit.md` — encoder DoRA finding (still true; encoder freeze is real bug, just not dominant)
   - `project_radio_frankenstein_diagnostic.md` — audit conclusion DISPROVEN
   - `project_radio_training_data_imbalance.md` — the actual root cause
3. **Spec:** [docs/superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md](../../docs/superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md)
4. **Plan:** [docs/superpowers/plans/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-plan.md](../../docs/superpowers/plans/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-plan.md)

That's it for required reading. The spec and plan are self-contained.

## Starting state

- **Branch:** start a new branch `feat/stage3-v3-data-rebalance` off `main` at SHA `6e9ed0e` (commit that landed the plan).
- **Verify before working:** `git log --oneline -1` should show `6e9ed0e docs(plan): data-pipeline audit + capacity audit + Stage 3 v3 retrain plan`.
- **Seder is on `main` and is current.** No additional sync needed beyond what the user does at session start.

## What was done this session (PRs merged)

| PR | What |
|---|---|
| #46 | Stage D part-alignment fix (prior session, merged earlier today). Background context. |
| #47 | First-emission key/time signature fix. Small pipeline correctness fix; ~+43% relative onset_f1 on Prelude. Already in main. |
| #48 | Stage 3 v2 training audit — diagnosed the encoder-DoRA-unfreeze bug. Already in main. |
| #49 | Failure-mode investigation — frankenstein experiment disproved the audit conclusion; Streams A/B identified decoder undergeneration on multi-staff content. Already in main. |

## Stale artifacts the fresh session should ignore

- **`docs/superpowers/specs/2026-05-11-stage3-v3-retrain-design.md`** — superseded by the new spec. The encoder-freeze design from it carries forward, but the implicit premise (that fixing the freeze alone would close the gap) was disproven.
- **`docs/superpowers/plans/2026-05-11-stage3-v3-retrain-plan.md`** — Tasks 1, 2, 3 were executed (resulting in PR #49). Tasks 4-9 are superseded by the new plan. Do not pick them up.
- **`docs/superpowers/specs/2026-05-11-stage3-v2-failure-mode-investigation-design.md`** — already executed (PR #49). Historical record only.
- **`docs/superpowers/plans/2026-05-11-stage3-v2-failure-mode-investigation-plan.md`** — already executed. Historical.

## Things that landed but didn't get a dedicated PR header

- `src/checkpoint_io.py` got a one-line loader fix (accepts both `model` and `model_state_dict` keys). On main.
- The audit infrastructure scripts in `scripts/audit/` exist on main: `_sample_picker.py`, `a1_preprocessing_parity.py`, `a2_encoder_parity.py`, `a3_decoder_on_training.py`, `pipeline_note_loss.py`, `build_frankenstein_checkpoint.py`. The new plan adds 5 more.

## First 10 minutes checklist for the fresh session

- [ ] `cd /home/ari/work/Clarity-OMR-Train-RADIO && git log --oneline -3` — confirms branch state
- [ ] `git checkout -b feat/stage3-v3-data-rebalance` — create the new branch
- [ ] Read memory entries (3 files, ~5 min)
- [ ] Skim the spec (~5 min — section headers + decision matrix)
- [ ] Skim Tasks 1-5 of the plan (independent Phase 1 streams; the rest will read in context as they execute)
- [ ] Verify SSH to seder works: `ssh 10.10.1.29 'echo ok'` (Windows cmd.exe, `&&` not `;`)
- [ ] Verify `venv-cu132\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` returns `True` on seder
- [ ] Dispatch Task 1 implementer via subagent-driven-development

## Operational details a fresh session needs

**Seder access:**
- SSH alias `10.10.1.29` configured for user "Jonathan Wesely"
- Windows cmd.exe shell — use `&&` to chain commands (NOT `;`); use `cd /D` to change drive+dir
- SCP: `scp <local> '10.10.1.29:<basename>'` lands the file in the home dir; then `ssh 10.10.1.29 'move /Y "%USERPROFILE%\<basename>" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\<path>"'`
- Python: `venv-cu132\Scripts\python.exe` (CUDA 13 nightly torch + PEFT installed)

**Key paths on seder:**
- Stage 2 v2 checkpoint: `checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt`
- Stage 3 v2 checkpoint: `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt`
- Encoder cache: `data/cache/encoder/ac8948ae4b5be3e9/`
- Combined manifest: `src/data/manifests/token_manifest_stage3.jsonl`
- Synthetic manifest: `data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl`

**Local agent vs sonnet subagent:**
- Memory entry `feedback_local_agent_during_gpu_training.md` says to skip `delegate_to_local_agent` during Clarity-OMR work because the local LLM saturates the same GPU. Use sonnet subagent for delegation.

## Project-level direction (for the user's awareness, not the fresh session's first concern)

The user has stated the goal is to match or exceed Audiveris quality. The current sub-project tests whether RADIO is viable when trained correctly. Outcomes:
- **SHIP:** RADIO viable; close out the audit/diagnose chapter; decide on deployment/scope-down.
- **Architecture sub-project:** RADIO insufficient; DaViT baseline + architecture investigation queued.
- **Re-audit:** data fix didn't take; regenerate synthetic likely the next sub-project.

The DaViT baseline reproduction remains a parallel-thread option throughout. Audiveris benchmark is independent.

## One more thing: the user's working style

- Prefers concrete, evidence-backed reasoning over speculation
- Has a memory entry `feedback_decide_dont_checkbox.md` — don't present obvious decisions as user-facing choices; decide and present the plan
- Has a memory entry `feedback_trust_progress.md` — don't re-ask "proceed?" between tasks the user has approved a plan for
- Likes the fast-exit pattern (halt gates) used in the prior audit plans — apply liberally

Good luck.
