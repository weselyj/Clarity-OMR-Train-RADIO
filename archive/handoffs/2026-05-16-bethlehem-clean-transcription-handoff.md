# Session-end handoff: Bethlehem clean-transcription — Tasks 1–2 done, Task 3 blocked on a scorer path bug (2026-05-16)

> Fresh session: read this, then the spec + plan. The seder-reliability problem
> that plagued this session is **solved** (scheduled-task mechanism, below) —
> reuse it; do not re-derive it.

## One-paragraph status

Executing the Bethlehem clean-transcription plan via subagent-driven-development.
Spec and plan are committed and approved. **Task 1** (GT scoring harness) and
**Task 2** (faint-ink Stage A augmentation arm) are DONE, committed, and
verified on seder (10/10 scan_noise tests pass incl. real pipeline-wiring).
**Task 3** (Stage A regression baseline) is blocked on a single, well-understood
bug: `eval/score_stage_a_only.py` looks up GT at a flat `scores_dir/{piece}.mxl`
but the lieder GT is deeply nested (`data/openscore_lieder/scores/<Composer>/_/<Title>/lcNNNN.mxl`
— 1462 `.mxl` recursively, 0 flat), so it matched 0/145 manifests → empty CSV
→ ZeroDivisionError. The 145 Stage-A manifests themselves built fine. Tasks 4
(multi-hour faint-ink YOLO retrain) and 5 (Phase 2 gate) are pending. Phase 3
(Stage B clef bias) is intentionally deferred per the spec.

## CRITICAL: seder long-job reliability is SOLVED — use the scheduled-task mechanism

Three seder jobs died this session. Root cause was **NOT reboots** (that was a
mis-diagnosis; corrected by timestamp evidence — a job lived ~3 s while seder
was continuously up). Root cause: **`Start-Process -NoNewWindow` launched via a
short-lived `ssh ... -Command` is reaped by Windows OpenSSH's job-object cleanup
when the ssh session closes.**

**The fix (validated end-to-end — 145/145 in ~90 s, ssh-independent):** launch
via a **Windows Scheduled Task**, which runs in the Task Scheduler service
session, independent of any ssh session (and survivable across reboots).

Mechanism (reusable for Task 4):
1. Write a self-contained worker `.ps1` locally that `Set-Location`s to the
   repo, runs the python, and writes its own log + a `.done`/`.failed` marker
   (stdout is NOT captured by schtasks — the script must self-log).
2. `scp` the worker to a **no-space path** on seder: `C:\radio_jobs\<job>.ps1`
   (avoids `schtasks /tr` quoting hell — repo path has a space).
3. Register + run:
   ```
   ssh 10.10.1.29 'schtasks /create /tn <job> /tr "powershell -NoProfile -ExecutionPolicy Bypass -File C:\radio_jobs\<job>.ps1" /sc ONCE /st 23:59 /f & schtasks /run /tn <job>'
   ```
4. Poll ssh-independently for the markers (`logs/<job>.done` / `.failed`), and
   treat "task not Running AND no marker" as DIED:
   ```
   schtasks /query /tn <job> /fo LIST | Select-String Status
   ```

Existing artifacts on seder:
- `C:\radio_jobs\stagea_baseline_task.ps1` (the validated baseline worker — copy
  its structure for the retrain worker)
- Scheduled task `radio_stagea_baseline` exists; baseline manifests are at
  `eval/results/stagea_baseline_manifests/` (145 `*_stage_a.jsonl`, complete)

**WIKI:** clarity-omr — Seder long jobs: `Start-Process -NoNewWindow` via
`ssh -Command` is reaped on ssh-session close (proven: ~3 s process lifetime,
no reboot). Use a Windows Scheduled Task at a no-space `C:\radio_jobs\` path;
the worker self-logs + writes `.done`/`.failed`; poll markers ssh-independently.

## The immediate blocker (Task 3) — exact fix

`eval/score_stage_a_only.py` (PR #38-era) hardcodes a flat lookup:
`mxl = scores_dir / f"{piece}.mxl"`. Real lieder GT layout (verified on seder):
`data/openscore_lieder/scores/<Composer>/_/<Title>/lcNNNN.mxl` — nested, 1462
files. Manifest piece IDs are bare `lcNNNN` (e.g. `lc28688206`).

Fix options (pick one; lowest-friction first):
- Build a one-time `{lcNNNN: full_path}` index by recursively globbing
  `scores_dir/**/*.mxl` and keying on `path.stem`, then look up by piece id.
  Either patch `score_stage_a_only.py` to do the recursive index (cleanest,
  reusable) or pass a corrected `--scores-dir` if a flat mirror exists (none
  found — recursive index is the real fix).
- TDD it: the script has tests at `tests/eval/test_score_stage_a_only.py` —
  add a nested-layout fixture, then make it pass. Commit direct to main.

After the scorer is fixed: re-run the baseline via the **scheduled-task
mechanism** (worker already exists, just re-run task `radio_stagea_baseline`
after the scorer fix is pulled on seder), capture `eval/results/stagea_baseline.csv`
aggregate (total_expected / total_detected / total_missing / recall). That CSV
is the pre-faint-ink reference for the Task 5 no-regression gate. Snapshot it
into the repo (`eval/results/stagea_baseline_pre_faintink.csv`) so the gate is
reproducible.

## What's done (commits on main this session, relevant to this plan)

| Commit | What |
|---|---|
| `4d6c5ff` | spec: Bethlehem clean-transcription design |
| `c3aba96` | plan: implementation plan |
| `6dfe84f` | Task 1 — GT scoring harness |
| `77d4963` | Task 1 fix — clef_accuracy per-occurrence (smoke caught per-part-majority masking the exact Defect-2 flip) |
| `9f74903` | plan: corrected Step 1.5 expected clef_accuracy → 0.83 |
| `1d30865` | plan: Task 2 — dropped a fake test + production hook, real pipeline introspection |
| `d6e00f9` | Task 2 — faint-ink augmentation arm in `src/train/scan_noise.py` |
| `bfbb344` | fix — repo-root sys.path bootstrap in `eval/run_stage_a_only.py` (was blocking the baseline) |

Task 1 real-data smoke (validated): Bethlehem v4 pred vs GT →
measure_recall 0.7647 (13/17), clef_accuracy 0.8333 (5/6 — the one flipped
bass system now visible, not masked), note_onset_f1 0.08.

Task 2: 10/10 `tests/train/test_scan_noise.py` pass on seder, including
`test_faint_ink_transform_wired_into_pipeline` PASSED (real Albumentations
Compose contains the faint-ink OneOf at p=0.25). NOTE: `tests/train/` is
CUDA-gated by conftest — these tests only execute on seder, not locally.

## What's left

- **Task 3** (controller): fix `score_stage_a_only.py` nested-GT lookup → re-run
  baseline via scheduled task → snapshot baseline CSV. ~30 min once the scorer
  is fixed.
- **Task 4** (controller, long pole): faint-ink YOLO retrain on seder via the
  **scheduled-task mechanism** (NOT Start-Process). Plan Task 4 has the exact
  `train_yolo.py --noise` command; locate the `data.yaml` from
  `runs/detect/runs/yolo26m_systems/args.yaml` first (plan Step 4.1). Multi-hour;
  the scheduled task is what makes it survivable.
- **Task 5** (controller): Phase 2 gate — Bethlehem 4-system recovery at default
  conf 0.25 with the new weights (`scripts/audit/dump_system_crops.py`) AND
  no-regression vs the Task 3 baseline. Then re-score Bethlehem with
  `scripts/audit/score_against_gt.py` vs GT. Verdict → `docs/audits/2026-05-16-bethlehem-results.md`.
- **Phase 3** (deferred): Stage B clef bias. Trigger = Phase 2 passes AND the
  Bethlehem re-score still shows the rest-heavy bass flipping to G2 (plan Step
  5.5). Gets its own spec/plan; baseline = v3 lieder mean 0.2398.

## Key paths

- Spec: `docs/superpowers/specs/2026-05-16-bethlehem-clean-transcription-design.md`
- Plan: `docs/superpowers/plans/2026-05-16-bethlehem-clean-transcription-plan.md`
- GT (Bethlehem): `/home/ari/musicxml/Scanned_20251208-0833_20260516.musicxml`
- Bethlehem scan on seder: `C:/Users/Jonathan Wesely/Downloads/Scanned_20251208-0833.jpg`
- Crop dumper (Stage-A diagnosis): `scripts/audit/dump_system_crops.py`
- GT scorer: `scripts/audit/score_against_gt.py`
- Current YOLO weights: `runs/detect/runs/yolo26m_systems/weights/best.pt`

## Workflow note (inherited via ~/CLAUDE.md)

This project is now **commit-direct-to-main**, no feature branches, no PRs
unless explicitly requested. Each commit one logical change, Conventional
Commit prefix, push when green. Already in `~/CLAUDE.md` — the fresh session
inherits it.

## Parked, unrelated

v4 Stage-3 lieder eval is parked at 129/146 cached pieces on seder
(`eval/results/lieder_stage3_v4_best/`). Separate from this plan; resume only
if v4 ship-gate numbers are needed. v4 itself shipped to main earlier
(val_loss 0.294).

## Don't repeat these this session

- Don't blame seder reboots for job deaths — it's the ssh-session-close kill;
  use the scheduled-task mechanism.
- Don't launch seder long jobs with `Start-Process -NoNewWindow` via `ssh -Command`.
- Don't pipe `|`-alternation regexes through `ssh → cmd → powershell` (cmd eats
  the `|`); use `Select-String -Pattern a,b` arrays or read files back and grep
  locally.
- `tests/train/` is CUDA-gated — verify those on seder, not locally.
