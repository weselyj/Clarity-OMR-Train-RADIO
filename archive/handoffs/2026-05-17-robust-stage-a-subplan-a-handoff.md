# Session-end handoff: Robust Stage-A — Sub-plan A ready to execute (2026-05-17)

> Fresh session: read this, then **execute Sub-plan A**. State is clean — this
> is a deliberate stopping point, not a mid-task handoff. Nothing is in flight.

## One-paragraph status

The Bethlehem clean-transcription plan (Phases 1–2) is **shipped and PASS**
(Defect 1 — missed faint system — resolved end-to-end; Defect 2 — bass-clef
flip — confirmed and deferred to Phase 3). A spot-check of the faint-ink model
on heterogeneous real scans exposed a Stage-A robustness gap (boxes a title as
a system *and* misses a real system; confident phantom systems on non-music
docs). That was brainstormed into a spec, decomposed into four sub-plans, and
**Sub-plan A (the strict eval & gate harness) is fully written**. The immediate
next action is to **execute Sub-plan A via subagent-driven-development**. It is
CPU-only and fully local — **no seder/GPU needed** for any of its 6 tasks.

## Do this next

Execute: **`docs/superpowers/plans/2026-05-17-robust-stage-a-eval-gate-harness.md`**

- Use `superpowers:subagent-driven-development` (fresh subagent per task +
  two-stage review). The plan is self-contained (exact paths, complete code,
  exact commands, expected output) — implementer subagents need only the task
  text, not session history.
- 6 TDD tasks: manifest schema/validator → IoU+matching → per-scenario
  geometry/lyric-clip verdict → lyric-system recall + combined gate →
  lieder-baseline CSV reader → CLI orchestrator. All verifiable on CPU here
  (`python3 -m pytest tests/robust_stage_a/ -v`). Only `run_gate._infer` touches
  YOLO/GPU and is *not* exercised in Sub-plan A (it's a Sub-plan-D concern).
- Commit-direct-to-main, per-commit green, Conventional Commit prefixes — the
  plan's per-task commit commands are already written.

## Context (spec, decomposition, sequencing)

- **Spec:** `docs/superpowers/specs/2026-05-17-robust-stage-a-clutter-detection-design.md`
  (committed `7cccc1a`). Goal: model-level robust Stage-A — never box
  text/garbage *and* find true staves amid clutter (balanced). Strictest ship
  gate: per-scenario zero bbox error on a held-out real set + lieder
  no-regression vs 0.930 + lyric-system recall sub-metric. The lieder/lyrics
  regression trap is the central risk; mitigations are explicit in the spec.
- **Decomposition (each its own spec→plan→execute cycle):**
  - **A — strict eval & gate harness** ← *written, execute now* (`fd74111`).
  - **B — retrain-hardening** (grad-clip/LR/AMP to kill the epoch-~34 NaN;
    seder-worker robustness). *Future; needs my seder/NaN context — write that
    plan in a session with that history, or re-read the memories below.*
  - **C — data engine** (synthetic clutter composition + hard-neg mining).
  - **D — iterative robustness loop** (integration of A+B+C; abstention; the
    B-fallback explicit-text-class contingency).
- **Phase 3 (bass-clef / Stage-B clef bias):** parked, deferred *after*
  robust-Stage-A per the spec sequencing (Stage-A errors poison everything).
- **User dependency:** the held-out real archetype set (~15–24 scans, ~6–8
  distinct failure modes incl. lyrics archetypes, never trained on) is
  user-provided. Sub-plan A does **not** need it — it's built/tested with
  synthetic fixtures; the real set is consumed later by `run_gate.py`.

## State guarantees (nothing lost by switching sessions)

- All work committed and pushed; `HEAD` = `fd74111` (Sub-plan A), local == remote.
  Key SHAs: spec `7cccc1a`, Sub-plan A `fd74111`, Bethlehem verdict `f4683e2`,
  faint-ink fix chain `1ffe7df`→`fd3f903`, scorer fix `a4387c1`, baseline
  snapshot `ce30da7`.
- **seder is verified clean**: redundant run #2 killed, both `radio_*`
  scheduled tasks deleted, session scratch removed, GPU idle, nothing will
  auto-relaunch. Production `runs/detect/runs/yolo26m_systems` and the
  validated faint-ink `yolo26m_systems_faintink/weights/best.pt` (epoch 33,
  mAP50 0.995, 44,296,793 bytes, mtime 5/16 11:02 PM) are intact.
- Memories updated: `project_radio_faintink_stagea` (best.pt = epoch 33,
  pre-NaN, usable; mid-training NaN at ~ep34 is a known instability),
  `project_seder_gpu_instability` (CORRECTED — TDR dumps are a display-level
  watch-item; baseline training works on driver 596.49; the deaths were our
  faint-ink bbox bug, not the box — do **not** roll the driver without
  capturing the actual fault). Wiki has the seder/PowerShell/UTF-16/NaN gotchas.

## Don't

- Don't run Sub-plan A on seder — it's CPU/local; that's the point of it.
- Don't start B/C/D from this plan — each gets its own spec→plan cycle.
- Don't touch the production baseline or the validated faint-ink `best.pt`.
- Don't re-derive seder state — it's clean and documented above; only revisit
  it when writing Sub-plan B (retrain-hardening), which is where that context
  actually matters.
