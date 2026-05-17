# Session-end handoff: Robust Stage-A — Sub-plan B design+code COMPLETE (2026-05-17)

> Fresh session: read this. State is clean — a deliberate stopping point,
> nothing in flight. There is a genuine fork in "Do this next" (a seder/GPU
> run vs CPU design work); pick per session type / user priority.

## One-paragraph status

Sub-plan A (strict eval & gate harness) was executed and is **shipped & PASS**
(this session's first half; 32/32 CPU tests; chain head `322004f`, NOTES at
`docs/superpowers/plans/2026-05-17-robust-stage-a-eval-gate-harness-NOTES.md`).
Sub-plan B (retrain-hardening) was then taken brainstorm → spec → plan →
execute and its **design + code is COMPLETE & PASS**: pure torch-free module
+ `train_yolo.py` seam + codified seder worker + acceptance runbook, 17/17 CPU
tests, 15 commits, chain head `a25d8c8`, all on `main`, `HEAD==origin/main`,
tree clean (only unrelated untracked `audit_results/`). **B's one remaining
step is the actual seder/GPU acceptance retrain** — deliberately deferred this
session (design-only by user choice). Sub-plans C and D, and Phase 3, remain.

## Working context to carry forward

- **LM Studio / local-LLM was DOWN all of this session.** Per CLAUDE.md, check
  it at session start (`claude mcp list` / the wiki-flush log) and **flag the
  outage before diagnostic work**. This session delegated all research/impl to
  **sonnet Task subagents** instead — that fallback worked well; reuse it if
  LM Studio is still down.
- **Commit direct to `main`, small commits, push immediately. No PRs.** User's
  explicit standing instruction (reaffirmed this session); it overrides the
  superpowers branch/PR ceremony.
- Execution method that worked: **`superpowers:subagent-driven-development`**
  (fresh sonnet subagent per task + two-stage spec→quality review). 10 of the
  ~18 task-cycles surfaced real review-driven fixes; applying
  `receiving-code-review` rigor (verify the reviewer's claim, don't reflexively
  implement or reject) caught two overstated-mechanism claims and one wrong
  "simpler fix" suggestion. Keep that discipline.
- Wiki auto-flush depends on the local LLM (down), so this session's WIKI
  markers likely did NOT file. The durable carriers are git + auto-memory
  (`project_radio_robust_stagea`, MEMORY.md index updated). If the local LLM is
  back, a manual wiki catch-up for the `clarity-omr-training` /`clarity-omr`
  markers is worthwhile (Ultralytics `--project` path-nesting gotcha; B shipped).

## Do this next — pick ONE track (genuine fork; C does not depend on B's run)

**Track 1 — Close Sub-plan B (needs a seder/GPU session).** Run
`docs/superpowers/plans/2026-05-17-robust-stage-a-subplan-b-ACCEPTANCE.md`
on seder `venv-cu132`. It is self-contained (dual gate: provenance `total>0`
then lieder recall ≥ 0.930 on current `mixed_systems_v1`; PASS = clean OR
guard-halt, NOT `.failed`). **Read its §0 pre-flight caveats first.** Residual
seder caveats also in memory: verify the on-disk path after the first
save-period epoch (Ultralytics `--project` nesting is version-dependent);
`trainer.stop`/`save_model()` on NaN-EMA are seder-only untested — confirm via
the `[stagea-hardening]` out-log line; **fix `docs/TRAINING.md`'s stale
`--project runs/detect/runs` → `--project runs` ONCE the run empirically
confirms the on-disk path** (do not edit speculatively before that).

**Track 2 — Start Sub-plan C (CPU design work; independent of B).** Data engine:
synthetic clutter composition + hard-negative mining. Its own
brainstorm → spec → plan → execute cycle (same flow B just used). Independent
of B per the spec/decomposition; does NOT need the seder run done first. This
is the long pole and is fully designable without GPU.

**Recommended default:** if it's a seder-available session → Track 1 (closes B
end-to-end, unblocks D). If it's a design session / seder busy → Track 2 (C is
independent and the biggest remaining unknown). D needs A+B+C all complete.

## State guarantees (nothing lost by switching sessions)

- All work committed + pushed. `HEAD==origin/main==a25d8c8`. Key SHAs:
  Sub-plan B spec `768affc`, plan `2c35214`, chain `a2af8a6…a25d8c8`;
  Sub-plan A chain head `322004f`, NOTES `763ff52`; parent spec `7cccc1a`.
- Sub-plan B deliverables: `src/train/stagea_hardening.py` (+ 17 tests in
  `tests/stagea_hardening/`), seam in `scripts/train_yolo.py`, worker
  `scripts/seder/run_stagea_hardened_retrain.ps1`, spec/plan/ACCEPTANCE docs.
  Module is torch-free at module top (torch only function-local — the seder
  seam); CPU suite not CUDA-gated.
- The validated faint-ink baseline is intact and **must not be overwritten**:
  `runs/detect/runs/yolo26m_systems_faintink/weights/best.pt` (epoch 33,
  mAP50 0.995). Lieder baseline CSV committed:
  `eval/results/stagea_baseline_pre_faintink.csv` (recall 0.930).
- Decomposition: A done; B design+code done (acceptance run pending); C
  independent, not started; D integrates A+B+C (+abstention) — gated on all
  three; Phase 3 (bass-clef / Stage-B clef bias) parked until robust-Stage-A
  lands. Held-out real archetype set (~15–24 scans, user-provided) is a D
  dependency only — A and B did not need it.

## Don't

- Don't run Sub-plan B's acceptance on anything but seder — it's the GPU run;
  the design/code is already CPU-verified and frozen.
- Don't overwrite the faint-ink `best.pt` or the production baseline.
- Don't edit `docs/TRAINING.md`'s `--project` line until the seder run
  empirically confirms the correct on-disk path (the right value depends on
  Ultralytics internals not verifiable off-seder).
- Don't start D from here, and don't bundle C+D — each is its own
  spec→plan→execute cycle. Don't reopen settled per-task dispositions
  (frozen-dataclass list-fields convention; guard-4 narrowed scope; the 5
  review-driven fixes) — all reviewed, accepted, recorded.
- Don't silently fall back to direct log/config reads if the local agent is
  down — flag it first (CLAUDE.md).
