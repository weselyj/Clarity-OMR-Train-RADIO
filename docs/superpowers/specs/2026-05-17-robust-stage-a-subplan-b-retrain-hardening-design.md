# Robust Stage-A — Sub-plan B: Retrain-Hardening — Design

> Spec for the "retrain-hardening" sub-plan of the Robust-Stage-A effort.
> Parent spec: `2026-05-17-robust-stage-a-clutter-detection-design.md` (`7cccc1a`).
> Sibling shipped: Sub-plan A (eval/gate harness), code head `322004f`.
> Decomposition: Sub-plan A done; **B = this doc**; C (data engine) independent;
> D (iterative loop) integrates A+B+C and is gated on B. Phase 3 deferred.

## Problem

The faint-ink Stage-A YOLO retrain (`yolo26m_systems_faintink`) had its **EMA go
NaN mid-training at epoch ~34**. Root cause is undiagnosed. It is confirmed *not*
hardware (driver 596.49 verified), *not* the already-fixed Albumentations
`Morphological`-on-bboxes bug (`1ffe7df`+`fd3f903`), and *not* covered by the
noise warmup (which only ramps the first ~2000 batches). The run survived only
**by accident**: Ultralytics skips checkpoint saves while the EMA is non-finite,
and `EarlyStopping(patience=20)` halted it 20 wasted epochs later, leaving a
usable `best.pt` (epoch 33, mAP50 0.995, mAP50-95 0.930, 0/768 non-finite
tensors). `--nan-guard` was **absent** from that run; no explicit `lr0`/`lrf`/
grad-clip `max_norm` were set (all Ultralytics defaults).

Per the parent spec: *"A perfect bar cannot be chased on a training process that
randomly NaNs. Retrain-hardening is a prerequisite to entering the loop, not an
optimization."* D's loop will invoke this retrain recipe repeatedly and mostly
unattended, so silent corruption / wasted GPU-hours compound.

## Goal

Make the Stage-A retrain **numerically safe and non-silently-wasteful** on the
*current* `mixed_systems_v1` faint-ink data — without chasing the NaN's root
cause and without perturbing the accuracy-validated recipe. Deliver a hardened
recipe + a reusable seder worker, both CPU-unit-tested where logic is pure, and
one acceptance retrain demonstrating the bar below.

### Success criteria (acceptance bar: graceful-safe + no-regression)

B PASSES iff a single hardened validation retrain on `mixed_systems_v1`:

1. **Either** completes cleanly **or**, if numerics still blow up, the active
   guard halts at the last good state having written an explicit checkpoint;
   **and**
2. that checkpoint passes `validate_checkpoint_finite` (0 non-finite tensors);
   **and**
3. that checkpoint meets **lieder recall ≥ 0.930** scored through the existing
   Sub-plan-A lieder path (`eval/score_stage_a_only.py` →
   `recall_from_stagea_csv`), i.e. no regression vs the faint-ink baseline
   `eval/results/stagea_baseline_pre_faintink.csv`.

We are explicitly **not** requiring "NaN never happens" (root cause deferred by
decision) and **not** chasing an accuracy gain (≥0.930 no-regression only). The
proof obligation is that the process is no longer *silently* corrupted or
wasteful and the recipe is sound for D to reuse.

## Approach

**Pure CPU-unit-tested hardening module + thin Ultralytics training seam**
(mirrors the Sub-plan-A pure-core / thin-GPU-seam architecture; reuses Stage-B's
already-proven NaN-halt + explicit-clip patterns rather than new logic).

```
src/train/stagea_hardening.py          # PURE — no training run needed to test
  is_nonfinite_state(loss, ema_finite, grad_norm) -> (bool, reason)
  should_halt(state, step, epoch, cfg) -> (bool, reason)   # mirrors Stage-B _should_sanity_halt
  build_hardened_overrides(args) -> dict   # the pinned lr0/lrf/max_norm/save_period/amp kwargs
  validate_checkpoint_finite(path) -> (ok, n_nonfinite, total, first_key)

scripts/train_yolo.py                  # THIN SEAM (GPU/seder only)
  registers Ultralytics callbacks (on_train_batch_end / on_fit_epoch_end) that
  call into stagea_hardening; passes build_hardened_overrides() into model.train()
```

### Defense-in-depth guards (the hardened recipe)

0. **`--nan-guard` always on** for the hardened recipe (it was absent from the
   NaN'd run). Existing behavior unchanged: zero NaN/Inf grads before the
   optimizer step.
1. **Explicit grad-clip `max_norm`** — exposed as `--max-grad-norm`, **default
   `1.0`** (Stage-B's proven value). Stops relying on Ultralytics' opaque
   internal clip.
2. **LR / AMP loss-scale review (pinned decision):** keep `lr0=0.01`,
   `lrf=0.01` **explicit** (not blind defaults) — these produced the validated
   mAP50 0.995; perturbing LR risks an accuracy regression and LR is not
   evidenced as the NaN cause. The stability intervention goes into the numerics
   path instead: `--nan-guard` + explicit grad-clip + a **conservative AMP
   posture**. AMP stays **on** by default with the guards; **documented
   contingency**: if the acceptance retrain still produces a guard-caught
   non-finite blow-up, the first confirmatory remediation is a rerun with
   `--amp` off (decision rule, exercised in the plan — not a standing default).
3. **Active halt-on-NaN + guaranteed checkpoint** — a callback that, on the
   first non-finite EMA/loss (checked `on_fit_epoch_end`, with a finer
   `on_train_batch_end` net), writes an explicit last-good checkpoint **and**
   sets `trainer.stop = True`. Replaces the accidental skip-save + patience=20
   net that wasted ~20 epochs.
4. **Cheap anomaly instrumentation (capture-if-recurs)** — per-epoch structured
   log (loss components, grad-norm, LR, AMP scale); on first non-finite, dump
   the immediate pre-NaN context. Yields the root-cause signal *for free if it
   recurs*, with no deliberate bisect. A slow `torch.autograd.set_detect_anomaly`
   path stays behind an off-by-default `--debug-anomaly` flag.

### best.pt provenance validation

`validate_checkpoint_finite(path)` loads the checkpoint state_dict on CPU
(`map_location="cpu"`), scans every tensor for NaN/Inf, returns
`(ok, n_nonfinite_tensors, total, first_offending_key)`. It is the **mandatory
gate** on whatever checkpoint the retrain produces (clean `best.pt` or the
active-halt last-good) **before** it may be scored for the ≥0.930 bar. A
non-finite checkpoint is an automatic B-FAIL regardless of metrics. Reusable by
D's loop.

### Seder-worker minimal-reliable hardening

One committed reusable worker script codifying the validated
`C:\radio_jobs\*.ps1` schtasks pattern (no-space path; `schtasks /create
/sc ONCE` + `/run`; `.done`/`.failed` markers; self-redirected logs), plus
exactly four fixes:

1. **UTF-8 logging** — `PYTHONIOENCODING=utf-8` + `-Encoding utf8` on all
   redirected logs (replaces the ASCII pid-file encoding).
2. **stderr non-fatal** — `$ErrorActionPreference="Continue"`, success gated
   **only** on `$LASTEXITCODE` (kills the git/libpng-stderr → NativeCommandError
   → job-death failure mode).
3. **`save_period`** — periodic epoch checkpoints, **pinned to every 5 epochs**
   (bounds mid-run-death waste to ≤5 epochs without excessive disk).
4. **Resume-from-last-checkpoint** — worker detects an existing `last.pt` and
   relaunches with Ultralytics `--resume` instead of restarting from epoch 0.

No independent stall watchdog/heartbeat (no evidence of training hangs; TDR is a
verified non-fatal display watch-item — out of scope by decision).

## Testing strategy

- **CPU unit tests** (`tests/stagea_hardening/`, no CUDA-gated path token):
  `is_nonfinite_state` / `should_halt` (finite, NaN, Inf, threshold cases
  mirroring Stage-B's proven `_should_sanity_halt`); `build_hardened_overrides`
  (exact pinned kwargs incl. `max_norm=1.0`, `lr0=0.01`, `lrf=0.01`,
  `save_period=5`, `amp=True`); `validate_checkpoint_finite` (synthetic
  finite / NaN-injected / Inf-injected checkpoints). The guard logic is proven
  correct without a GPU run.
- **GPU/seder integration** = B's single acceptance retrain on
  `mixed_systems_v1`, scored against ≥0.930 through the existing Sub-plan-A
  path. Not unit-tested (same posture as `run_gate.py`). Runs via the hardened
  worker on seder's `venv-cu132`.

## Non-goals (explicit)

No deliberate NaN reproduction/bisection; no stall watchdog/heartbeat; no
accuracy chasing (≥0.930 no-regression only); not Sub-plan C's clutter data;
not the user-provided held-out archetype set; not D's loop integration; not the
explicit text/non-music class fallback; not Phase 3. B ends when the hardened
recipe + reusable worker exist, the pure logic is CPU-tested, and one acceptance
retrain has demonstrated graceful-safe + no-regression.

## Pinned decisions (resolving the parent spec's deferred knobs for B)

- Strategy: **defense-in-depth** (not root-cause-bisect, not minimal).
- Worker: **minimal-reliable** (codify schtasks + UTF-8 + stderr/$LASTEXITCODE +
  `save_period=5` + resume-from-last; **no** watchdog).
- Acceptance: **graceful-safe + no-regression** (guard-caught halt with a
  finite checkpoint ≥0.930 is a PASS; "must-not-NaN" and "accuracy margin"
  bars rejected).
- Architecture: **Approach 2** — pure module + thin Ultralytics callback seam.
- Data: current `mixed_systems_v1` (isolated from C and the held-out set).
- `--max-grad-norm` default `1.0`; `lr0=0.01`/`lrf=0.01` kept explicit;
  `save_period=5`; AMP on with guards, `--amp`-off as a documented contingency
  decision rule (not a default).

## Reusable assets

- Validated faint-ink checkpoint (do NOT overwrite):
  `runs/detect/runs/yolo26m_systems_faintink/weights/best.pt` (epoch 33).
- Lieder baseline: `eval/results/stagea_baseline_pre_faintink.csv` (0.930).
- Sub-plan-A scorer: `eval/score_stage_a_only.py` /
  `eval/robust_stage_a/gate.py:recall_from_stagea_csv`.
- Existing `--nan-guard` in `scripts/train_yolo.py`; Stage-B's
  `_should_sanity_halt()` + `clip_grad_norm_(max_norm=1.0)` in
  `src/train/train.py` as the proven patterns to mirror.
- Scan-noise pipeline (bbox bug fixed): `src/train/scan_noise.py`.
- Validated schtasks worker structure: `archive/handoffs/2026-05-16-bethlehem-clean-transcription-handoff.md`.
