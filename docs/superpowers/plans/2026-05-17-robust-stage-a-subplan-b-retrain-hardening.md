# Robust Stage-A — Sub-plan B: Retrain-Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Stage-A YOLO retrain numerically safe and non-silently-wasteful via a CPU-unit-tested hardening module + a thin Ultralytics callback seam + a reliable seder worker, validated by one graceful-safe + no-regression acceptance retrain.

**Architecture:** Pure decision/validation logic (`is_nonfinite_state`, `should_halt`, `build_hardened_overrides`, `scan_state_for_nonfinite`) lives torch-free in `src/train/stagea_hardening.py` and is fully CPU-unit-tested. `scripts/train_yolo.py` registers Ultralytics callbacks that feed plain scalars into the pure logic (the thin GPU/seder seam). `validate_checkpoint_finite` is a thin `torch.load` wrapper that delegates counting to the pure `scan_state_for_nonfinite` — same untested-seam posture as Sub-plan A's `run_gate._infer`. A codified seder worker runs the acceptance retrain.

**Tech Stack:** Python 3.14 (CPU for the pure module/tests), pytest, Ultralytics YOLO 8.4.48 (`model.add_callback`/trainer hooks — seam only), PowerShell schtasks worker on seder `venv-cu132`, the existing `eval/score_stage_a_only.py` lieder scorer.

**Spec:** [docs/superpowers/specs/2026-05-17-robust-stage-a-subplan-b-retrain-hardening-design.md](../specs/2026-05-17-robust-stage-a-subplan-b-retrain-hardening-design.md)

---

## Spec refinement (conscious, documented — like Sub-plan A's deviations)

The spec lists `validate_checkpoint_finite` among the pure module functions and says it is CPU-unit-tested with synthetic checkpoints. Local reality: torch is not installed locally (project policy: torch lives on seder's `venv-cu132`; do not warp production code around a local CPU torch — memory `feedback_use_gpu_box_for_torch_tests`). Resolution, preserving the spec's *intent* (CPU-verifiable provenance logic; torch isolated to a thin seam exactly as `run_gate._infer`): the provenance check is split into

- `scan_state_for_nonfinite(items)` — **pure, torch-free, fully CPU-unit-tested** (the actual counting logic), and
- `validate_checkpoint_finite(path)` — a **thin torch seam** that does `torch.load(map_location="cpu")`, computes per-tensor finiteness, and delegates the count to `scan_state_for_nonfinite`. Not unit-tested locally (same posture as Sub-plan A's `run_gate._infer`); exercised on seder during the acceptance run.

No spec requirement is dropped; the verifiable logic is still 100% CPU-tested.

## File structure

**New:**
- `src/train/stagea_hardening.py` — pure hardening logic + the thin torch provenance seam. No top-level torch import.
- `tests/stagea_hardening/__init__.py` — package marker (empty).
- `tests/stagea_hardening/test_stagea_hardening.py` — CPU unit tests for all pure logic.
- `scripts/seder/run_stagea_hardened_retrain.ps1` — codified schtasks worker (UTF-8, stderr-safe, save_period, resume-from-last).
- `docs/superpowers/plans/2026-05-17-robust-stage-a-subplan-b-ACCEPTANCE.md` — the exact acceptance runbook (dual gate: provenance then ≥0.930).

**Modified:**
- `scripts/train_yolo.py` — add `--max-grad-norm`/`--save-period`/`--debug-anomaly`; make `_patch_nan_guard` honor a configured max_norm; register the active halt-on-NaN callback; pass hardened overrides into `model.train()`. (Currently: argparse 19–76, `_patch_nan_guard` 79–116, `main` 119–151, `model.train(**train_kwargs)` at line 147.)

**Why `tests/stagea_hardening/`:** `tests/conftest.py` CUDA-gates only test paths whose parts intersect `{"inference","pipeline","cli","models","train"}`. `stagea_hardening` matches none, so these tests run on CPU. The pure module must never import torch at top level so collection/imports succeed without torch. (Confirmed: the gate keys on the *test file* fspath parts only; `src/train/stagea_hardening.py` being under `src/train/` is irrelevant to gating.)

---

## Task 1: Pure module skeleton + `is_nonfinite_state`

**Files:**
- Create: `src/train/stagea_hardening.py`
- Create: `tests/stagea_hardening/__init__.py`
- Test: `tests/stagea_hardening/test_stagea_hardening.py`

- [ ] **Step 1: Create the empty package marker**

```bash
mkdir -p tests/stagea_hardening
: > tests/stagea_hardening/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/stagea_hardening/test_stagea_hardening.py`:

```python
"""CPU unit tests for the pure Stage-A retrain-hardening logic."""
import pytest

from src.train.stagea_hardening import is_nonfinite_state


def test_finite_state_is_clean():
    assert is_nonfinite_state(0.42, ema_finite=True, grad_norm=1.3) == (False, "")


def test_loss_none_is_nonfinite():
    nf, reason = is_nonfinite_state(None, ema_finite=True)
    assert nf is True and "loss non-finite" in reason


def test_loss_nan_is_nonfinite():
    nf, reason = is_nonfinite_state(float("nan"), ema_finite=True)
    assert nf is True and "loss non-finite" in reason


def test_loss_inf_is_nonfinite():
    nf, reason = is_nonfinite_state(float("inf"), ema_finite=True)
    assert nf is True and "loss non-finite" in reason


def test_ema_nonfinite_is_nonfinite():
    nf, reason = is_nonfinite_state(0.5, ema_finite=False)
    assert nf is True and reason == "EMA weights non-finite"


def test_grad_norm_nan_is_nonfinite():
    nf, reason = is_nonfinite_state(0.5, ema_finite=True, grad_norm=float("nan"))
    assert nf is True and "grad_norm non-finite" in reason


def test_grad_norm_none_is_ignored():
    assert is_nonfinite_state(0.5, ema_finite=True, grad_norm=None) == (False, "")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.train.stagea_hardening'`

- [ ] **Step 4: Implement the module skeleton + `is_nonfinite_state`**

Create `src/train/stagea_hardening.py`:

```python
"""Stage-A retrain numerical-hardening: pure decision/validation logic.

Pure and CPU-importable: NO top-level torch import. The only torch touch is a
LOCAL import inside validate_checkpoint_finite (the thin seder seam, same
posture as eval/robust_stage_a/run_gate.py:_infer). All decision logic
(is_nonfinite_state, should_halt, build_hardened_overrides) and the provenance
scan core (scan_state_for_nonfinite) are torch-free and CPU-unit-tested.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def is_nonfinite_state(
    loss: float | None,
    ema_finite: bool,
    grad_norm: float | None = None,
) -> tuple[bool, str]:
    """Detect a non-finite training state from plain scalars.

    The Ultralytics callback seam extracts float(trainer.loss), a precomputed
    bool 'is the EMA all-finite', and an optional grad-norm float, and passes
    them here. Returns (is_nonfinite, reason); reason == '' when finite.
    """
    if loss is None or math.isnan(loss) or math.isinf(loss):
        return (True, f"loss non-finite ({loss!r})")
    if not ema_finite:
        return (True, "EMA weights non-finite")
    if grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm)):
        return (True, f"grad_norm non-finite ({grad_norm!r})")
    return (False, "")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v`
Expected: 7 PASS

- [ ] **Step 6: Commit**

```bash
git add src/train/stagea_hardening.py tests/stagea_hardening/__init__.py \
        tests/stagea_hardening/test_stagea_hardening.py
git commit -m "feat(train): stage-a hardening — is_nonfinite_state pure detector"
git push origin main
```

---

## Task 2: `should_halt` (mirrors Stage-B halt contract)

**Files:**
- Modify: `src/train/stagea_hardening.py`
- Test: `tests/stagea_hardening/test_stagea_hardening.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/stagea_hardening/test_stagea_hardening.py`:

```python
from src.train.stagea_hardening import should_halt  # noqa: E402


def test_should_halt_when_nonfinite():
    msg, halt = should_halt(nonfinite=True, reason="EMA weights non-finite")
    assert halt is True
    assert msg == "stage-a halt: EMA weights non-finite"


def test_should_not_halt_when_finite():
    assert should_halt(nonfinite=False, reason="") == ("", False)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v -k should_halt`
Expected: FAIL — `ImportError: cannot import name 'should_halt'`

- [ ] **Step 3: Append `should_halt`**

Append to `src/train/stagea_hardening.py`:

```python
def should_halt(*, nonfinite: bool, reason: str) -> tuple[str, bool]:
    """Halt policy. Mirrors src/train/train.py:_should_sanity_halt's
    (message, should_halt) contract so post-mortem tooling can grep the
    message. Defense-in-depth policy: halt immediately on any non-finite
    state — do not let EMA corruption waste epochs (the epoch-34 incident
    wasted ~20 epochs before EarlyStopping)."""
    if nonfinite:
        return (f"stage-a halt: {reason}", True)
    return ("", False)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v`
Expected: all PASS (Task 1 + Task 2)

- [ ] **Step 5: Commit**

```bash
git add src/train/stagea_hardening.py tests/stagea_hardening/test_stagea_hardening.py
git commit -m "feat(train): stage-a hardening — should_halt policy (Stage-B contract)"
git push origin main
```

---

## Task 3: `build_hardened_overrides` (pinned recipe)

**Files:**
- Modify: `src/train/stagea_hardening.py`
- Test: `tests/stagea_hardening/test_stagea_hardening.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/stagea_hardening/test_stagea_hardening.py`:

```python
from src.train.stagea_hardening import (  # noqa: E402
    HardenedOverrides,
    build_hardened_overrides,
)


def test_hardened_overrides_pinned_defaults():
    o = build_hardened_overrides(amp=True)
    assert isinstance(o, HardenedOverrides)
    assert o.lr0 == 0.01 and o.lrf == 0.01
    assert o.save_period == 5
    assert o.max_grad_norm == 1.0
    assert o.amp is True


def test_hardened_overrides_amp_passthrough():
    assert build_hardened_overrides(amp=False).amp is False


def test_hardened_overrides_rejects_bad_save_period():
    with pytest.raises(ValueError, match="save_period"):
        build_hardened_overrides(amp=True, save_period=0)


def test_hardened_overrides_rejects_bad_max_grad_norm():
    with pytest.raises(ValueError, match="max_grad_norm"):
        build_hardened_overrides(amp=True, max_grad_norm=0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v -k hardened_overrides`
Expected: FAIL — `ImportError: cannot import name 'HardenedOverrides'`

- [ ] **Step 3: Append `HardenedOverrides` + `build_hardened_overrides`**

Append to `src/train/stagea_hardening.py`:

```python
@dataclass(frozen=True)
class HardenedOverrides:
    lr0: float
    lrf: float
    save_period: int
    amp: bool
    max_grad_norm: float


def build_hardened_overrides(
    *,
    amp: bool,
    save_period: int = 5,
    max_grad_norm: float = 1.0,
    lr0: float = 0.01,
    lrf: float = 0.01,
) -> HardenedOverrides:
    """Pinned hardened recipe (spec §Pinned decisions). lr0/lrf stay at the
    accuracy-validated 0.01 — stability goes into grad-clip + nan-guard + AMP
    posture + active halt, not LR perturbation. save_period=5 bounds
    mid-run-death waste to <=5 epochs. max_grad_norm=1.0 mirrors the proven
    Stage-B clip (src/train/train.py:2988)."""
    if save_period < 1:
        raise ValueError(f"save_period must be >= 1, got {save_period}")
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
    return HardenedOverrides(
        lr0=lr0, lrf=lrf, save_period=save_period,
        amp=amp, max_grad_norm=max_grad_norm,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/train/stagea_hardening.py tests/stagea_hardening/test_stagea_hardening.py
git commit -m "feat(train): stage-a hardening — pinned build_hardened_overrides"
git push origin main
```

---

## Task 4: `scan_state_for_nonfinite` (pure provenance core)

**Files:**
- Modify: `src/train/stagea_hardening.py`
- Test: `tests/stagea_hardening/test_stagea_hardening.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/stagea_hardening/test_stagea_hardening.py`:

```python
from src.train.stagea_hardening import scan_state_for_nonfinite  # noqa: E402


def test_scan_all_finite_is_ok():
    items = [("model.a", True), ("model.b", True), ("ema.a", True)]
    assert scan_state_for_nonfinite(items) == (True, 0, 3, None)


def test_scan_reports_first_offender_and_count():
    items = [("model.a", True), ("model.b", False),
             ("ema.a", False), ("ema.b", True)]
    ok, n, total, first = scan_state_for_nonfinite(items)
    assert ok is False
    assert n == 2
    assert total == 4
    assert first == "model.b"


def test_scan_empty_is_ok_zero():
    assert scan_state_for_nonfinite([]) == (True, 0, 0, None)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v -k scan`
Expected: FAIL — `ImportError: cannot import name 'scan_state_for_nonfinite'`

- [ ] **Step 3: Append `scan_state_for_nonfinite`**

Append to `src/train/stagea_hardening.py`:

```python
def scan_state_for_nonfinite(
    items: Iterable[tuple[str, bool]],
) -> tuple[bool, int, int, str | None]:
    """Pure provenance core. `items` = (tensor_name, is_all_finite) pairs.
    Returns (ok, n_nonfinite_tensors, total, first_offending_key);
    ok == (n_nonfinite == 0). The torch seam computes the per-tensor bool;
    this only counts, so it is fully CPU-testable without torch."""
    total = 0
    n_nonfinite = 0
    first_key: str | None = None
    for name, is_finite in items:
        total += 1
        if not is_finite:
            n_nonfinite += 1
            if first_key is None:
                first_key = name
    return (n_nonfinite == 0, n_nonfinite, total, first_key)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/train/stagea_hardening.py tests/stagea_hardening/test_stagea_hardening.py
git commit -m "feat(train): stage-a hardening — pure scan_state_for_nonfinite"
git push origin main
```

---

## Task 5: `validate_checkpoint_finite` (thin torch seam) + CPU import contract

**Files:**
- Modify: `src/train/stagea_hardening.py`
- Test: `tests/stagea_hardening/test_stagea_hardening.py` (append)

- [ ] **Step 1: Append the failing test (CPU import/contract only — no torch)**

Append to `tests/stagea_hardening/test_stagea_hardening.py`:

```python
import inspect  # noqa: E402

from src.train import stagea_hardening as _sh  # noqa: E402


def test_validate_checkpoint_finite_is_importable_without_torch():
    # The module must import on CPU with no torch installed; the torch
    # import must be LOCAL to validate_checkpoint_finite (seam posture,
    # mirrors run_gate._infer). Assert the symbol exists and that no
    # top-level `import torch` exists in the module source.
    assert hasattr(_sh, "validate_checkpoint_finite")
    assert callable(_sh.validate_checkpoint_finite)
    src = inspect.getsource(_sh)
    module_head = src.split("def validate_checkpoint_finite", 1)[0]
    assert "\nimport torch" not in module_head
    assert "    import torch" in inspect.getsource(_sh.validate_checkpoint_finite)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/stagea_hardening/test_stagea_hardening.py -v -k validate_checkpoint_finite`
Expected: FAIL — `AttributeError: module 'src.train.stagea_hardening' has no attribute 'validate_checkpoint_finite'`

- [ ] **Step 3: Append `validate_checkpoint_finite`**

Append to `src/train/stagea_hardening.py`:

```python
def validate_checkpoint_finite(
    path: str,
) -> tuple[bool, int, int, str | None]:
    """Thin torch seam (NOT unit-tested locally — torch lives on seder's
    venv-cu132; same posture as eval/robust_stage_a/run_gate.py:_infer).
    Loads an Ultralytics checkpoint on CPU and scans every tensor in the
    'model' and 'ema' state_dicts for NaN/Inf, delegating the count to the
    pure scan_state_for_nonfinite. Returns
    (ok, n_nonfinite_tensors, total, first_offending_key)."""
    import torch  # local: keeps the module CPU-importable without torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    def _named_tensors():
        for ckpt_key in ("model", "ema"):
            obj = ckpt.get(ckpt_key) if isinstance(ckpt, dict) else None
            if obj is None:
                continue
            sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
            for tname, t in sd.items():
                if hasattr(t, "isfinite"):
                    yield (
                        f"{ckpt_key}.{tname}",
                        bool(torch.isfinite(t).all().item()),
                    )

    return scan_state_for_nonfinite(_named_tensors())
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `python3 -m pytest tests/stagea_hardening/ -v`
Expected: all PASS, 0 skipped (CPU, not CUDA-gated).

- [ ] **Step 5: Commit**

```bash
git add src/train/stagea_hardening.py tests/stagea_hardening/test_stagea_hardening.py
git commit -m "feat(train): stage-a hardening — validate_checkpoint_finite torch seam"
git push origin main
```

---

## Task 6: Wire the thin seam into `scripts/train_yolo.py`

`scripts/train_yolo.py` is the GPU/seder seam: it adds the hardened flags, makes `_patch_nan_guard` honor a configured `max_norm`, registers the active halt-on-NaN callback, and passes the hardened overrides into `model.train()`. Not CPU-unit-tested (drives Ultralytics); verified by an arg-wiring smoke + a no-torch import check + code review — same posture as Sub-plan A Task 6.

**Files:**
- Modify: `scripts/train_yolo.py`

- [ ] **Step 1: Add the three hardened flags to `parse_args`**

In `scripts/train_yolo.py`, immediately after the `--noise-warmup-steps` argument block (currently ending at line 75, before `return parser.parse_args()` at line 76), insert:

```python
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Explicit gradient-clip max_norm (overrides Ultralytics' opaque "
             "internal value). Mirrors the proven Stage-B clip=1.0. Only "
             "applied when --nan-guard is on (it owns the clip_grad_norm_ hook).",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="Save a checkpoint every N epochs (enables resume-from-last on a "
             "mid-run death; bounds wasted work). Default 5.",
    )
    parser.add_argument(
        "--debug-anomaly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.autograd.set_detect_anomaly (slow; off by default). "
             "Use only to capture a recurring NaN's origin.",
    )
```

- [ ] **Step 2: Make `_patch_nan_guard` honor a configured max_norm**

Replace the `_patch_nan_guard` signature and the `safe_clip` body (currently lines 79–114) so the hook can override Ultralytics' max_norm. Replace:

```python
def _patch_nan_guard() -> None:
```
with:
```python
def _patch_nan_guard(max_grad_norm: float | None = None) -> None:
```

And replace the `safe_clip` function (currently lines 100–112) with:

```python
    def safe_clip(parameters, max_norm, *args, **kwargs):
        # Materialize parameters so we can iterate twice (NaN scan + original clip)
        params = list(parameters) if not isinstance(parameters, list) else parameters
        any_nan = False
        for p in params:
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    p.grad.zero_()
                    any_nan = True
        if any_nan:
            nan_event_count[0] += 1
            print(f"[nan-guard] zeroed NaN/Inf grads (occurrence #{nan_event_count[0]})")
        effective_max_norm = max_grad_norm if max_grad_norm is not None else max_norm
        return original_clip(params, effective_max_norm, *args, **kwargs)
```

- [ ] **Step 3: Add the active halt-on-NaN callback registrar**

In `scripts/train_yolo.py`, immediately after `_patch_nan_guard` (after its final line `torch.nn.utils.clip_grad_norm_ = safe_clip`, currently line 114), add:

```python
def _register_stagea_hardening(model, *, debug_anomaly: bool) -> None:
    """Register the active halt-on-NaN callback (the thin seam). Feeds plain
    scalars into the pure src.train.stagea_hardening logic; on a non-finite
    state it attempts an explicit checkpoint save and sets trainer.stop so the
    run does not waste epochs (the epoch-34 incident wasted ~20). If the EMA is
    already non-finite, trainer.save_model() will not write a good EMA — the
    last clean periodic/best checkpoint (from --save-period) is the last-good;
    the acceptance runbook gates that checkpoint through validate_checkpoint_finite."""
    import torch
    from src.train.stagea_hardening import is_nonfinite_state, should_halt

    if debug_anomaly:
        torch.autograd.set_detect_anomaly(True)

    def _ema_finite(trainer) -> bool:
        ema = getattr(trainer, "ema", None)
        m = getattr(ema, "ema", None) if ema is not None else None
        if m is None:
            return True
        for t in m.state_dict().values():
            if hasattr(t, "isfinite") and not bool(torch.isfinite(t).all().item()):
                return False
        return True

    def _check(trainer) -> None:
        loss = getattr(trainer, "loss", None)
        if loss is None:
            loss_f = None
        elif hasattr(loss, "item"):
            try:
                loss_f = float(loss.item())
            except Exception:
                loss_f = None
        else:
            loss_f = float(loss)
        nf, reason = is_nonfinite_state(loss_f, _ema_finite(trainer))
        msg, halt = should_halt(nonfinite=nf, reason=reason)
        if halt:
            print(f"[stagea-hardening] {msg} at epoch "
                  f"{getattr(trainer, 'epoch', '?')}; saving + halting")
            try:
                trainer.save_model()
            except Exception as exc:  # never let the guard itself crash the run
                print(f"[stagea-hardening] save_model raised: {exc!r}")
            trainer.stop = True

    model.add_callback("on_train_batch_end", _check)
    model.add_callback("on_fit_epoch_end", _check)
```

- [ ] **Step 4: Wire flags + overrides + callback into `main`**

In `main` (currently lines 119–147), replace the block from `if args.nan_guard:` through `model.train(**train_kwargs)` with:

```python
    if args.nan_guard:
        _patch_nan_guard(max_grad_norm=args.max_grad_norm)
    if args.noise:
        from src.train.scan_noise import patch_albumentations_for_scan_noise
        patch_albumentations_for_scan_noise(warmup_steps=args.noise_warmup_steps)
    from src.train.stagea_hardening import build_hardened_overrides
    hardened = build_hardened_overrides(
        amp=args.amp, save_period=args.save_period,
        max_grad_norm=args.max_grad_norm,
    )
    model = YOLO(args.model)
    _register_stagea_hardening(model, debug_anomaly=args.debug_anomaly)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        rect=True,
        cos_lr=True,
        name=args.name,
        project=args.project,
        hsv_h=0, hsv_s=0,
        flipud=0, fliplr=0,
        mosaic=0, mixup=0,
        save=True,
        save_period=hardened.save_period,
        lr0=hardened.lr0,
        lrf=hardened.lrf,
        patience=args.patience,
        workers=args.workers,
        amp=hardened.amp,
    )
    if args.compile:
        train_kwargs["compile"] = True
    model.train(**train_kwargs)
```

- [ ] **Step 5: Smoke the CLI arg wiring (no GPU, torch may be absent)**

Run: `python3 scripts/train_yolo.py --help`
Expected: argparse help prints and lists `--max-grad-norm`, `--save-period`, `--debug-anomaly` alongside the existing args; exit 0. (Imports at module top are only `argparse`/`sys`/`pathlib`/`ultralytics`; `torch` and `src.train.stagea_hardening` are imported lazily inside functions, so `--help` must not require torch. If `ultralytics` import fails locally, instead run `python3 -c "import ast; ast.parse(open('scripts/train_yolo.py').read()); print('syntax OK')"` and note ultralytics is seder-only.)

- [ ] **Step 6: Re-run the pure suite (unchanged, must stay green)**

Run: `python3 -m pytest tests/stagea_hardening/ -q`
Expected: all PASS (Task 6 only touches the seam; the pure module is unchanged).

- [ ] **Step 7: Commit**

```bash
git add scripts/train_yolo.py
git commit -m "feat(train): wire stage-a hardening seam (flags, max_norm, halt callback, overrides)"
git push origin main
```

---

## Task 7: Codified seder hardened-retrain worker

A reusable PowerShell worker codifying the validated `C:\radio_jobs\*.ps1` schtasks pattern + the four minimal-reliable fixes. Not unit-tested (seder-only); verified by structure review + the acceptance run.

**Files:**
- Create: `scripts/seder/run_stagea_hardened_retrain.ps1`

- [ ] **Step 1: Create the worker script**

Create `scripts/seder/run_stagea_hardened_retrain.ps1`:

```powershell
<#
Stage-A hardened retrain worker (seder). Codifies the validated schtasks
pattern + the four Sub-plan-B minimal-reliable fixes:
  1. UTF-8 logging (PYTHONIOENCODING + -Encoding utf8)
  2. stderr non-fatal: $ErrorActionPreference='Continue', gate on $LASTEXITCODE
  3. save_period (passed via --save-period; default 5)
  4. resume-from-last: relaunch with --resume if last.pt exists

Deploy to a no-space path (e.g. C:\radio_jobs\run_stagea_hardened_retrain.ps1)
and register:
  schtasks /create /tn radio_stagea_hardened /sc ONCE /st 23:59 /f ^
    /tr "powershell -ExecutionPolicy Bypass -File C:\radio_jobs\run_stagea_hardened_retrain.ps1"
  schtasks /run /tn radio_stagea_hardened
Poll: schtasks /query /tn radio_stagea_hardened  +  the .done/.failed markers.
#>
param(
    [string]$Repo       = "$env:USERPROFILE\Clarity-OMR-Train-RADIO",
    [string]$Venv       = "venv-cu132",
    [string]$Data       = "data/processed/mixed_systems_v1/data.yaml",
    [string]$Model      = "yolo26m.pt",
    [string]$Name       = "yolo26m_systems_hardened",
    [int]   $Epochs     = 100,
    [int]   $Batch      = 4,
    [int]   $SavePeriod = 5,
    [string]$JobTag     = "radio_stagea_hardened"
)

$ErrorActionPreference = "Continue"          # fix #2: child stderr is NOT fatal
$env:PYTHONIOENCODING  = "utf-8"             # fix #1: UTF-8 stdio

Set-Location $Repo
$py        = Join-Path $Repo "$Venv\Scripts\python.exe"
$logDir    = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logOut    = Join-Path $logDir "$JobTag.out.log"
$logErr    = Join-Path $logDir "$JobTag.err.log"
$doneMark  = Join-Path $logDir "$JobTag.done"
$failMark  = Join-Path $logDir "$JobTag.failed"
Remove-Item $doneMark,$failMark -ErrorAction SilentlyContinue

# fix #4: resume-from-last if a checkpoint exists for this run name
$lastPt = Join-Path $Repo "runs\detect\runs\$Name\weights\last.pt"
$modelArg = $Model
$resumeArg = @()
if (Test-Path $lastPt) {
    $modelArg  = $lastPt
    $resumeArg = @("--resume")
    "[$JobTag] resuming from $lastPt" | Out-File -Encoding utf8 -Append $logOut
}

$argList = @(
    "scripts/train_yolo.py",
    "--model", $modelArg,
    "--data", $Data,
    "--name", $Name,
    "--project", "runs/detect/runs",
    "--epochs", $Epochs,
    "--batch", $Batch,
    "--device", "0",
    "--amp", "--nan-guard",
    "--noise", "--noise-warmup-steps", "2000",
    "--max-grad-norm", "1.0",
    "--save-period", $SavePeriod
) + $resumeArg

"[$JobTag] launching: $py $($argList -join ' ')" |
    Out-File -Encoding utf8 -Append $logOut

# fix #1+#2: capture both streams to UTF-8 logs; success gates on exit code only
& $py @argList *>> $logOut 2>> $logErr
$code = $LASTEXITCODE

if ($code -eq 0) {
    "[$JobTag] DONE exit=0" | Out-File -Encoding utf8 -Append $logOut
    New-Item -ItemType File -Force -Path $doneMark | Out-Null
} else {
    "[$JobTag] FAILED exit=$code" | Out-File -Encoding utf8 -Append $logErr
    New-Item -ItemType File -Force -Path $failMark | Out-Null
}
exit $code
```

- [ ] **Step 2: Static-validate the script (no seder needed)**

Run: `python3 -c "p=open('scripts/seder/run_stagea_hardened_retrain.ps1').read(); assert 'ErrorActionPreference = \"Continue\"' in p and 'PYTHONIOENCODING' in p and '--save-period' in p and '--resume' in p and 'Out-File -Encoding utf8' in p; print('worker contract OK')"`
Expected: `worker contract OK` (asserts all four fixes are present in the committed script).

- [ ] **Step 3: Commit**

```bash
git add scripts/seder/run_stagea_hardened_retrain.ps1
git commit -m "feat(seder): codified hardened-retrain worker (utf8, stderr-safe, save_period, resume)"
git push origin main
```

---

## Task 8: Acceptance runbook (the dual-gate integration test definition)

Documentation task — defines the exact, reproducible acceptance procedure. No code; every command is concrete (writing-plans: no placeholders).

**Files:**
- Create: `docs/superpowers/plans/2026-05-17-robust-stage-a-subplan-b-ACCEPTANCE.md`

- [ ] **Step 1: Write the runbook**

Create `docs/superpowers/plans/2026-05-17-robust-stage-a-subplan-b-ACCEPTANCE.md`:

```markdown
# Sub-plan B — Acceptance Runbook (graceful-safe + no-regression)

Run ON SEDER (`venv-cu132`). B PASSES iff: the hardened retrain either
completes clean OR the active guard halts it; AND the selected checkpoint
passes the provenance gate (0 non-finite tensors); AND it meets lieder
recall >= 0.930. Do NOT overwrite the validated faint-ink best.pt
(runs/detect/runs/yolo26m_systems_faintink/weights/best.pt).

## 1. Launch the hardened retrain (codified worker)

Deploy + register the worker (no-space path avoids schtasks quoting):

    copy scripts\seder\run_stagea_hardened_retrain.ps1 C:\radio_jobs\
    schtasks /create /tn radio_stagea_hardened /sc ONCE /st 23:59 /f ^
      /tr "powershell -ExecutionPolicy Bypass -File C:\radio_jobs\run_stagea_hardened_retrain.ps1"
    schtasks /run /tn radio_stagea_hardened

Poll: `schtasks /query /tn radio_stagea_hardened` + `logs\radio_stagea_hardened.done|.failed`.
Watch `logs\radio_stagea_hardened.out.log` for `[nan-guard]` / `[stagea-hardening]` lines.

## 2. Select the checkpoint to gate

- Clean completion -> `runs/detect/runs/yolo26m_systems_hardened/weights/best.pt`
- Active-halt fired -> the newest finite checkpoint: prefer `best.pt`; if its
  provenance fails, fall back to the highest-epoch `--save-period` checkpoint
  under `runs/detect/runs/yolo26m_systems_hardened/weights/`.

## 3. Provenance gate (mandatory, blocks scoring)

    venv-cu132\Scripts\python.exe -c "from src.train.stagea_hardening import validate_checkpoint_finite as v; ok,n,t,k=v(r'runs/detect/runs/yolo26m_systems_hardened/weights/best.pt'); print(ok,n,t,k); import sys; sys.exit(0 if ok else 1)"

A non-zero exit / `ok=False` is an automatic B-FAIL regardless of metrics.

## 4. Lieder no-regression gate (>= 0.930)

Two-step (same scorer Sub-plan A reads):

    venv-cu132\Scripts\python.exe eval/run_stage_a_only.py ^
      --pdf-dir data/openscore_lieder/pdfs ^
      --yolo-weights runs/detect/runs/yolo26m_systems_hardened/weights/best.pt ^
      --out-dir eval/results/stagea_manifests/hardened

    venv-cu132\Scripts\python.exe eval/score_stage_a_only.py ^
      --manifest-dir eval/results/stagea_manifests/hardened ^
      --scores-dir data/openscore_lieder/scores ^
      --out-csv eval/results/stagea_hardened.csv

Compute aggregate recall with the Sub-plan-A reader and compare to baseline:

    venv-cu132\Scripts\python.exe -c "from eval.robust_stage_a.gate import recall_from_stagea_csv as r; new=r('eval/results/stagea_hardened.csv'); base=r('eval/results/stagea_baseline_pre_faintink.csv'); print('new',new,'base',base,'PASS' if new>=base else 'FAIL')"

## 5. Verdict

B PASS iff: training ended clean-or-guard-halted AND step 3 ok=True AND
step 4 prints PASS (new >= base, base = 0.930). Record the outcome + the
[nan-guard]/[stagea-hardening] log evidence in the Sub-plan-B handoff and
update memory `project_radio_robust_stagea`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-05-17-robust-stage-a-subplan-b-ACCEPTANCE.md
git commit -m "docs(plan): Sub-plan B acceptance runbook (provenance + >=0.930 dual gate)"
git push origin main
```

---

## Self-review

- **Spec coverage:** spec §"Defense-in-depth guards" → Task 1 (`is_nonfinite_state`), Task 2 (`should_halt`), Task 3 (`build_hardened_overrides`: `--nan-guard` on, explicit `max_grad_norm=1.0`, pinned `lr0/lrf=0.01`, AMP passthrough, `save_period=5`), Task 6 (active halt-on-NaN callback on `on_train_batch_end`+`on_fit_epoch_end`; `--debug-anomaly` slow path; explicit-max_norm nan-guard). spec §"best.pt provenance validation" → Task 4 (pure `scan_state_for_nonfinite`) + Task 5 (`validate_checkpoint_finite` torch seam) + Task 8 step 3 (mandatory gate). spec §"Seder-worker minimal-reliable hardening" → Task 7 (UTF-8, stderr/$LASTEXITCODE, save_period, resume-from-last; no watchdog). spec §"Testing strategy" → Tasks 1–5 CPU-tested in `tests/stagea_hardening/` (not CUDA-gated), Task 6 seam smoke, Task 8 = the seder acceptance integration test. spec §"Success criteria" (graceful-safe + no-regression) → Task 8 steps 3–5 dual gate. spec §"Non-goals" honored: no NaN bisect (defense-in-depth only), no watchdog (Task 7 explicitly excludes), no accuracy chasing (Task 8 gate is ≥baseline only), current `mixed_systems_v1` data (Task 7 default `--data`), not C/D/Phase-3/text-class.
- **Placeholder scan:** no TBD/TODO; every code step is complete and paste-able; the spec-refinement (provenance split) is stated explicitly with rationale, not deferred; all pinned values concrete (`max_grad_norm=1.0`, `lr0=lrf=0.01`, `save_period=5`, AMP on, `--amp`-off is a runbook contingency not a default).
- **Type consistency:** `is_nonfinite_state(loss, ema_finite, grad_norm) -> (bool,str)`, `should_halt(*, nonfinite, reason) -> (str,bool)` (Stage-B contract), `HardenedOverrides(lr0,lrf,save_period,amp,max_grad_norm)` + `build_hardened_overrides(*, amp, save_period=5, max_grad_norm=1.0, lr0=0.01, lrf=0.01)`, `scan_state_for_nonfinite(items)->(ok,n,total,first)`, `validate_checkpoint_finite(path)->(ok,n,total,first)` — names/signatures identical across the module, the train_yolo.py seam (Task 6 calls `build_hardened_overrides(amp=,save_period=,max_grad_norm=)`, `_patch_nan_guard(max_grad_norm=)`, `is_nonfinite_state`/`should_halt` in `_check`), and the Task 8 runbook (`validate_checkpoint_finite`, `recall_from_stagea_csv`). `_patch_nan_guard` gains an optional `max_grad_norm` kwarg (back-compatible default None = prior behavior).
