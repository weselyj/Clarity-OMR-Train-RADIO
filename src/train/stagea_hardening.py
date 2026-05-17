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


def should_halt(*, nonfinite: bool, reason: str) -> tuple[str, bool]:
    """Halt policy. Mirrors src/train/train.py:_should_sanity_halt's
    (message, should_halt) contract so post-mortem tooling can grep the
    message. Defense-in-depth policy: halt immediately on any non-finite
    state — do not let EMA corruption waste epochs (the epoch-34 incident
    wasted ~20 epochs before EarlyStopping)."""
    if nonfinite:
        return (f"stage-a halt: {reason}", True)
    return ("", False)


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
