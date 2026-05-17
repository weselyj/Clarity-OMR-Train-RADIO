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
