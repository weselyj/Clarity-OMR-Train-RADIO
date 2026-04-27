"""Tests for Item A: --diag-cadence gating (Tier 1 #5).

These tests exercise the _should_run_diagnostics predicate only — no GPU,
no model construction, no training loop.  They run on CPU and require only
the standard library (sys/pathlib) plus the module under test.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when run directly or via pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.train import _should_run_diagnostics


class TestShouldRunDiagnostics:
    """Unit tests for the _should_run_diagnostics(optimizer_step, cadence) predicate."""

    # --- cadence=1 (legacy behavior) ---

    def test_cadence_1_always_true(self):
        """With cadence=1, every step is a diagnostic step."""
        for step in range(1, 101):
            assert _should_run_diagnostics(step, 1), f"expected True at step {step}"

    # --- cadence=25 (default) ---

    def test_cadence_25_fires_on_step_1(self):
        """First optimizer step is always a diagnostic step (catch early non-finites)."""
        assert _should_run_diagnostics(1, 25)

    def test_cadence_25_skips_non_multiples(self):
        """Non-cadence steps (2..24) must be skipped."""
        for step in range(2, 25):
            assert not _should_run_diagnostics(step, 25), (
                f"expected False at step {step} with cadence=25"
            )

    def test_cadence_25_fires_on_multiples(self):
        """Steps that are exact multiples of cadence fire."""
        for step in [25, 50, 75, 100, 4675]:
            assert _should_run_diagnostics(step, 25), (
                f"expected True at step {step} with cadence=25"
            )

    def test_cadence_25_skips_between_multiples(self):
        """Non-cadence, non-first steps between multiples must be skipped."""
        skip_steps = [26, 27, 49, 51, 99]
        for step in skip_steps:
            assert not _should_run_diagnostics(step, 25), (
                f"expected False at step {step} with cadence=25"
            )

    # --- cadence=0 or negative (clamped to 1 by callers, but predicate itself) ---

    def test_cadence_zero_treated_as_always_on(self):
        """cadence <= 1 always returns True (defensive; callers should clamp)."""
        for step in range(1, 10):
            assert _should_run_diagnostics(step, 0)

    def test_cadence_negative_treated_as_always_on(self):
        assert _should_run_diagnostics(5, -1)

    # --- arbitrary cadence values ---

    def test_cadence_10_pattern(self):
        """Verify fire/skip pattern for cadence=10 over 30 steps."""
        for step in range(1, 31):
            expected = (step == 1) or (step % 10 == 0)
            result = _should_run_diagnostics(step, 10)
            assert result == expected, (
                f"cadence=10, step={step}: expected {expected}, got {result}"
            )

    def test_reduction_ratio(self):
        """With cadence=25, confirm ~25x reduction over 4688 steps."""
        cadence = 25
        total_steps = 4688
        diag_count = sum(
            1 for s in range(1, total_steps + 1) if _should_run_diagnostics(s, cadence)
        )
        # Step 1 + every 25th: 1 + 4688//25 = 1 + 187 = 188 (if 4688 % 25 == 0: 188+1=189)
        # Exact count isn't critical; what matters is roughly 1/25 of total.
        ratio = total_steps / diag_count
        assert ratio >= 20, f"expected >=20x reduction, got {ratio:.1f}x"
        assert ratio <= 30, f"expected <=30x reduction, got {ratio:.1f}x"
