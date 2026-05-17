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


from src.train.stagea_hardening import should_halt  # noqa: E402


def test_should_halt_when_nonfinite():
    msg, halt = should_halt(nonfinite=True, reason="EMA weights non-finite")
    assert halt is True
    assert msg == "stage-a halt: EMA weights non-finite"


def test_should_not_halt_when_finite():
    assert should_halt(nonfinite=False, reason="") == ("", False)
