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
