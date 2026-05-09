"""Sanity halt fires on val_loss > 5.0 (first 200 steps) OR NaN (any time)."""
from __future__ import annotations
import math
import pytest


def test_sanity_halt_returns_true_on_val_loss_above_5_in_first_200_steps():
    from src.train.train import _should_sanity_halt

    assert _should_sanity_halt(val_loss=6.0, global_step=100) == ("val_loss>5 in first 200 steps", True)


def test_sanity_halt_returns_false_on_val_loss_above_5_after_200_steps():
    from src.train.train import _should_sanity_halt

    msg, halt = _should_sanity_halt(val_loss=6.0, global_step=300)
    assert halt is False


def test_sanity_halt_returns_true_on_nan_at_any_step():
    from src.train.train import _should_sanity_halt

    msg, halt = _should_sanity_halt(val_loss=math.nan, global_step=100)
    assert halt is True
    msg, halt = _should_sanity_halt(val_loss=math.nan, global_step=10000)
    assert halt is True


def test_sanity_halt_returns_false_on_normal_loss():
    from src.train.train import _should_sanity_halt

    msg, halt = _should_sanity_halt(val_loss=0.3, global_step=100)
    assert halt is False
    msg, halt = _should_sanity_halt(val_loss=4.99, global_step=199)
    assert halt is False
