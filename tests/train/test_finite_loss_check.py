"""Verify the finite-loss check runs once per accumulation window, not per micro-batch."""
import pytest
import torch


def test_finite_loss_or_accumulates_across_window():
    """The accumulator must be a tensor we OR into; the .item() check happens
    only at the opt-step boundary."""
    from src.train.train import _step_window_corrupted

    prior = torch.tensor(False)
    finite_loss = torch.tensor(1.5)
    nonfinite_loss = torch.tensor(float("nan"))

    # Clean window: stays clean.
    state = _step_window_corrupted(prior, finite_loss)
    assert state.dtype == torch.bool
    assert bool(state.item()) is False

    # Hit non-finite: flips to True.
    state = _step_window_corrupted(state, nonfinite_loss)
    assert bool(state.item()) is True

    # Once corrupted, stays corrupted within the window.
    state = _step_window_corrupted(state, finite_loss)
    assert bool(state.item()) is True


def test_finite_loss_helper_keeps_state_on_device():
    """The helper must NOT call .item() internally — confirmed by the return
    type still being a torch.Tensor (not a Python bool)."""
    from src.train.train import _step_window_corrupted

    prior = torch.tensor(False)
    loss = torch.tensor(2.0)
    out = _step_window_corrupted(prior, loss)
    assert isinstance(out, torch.Tensor), "Helper must keep state on device"


def test_finite_loss_helper_handles_inf():
    """torch.isfinite returns False for ±inf; helper must catch this too."""
    from src.train.train import _step_window_corrupted

    prior = torch.tensor(False)
    inf_loss = torch.tensor(float("inf"))
    state = _step_window_corrupted(prior, inf_loss)
    assert bool(state.item()) is True


def test_finite_loss_helper_works_with_scalar_tensor_loss():
    """Real losses are 0-d scalar tensors. Verify .all() works on those."""
    from src.train.train import _step_window_corrupted

    prior = torch.tensor(False)
    # 0-d scalar (typical loss)
    scalar_loss = torch.tensor(1.0)
    state = _step_window_corrupted(prior, scalar_loss)
    assert bool(state.item()) is False
