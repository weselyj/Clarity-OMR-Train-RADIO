"""Test that loss.backward() is skipped on corrupt (non-finite loss) micro-batches.

Issue #6: The OR-accumulator (accum_corruption) was updated BEFORE
loss.backward() was called.  If micro-batch 3 of an 8-step window produced a
NaN loss, micro-batches 4-7 still ran backward() through the NaN-gradient
state, wasting GPU time and risking CUDA errors on some architectures.

The fix re-adds an early skip: if loss is non-finite, set the corruption flag
and skip backward() for that micro-batch.  The boundary logic (zero_grad /
skip optimizer step) remains driven by the accumulated flag.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


def test_finite_loss_does_not_skip_backward():
    """When loss is finite, backward() MUST be called (normal path unaffected)."""
    from src.train.train import _should_skip_backward_on_corrupt

    finite_loss = torch.tensor(1.5)
    assert _should_skip_backward_on_corrupt(finite_loss) is False, (
        "Finite loss must NOT trigger early skip"
    )


def test_nan_loss_triggers_skip_backward():
    """When loss is NaN, backward() must NOT be called (early skip)."""
    from src.train.train import _should_skip_backward_on_corrupt

    nan_loss = torch.tensor(float("nan"))
    assert _should_skip_backward_on_corrupt(nan_loss) is True, (
        "NaN loss must trigger early skip"
    )


def test_inf_loss_triggers_skip_backward():
    """When loss is +inf, backward() must NOT be called."""
    from src.train.train import _should_skip_backward_on_corrupt

    inf_loss = torch.tensor(float("inf"))
    assert _should_skip_backward_on_corrupt(inf_loss) is True, (
        "+inf loss must trigger early skip"
    )


def test_neg_inf_loss_triggers_skip_backward():
    """When loss is -inf, backward() must NOT be called."""
    from src.train.train import _should_skip_backward_on_corrupt

    neg_inf_loss = torch.tensor(float("-inf"))
    assert _should_skip_backward_on_corrupt(neg_inf_loss) is True, (
        "-inf loss must trigger early skip"
    )


def test_corruption_flag_set_even_when_backward_skipped():
    """Even if backward() is skipped on a corrupt micro-batch, accum_corruption
    must still be updated to True so the boundary logic discards the window.

    We simulate the training loop's accumulator update + skip-backward logic.
    """
    from src.train.train import _step_window_corrupted, _should_skip_backward_on_corrupt

    accum_corruption = torch.zeros((), dtype=torch.bool)
    nan_loss = torch.tensor(float("nan"))

    # This is the order the fixed code must use:
    # 1. Update accumulator (corruption flag set even if we skip backward).
    accum_corruption = _step_window_corrupted(accum_corruption, nan_loss)

    # 2. Early skip check.
    should_skip = _should_skip_backward_on_corrupt(nan_loss)

    assert should_skip is True, "Should skip backward for NaN loss"
    assert bool(accum_corruption.item()) is True, (
        "Corruption flag must be set even when backward() is skipped"
    )


def test_corruption_flag_persists_for_subsequent_finite_microbatches():
    """Once corruption is set in a window, finite micro-batches after the
    corrupt one must NOT clear it — the whole window is discarded.
    """
    from src.train.train import _step_window_corrupted, _should_skip_backward_on_corrupt

    accum_corruption = torch.zeros((), dtype=torch.bool)

    # Micro-batch 1: finite — no skip.
    loss1 = torch.tensor(0.8)
    accum_corruption = _step_window_corrupted(accum_corruption, loss1)
    assert _should_skip_backward_on_corrupt(loss1) is False
    assert bool(accum_corruption.item()) is False

    # Micro-batch 2: NaN — skip backward, set flag.
    loss2 = torch.tensor(float("nan"))
    accum_corruption = _step_window_corrupted(accum_corruption, loss2)
    assert _should_skip_backward_on_corrupt(loss2) is True
    assert bool(accum_corruption.item()) is True

    # Micro-batch 3: finite — but corruption flag must stay True.
    loss3 = torch.tensor(0.5)
    accum_corruption = _step_window_corrupted(accum_corruption, loss3)
    assert _should_skip_backward_on_corrupt(loss3) is False  # loss itself OK
    assert bool(accum_corruption.item()) is True, (
        "Corruption flag must remain set for subsequent finite micro-batches"
    )
