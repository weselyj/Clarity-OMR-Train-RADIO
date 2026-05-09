"""Sampler resume: rebuild same list, skip consumed prefix."""
from __future__ import annotations
import torch
import pytest


def test_save_checkpoint_persists_last_batch_idx():
    """_save_checkpoint persists last_batch_idx in the payload when given."""
    from src.train.train import _save_checkpoint
    import tempfile
    from pathlib import Path

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_checkpoint(
            checkpoint_dir=Path(tmpdir),
            model=model, optimizer=optimizer,
            stage_name="stage3-test",
            global_step=500,
            stage_step=500,
            best_val_loss=0.3,
            last_batch_idx=123,
        )
        payload = torch.load(path, map_location="cpu")
    assert payload["last_batch_idx"] == 123


def test_tier_grouped_batch_sampler_skips_prefix_on_set_start_idx():
    from src.train.train import _TierGroupedBatchSampler

    batches = [[1, 2], [3, 4], [5, 6], [7, 8]]
    sampler = _TierGroupedBatchSampler(batches)
    sampler.set_start_idx(2)
    assert list(sampler) == [[5, 6], [7, 8]]
    assert len(sampler) == 2


def test_tier_grouped_batch_sampler_set_start_idx_out_of_range():
    from src.train.train import _TierGroupedBatchSampler
    sampler = _TierGroupedBatchSampler([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        sampler.set_start_idx(10)
    with pytest.raises(ValueError):
        sampler.set_start_idx(-1)


def test_resync_batch_idx_after_rebuild_returns_sampler_start_idx():
    """After a StopIteration rebuild, _batch_idx_consumed must be reset to the
    rebuilt iterator's actual position (== sampler._start_idx) so the loop's
    standard `_batch_idx_consumed += 1` ends up matching the next un-consumed
    micro-batch.

    Without the resync, _batch_idx_consumed continues counting from its
    pre-rebuild value, inflating the next checkpoint's last_batch_idx and
    causing future resumes to skip too many batches via set_start_idx.
    """
    from src.train.train import _TierGroupedBatchSampler, _resync_batch_idx_after_rebuild

    sampler = _TierGroupedBatchSampler([[i] for i in range(10)])
    sampler.set_start_idx(2)
    # Iterator yields batches starting at index 2; helper returns 2 so the
    # loop's `_batch_idx_consumed += 1` afterwards leaves it at 3 — the
    # position of the next un-consumed batch.
    assert _resync_batch_idx_after_rebuild(sampler) == 2

    sampler_default = _TierGroupedBatchSampler([[i] for i in range(4)])
    # Default _start_idx is 0 (no set_start_idx call).
    assert _resync_batch_idx_after_rebuild(sampler_default) == 0


def test_resync_batch_idx_after_rebuild_defaults_to_zero_without_start_idx():
    """Defensive: samplers without a _start_idx attribute (legacy/non-tier-grouped
    path) yield 0. _batch_idx_consumed is irrelevant in those modes (last_batch_idx
    is None on save) but the helper must not crash."""
    from src.train.train import _resync_batch_idx_after_rebuild

    class FakeSampler:
        pass

    assert _resync_batch_idx_after_rebuild(FakeSampler()) == 0
