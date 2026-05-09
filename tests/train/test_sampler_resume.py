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
