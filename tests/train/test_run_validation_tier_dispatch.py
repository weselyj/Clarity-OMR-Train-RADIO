"""_run_validation handles cached and live batches transparently."""
from __future__ import annotations
from unittest.mock import MagicMock

import torch
import pytest


def _make_mixed_val_loader(n_cached: int = 1, n_live: int = 1):
    """A simple iterable that yields a mix of cached and live collated batches."""
    cached = {
        "tier": "cached",
        "encoder_hidden": torch.zeros(2, 156, 1280, dtype=torch.bfloat16),
        "_h16": 16,
        "_w16": 156,
        "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
        "labels": torch.zeros(2, 511, dtype=torch.long),
        "contour_targets": torch.zeros(2, 32, dtype=torch.long),
    }
    live = {
        "tier": "live",
        "images": torch.zeros(2, 1, 250, 2500, dtype=torch.float32),
        "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
        "labels": torch.zeros(2, 511, dtype=torch.long),
        "contour_targets": torch.zeros(2, 32, dtype=torch.long),
        "content_widths": torch.tensor([2500, 2500], dtype=torch.long),
    }
    return [cached] * n_cached + [live] * n_live


def test_run_validation_handles_cached_and_live_batches():
    """_run_validation iterates a mixed-tier loader without KeyError on 'images'."""
    from src.train.train import _run_validation, StageTrainingConfig, DatasetMix

    stage = StageTrainingConfig(
        stage_name="test",
        stage_b_encoder="radio_h",
        epochs=1, effective_samples_per_epoch=100, batch_size=2, max_sequence_length=512,
        lr_dora=0.0, lr_new_modules=0.0, warmup_steps=0, schedule="cosine",
        weight_decay=0.0, label_smoothing=0.0, contour_loss_weight=0.01,
        checkpoint_every_steps=500, validate_every_steps=500,
        grad_accumulation_steps=1, loraplus_lr_ratio=1.0,
        dataset_mix=(DatasetMix(dataset="grandstaff_systems", ratio=1.0),),
    )

    model = MagicMock()
    model.return_value = {
        "logits": torch.zeros(2, 511, 100),
        "contour_logits": torch.zeros(2, 3, 32),
    }

    loader = _make_mixed_val_loader(n_cached=2, n_live=1)
    result = _run_validation(
        model, stage, iter(loader), torch.device("cpu"),
        bf16_enabled=False, validation_batches=3, vocab_size=100,
    )

    assert result is not None
    assert "val_loss" in result
    # 3 calls (2 cached + 1 live), one of which used cached_features:
    cached_calls = [c for c in model.call_args_list if "cached_features" in c.kwargs]
    live_calls = [c for c in model.call_args_list if "pixel_values" in c.kwargs]
    assert len(cached_calls) == 2
    assert len(live_calls) == 1
