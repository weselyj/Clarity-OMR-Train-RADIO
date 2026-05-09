"""Train loop dispatches on batch_dict['tier'] when forwarding through the model."""
from __future__ import annotations
from unittest.mock import MagicMock

import torch
import pytest


def _make_cached_batch():
    """Mimics StageBDataset.collate_fn output for a cached-tier batch (b=2)."""
    return {
        "tier": "cached",
        "encoder_hidden": torch.zeros(2, 156, 1280, dtype=torch.bfloat16),
        "_h16": 16,
        "_w16": 156,
        "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
        "labels": torch.zeros(2, 511, dtype=torch.long),
        "contour_targets": torch.zeros(2, 32, dtype=torch.long),
    }


def _make_live_batch():
    """Mimics StageBDataset.collate_fn output for a live-tier batch (b=2)."""
    return {
        "tier": "live",
        "images": torch.zeros(2, 1, 250, 2500, dtype=torch.float32),
        "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
        "labels": torch.zeros(2, 511, dtype=torch.long),
        "contour_targets": torch.zeros(2, 32, dtype=torch.long),
        "content_widths": torch.tensor([2500, 2500], dtype=torch.long),
    }


def test_dispatch_cached_batch_calls_model_with_cached_features():
    from src.train.train import _forward_batch_for_train

    model = MagicMock()
    model.return_value = {
        "logits": torch.zeros(2, 511, 100),
        "contour_logits": torch.zeros(2, 3, 32),
    }
    device = torch.device("cpu")
    batch = _make_cached_batch()

    _forward_batch_for_train(model, batch, device, bf16_enabled=False, channels_last=False)

    args, kwargs = model.call_args
    assert "cached_features" in kwargs
    assert "pixel_values" not in kwargs
    assert kwargs["cached_features"].shape == (2, 156, 1280)
    assert kwargs["_h16"] == 16
    assert kwargs["_w16"] == 156


def test_dispatch_live_batch_calls_model_with_pixel_values():
    from src.train.train import _forward_batch_for_train

    model = MagicMock()
    model.return_value = {
        "logits": torch.zeros(2, 511, 100),
        "contour_logits": torch.zeros(2, 3, 32),
    }
    device = torch.device("cpu")
    batch = _make_live_batch()

    _forward_batch_for_train(model, batch, device, bf16_enabled=False, channels_last=False)

    args, kwargs = model.call_args
    assert "pixel_values" in kwargs
    assert "cached_features" not in kwargs
    assert kwargs["pixel_values"].shape == (2, 1, 250, 2500)
