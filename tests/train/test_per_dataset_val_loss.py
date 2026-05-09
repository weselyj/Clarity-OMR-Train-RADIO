"""Per-dataset val_loss: 4 disjoint passes + sample-weighted aggregate."""
from __future__ import annotations
from unittest.mock import MagicMock

import torch
import pytest


def test_run_validation_per_dataset_returns_one_loss_per_dataset_plus_aggregate():
    """The per-dataset entry point returns a dict with one val_loss per dataset
    and one aggregate val_loss (sample-weighted by dataset_mix)."""
    from src.train.train import _run_validation_per_dataset, StageTrainingConfig, DatasetMix

    stage = StageTrainingConfig(
        stage_name="stage3-test",
        stage_b_encoder="radio_h",
        epochs=1, effective_samples_per_epoch=100, batch_size=2, max_sequence_length=512,
        lr_dora=0.0, lr_new_modules=0.0, warmup_steps=0, schedule="cosine",
        weight_decay=0.0, label_smoothing=0.0, contour_loss_weight=0.01,
        checkpoint_every_steps=500, validate_every_steps=500,
        grad_accumulation_steps=1, loraplus_lr_ratio=1.0,
        dataset_mix=(
            DatasetMix(dataset="synthetic_systems", ratio=0.7),
            DatasetMix(dataset="grandstaff_systems", ratio=0.1),
            DatasetMix(dataset="primus_systems", ratio=0.1),
            DatasetMix(dataset="cameraprimus_systems", ratio=0.1),
        ),
        tier_grouped_sampling=True,
        b_cached=2, b_live=2,
        grad_accumulation_steps_cached=1, grad_accumulation_steps_live=1,
        cached_data_ratio=0.9,
        cache_root="x", cache_hash16="x",
    )

    model = MagicMock()
    model.return_value = {
        "logits": torch.zeros(2, 511, 100),
        "contour_logits": torch.zeros(2, 3, 32),
    }

    # Mock per-dataset loaders: each yields 1 batch with all-zeros tensors.
    def _mk_loader(tier: str):
        if tier == "cached":
            batch = {
                "tier": "cached",
                "encoder_hidden": torch.zeros(2, 156, 1280, dtype=torch.bfloat16),
                "_h16": 16, "_w16": 156,
                "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
                "labels": torch.zeros(2, 511, dtype=torch.long),
                "contour_targets": torch.zeros(2, 32, dtype=torch.long),
            }
        else:
            batch = {
                "tier": "live",
                "images": torch.zeros(2, 1, 250, 2500),
                "decoder_inputs": torch.zeros(2, 511, dtype=torch.long),
                "labels": torch.zeros(2, 511, dtype=torch.long),
                "contour_targets": torch.zeros(2, 32, dtype=torch.long),
                "content_widths": torch.tensor([2500, 2500], dtype=torch.long),
            }
        return [batch]

    per_dataset_loaders = {
        "synthetic_systems": _mk_loader("cached"),
        "grandstaff_systems": _mk_loader("cached"),
        "primus_systems": _mk_loader("cached"),
        "cameraprimus_systems": _mk_loader("live"),
    }

    result = _run_validation_per_dataset(
        model, stage, per_dataset_loaders, torch.device("cpu"),
        bf16_enabled=False, validation_batches=1, vocab_size=100,
    )

    assert "val_loss_per_dataset" in result
    assert set(result["val_loss_per_dataset"].keys()) == {
        "synthetic_systems", "grandstaff_systems", "primus_systems", "cameraprimus_systems",
    }
    assert "val_loss" in result  # aggregate
    # The aggregate must equal the dataset_mix-weighted mean of per-dataset losses.
    weights = {dm.dataset: dm.ratio for dm in stage.dataset_mix}
    expected = sum(weights[k] * v for k, v in result["val_loss_per_dataset"].items())
    assert result["val_loss"] == pytest.approx(expected, rel=1e-6)
