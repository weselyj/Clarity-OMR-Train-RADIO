"""Per-dataset val_loss: 4 disjoint passes + sample-weighted aggregate."""
from __future__ import annotations
from unittest.mock import MagicMock

import torch
import pytest


def _mk_stage(dataset_mix_tuple, cached_data_ratio=0.9, tier_grouped=True):
    """Helper: build a minimal StageTrainingConfig for val-aggregation tests."""
    from src.train.train import StageTrainingConfig, DatasetMix
    return StageTrainingConfig(
        stage_name="stage3-test",
        stage_b_encoder="radio_h",
        epochs=1, effective_samples_per_epoch=100, batch_size=2,
        max_sequence_length=512,
        lr_dora=0.0, lr_new_modules=0.0, warmup_steps=0, schedule="cosine",
        weight_decay=0.0, label_smoothing=0.0, contour_loss_weight=0.01,
        checkpoint_every_steps=500, validate_every_steps=500,
        grad_accumulation_steps=1, loraplus_lr_ratio=1.0,
        dataset_mix=tuple(DatasetMix(dataset=d, ratio=r) for d, r in dataset_mix_tuple),
        tier_grouped_sampling=tier_grouped,
        b_cached=2, b_live=2,
        grad_accumulation_steps_cached=1, grad_accumulation_steps_live=1,
        cached_data_ratio=cached_data_ratio,
        cache_root="x", cache_hash16="x",
    )


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
    # (In this fixture the YAML ratios already sum to 1.0 with non-zero camera, so
    # the cached_data_ratio re-projection produces the same weights.)
    weights = {dm.dataset: dm.ratio for dm in stage.dataset_mix}
    expected = sum(weights[k] * v for k, v in result["val_loss_per_dataset"].items())
    assert result["val_loss"] == pytest.approx(expected, rel=1e-6)


def test_aggregate_uses_cached_data_ratio_not_dataset_mix_directly(monkeypatch):
    """Production YAML has cameraprimus_systems.ratio=0.0 (it's the live tier's
    dataset; that field is the cached-tier WeightedRandomSampler weight, not the
    global aggregate weight). Spec Decision #4 says the aggregate must still
    weight cameraprimus at ``1 - cached_data_ratio = 0.10``. The previous code
    weighted by ``dm.ratio`` directly, silently dropping cameraprimus from
    ``best_val_loss`` and the sanity halt.
    """
    from src.train import train as train_mod

    stage = _mk_stage(
        # Production-shaped: cameraprimus ratio=0.0; cached ratios sum to 1.0.
        dataset_mix_tuple=(
            ("synthetic_systems", 0.7778),
            ("grandstaff_systems", 0.1111),
            ("primus_systems", 0.1111),
            ("cameraprimus_systems", 0.0),
        ),
        cached_data_ratio=0.9,
        tier_grouped=True,
    )

    # Stub _run_validation to return distinct, hand-picked losses per dataset
    # keyed by loader id (we map id(loader) -> dataset_name via the dict order
    # we pass into _run_validation_per_dataset).
    preset_losses = {
        "synthetic_systems": 0.5,
        "grandstaff_systems": 1.0,
        "primus_systems": 1.5,
        "cameraprimus_systems": 2.0,
    }
    preset_contour = {
        "synthetic_systems": 0.05,
        "grandstaff_systems": 0.10,
        "primus_systems": 0.15,
        "cameraprimus_systems": 0.20,
    }
    # Build sentinel "loader" objects we can identify by id().
    loaders = {name: object() for name in preset_losses}
    id_to_name = {id(loader): name for name, loader in loaders.items()}

    def fake_run_validation(model, stage_arg, loader, device, **_kwargs):
        name = id_to_name[id(loader)]
        return {
            "val_loss": preset_losses[name],
            "val_contour_loss": preset_contour[name],
        }

    monkeypatch.setattr(train_mod, "_run_validation", fake_run_validation)

    result = train_mod._run_validation_per_dataset(
        model=MagicMock(),
        stage=stage,
        per_dataset_loaders=loaders,
        device=torch.device("cpu"),
        bf16_enabled=False,
        validation_batches=1,
        vocab_size=100,
    )

    # Per-dataset losses should round-trip unchanged.
    assert result["val_loss_per_dataset"] == preset_losses
    assert result["val_contour_loss_per_dataset"] == preset_contour

    # Aggregate weights derived from cached_data_ratio=0.9 + per-tier
    # dataset_mix shares: synth=0.7, grand=0.1, primus=0.1, camera=0.1.
    # Cameraprimus must be 0.1 (live_share), NOT 0.0 (its dm.ratio).
    expected_loss = (
        0.7 * preset_losses["synthetic_systems"]
        + 0.1 * preset_losses["grandstaff_systems"]
        + 0.1 * preset_losses["primus_systems"]
        + 0.1 * preset_losses["cameraprimus_systems"]
    )
    expected_contour = (
        0.7 * preset_contour["synthetic_systems"]
        + 0.1 * preset_contour["grandstaff_systems"]
        + 0.1 * preset_contour["primus_systems"]
        + 0.1 * preset_contour["cameraprimus_systems"]
    )
    # 0.7*0.5 + 0.1*1.0 + 0.1*1.5 + 0.1*2.0 = 0.35 + 0.1 + 0.15 + 0.2 = 0.8
    assert expected_loss == pytest.approx(0.8, rel=1e-9)
    assert result["val_loss"] == pytest.approx(expected_loss, rel=1e-9)
    assert result["val_contour_loss"] == pytest.approx(expected_contour, rel=1e-9)

    # Regression guard: the buggy implementation would have weighted by
    # dm.ratio directly, with weight_sum = 0.7778 + 0.1111 + 0.1111 + 0.0 = 1.0
    # and dropped cameraprimus entirely.
    buggy_weight_sum = 0.7778 + 0.1111 + 0.1111 + 0.0
    buggy_agg = (
        0.7778 * preset_losses["synthetic_systems"]
        + 0.1111 * preset_losses["grandstaff_systems"]
        + 0.1111 * preset_losses["primus_systems"]
        + 0.0 * preset_losses["cameraprimus_systems"]
    ) / buggy_weight_sum
    assert result["val_loss"] != pytest.approx(buggy_agg, rel=1e-3), (
        "Aggregate matches the buggy dm.ratio-weighted result; the fix did not "
        "take effect (cameraprimus_systems was dropped from the aggregate)."
    )


def test_aggregate_falls_back_to_dataset_mix_for_legacy_stages(monkeypatch):
    """Legacy (non-tier-grouped) stages do not set cached_data_ratio. The
    aggregator must fall back to dataset_mix.ratio directly so existing
    callers are unaffected by the tier-grouped re-projection.
    """
    from src.train import train as train_mod

    # Legacy stage: tier_grouped_sampling=False AND cached_data_ratio=None.
    stage = _mk_stage(
        dataset_mix_tuple=(
            ("synthetic_systems", 0.5),
            ("grandstaff_systems", 0.3),
            ("primus_systems", 0.2),
        ),
        cached_data_ratio=None,
        tier_grouped=False,
    )

    preset_losses = {
        "synthetic_systems": 1.0,
        "grandstaff_systems": 2.0,
        "primus_systems": 4.0,
    }
    loaders = {name: object() for name in preset_losses}
    id_to_name = {id(loader): name for name, loader in loaders.items()}

    def fake_run_validation(model, stage_arg, loader, device, **_kwargs):
        name = id_to_name[id(loader)]
        return {"val_loss": preset_losses[name], "val_contour_loss": 0.0}

    monkeypatch.setattr(train_mod, "_run_validation", fake_run_validation)

    result = train_mod._run_validation_per_dataset(
        model=MagicMock(),
        stage=stage,
        per_dataset_loaders=loaders,
        device=torch.device("cpu"),
        bf16_enabled=False,
        validation_batches=1,
        vocab_size=100,
    )

    # Legacy path: weights == dm.ratio directly.
    expected = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 4.0  # = 1.9
    assert result["val_loss"] == pytest.approx(expected, rel=1e-9)
