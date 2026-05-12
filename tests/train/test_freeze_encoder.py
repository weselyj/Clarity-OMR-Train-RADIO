"""Regression tests for the Stage 3 v2 encoder-freeze bug.

Bug recap (`docs/audits/2026-05-11-stage3-v2-training-audit.md`, PR #48):
Stage 3 v2 training trained the encoder DoRA adapters despite the config and
checkpoint name claiming a frozen encoder. The fix in `_prepare_model_for_dora`
makes the freeze a structural consequence of using the encoder cache — when
`stage_config.cache_root` and `stage_config.cache_hash16` are both set, encoder
params must end with `requires_grad=False`.

CUDA-gated by `tests/conftest.py` (path matches CUDA_REQUIRED_DIRS); these
tests are SKIPPED locally and run on seder (venv-cu132).
"""
from __future__ import annotations

from typing import Optional


def _build_components_and_dora_config():
    """Build a small but production-shaped Stage B model via the model factory.

    Uses RADIO encoder + decoder_dim=64 / decoder_layers=1 / decoder_heads=2
    to minimize model build time. The encoder itself is fixed-size (RADIO-H
    1280-dim); only the decoder is sized down.
    """
    from src.train.model_factory import build_stage_b_components, ModelFactoryConfig
    factory_cfg = ModelFactoryConfig(
        stage_b_vocab_size=64,
        stage_b_encoder="radio_h",
        stage_b_decoder_dim=64,
        stage_b_decoder_heads=2,
        stage_b_decoder_layers=1,
        stage_b_dora_rank=8,
    )
    components = build_stage_b_components(factory_cfg)
    return components["model"], components["dora_config"]


def _stage_config(*, cache_root: Optional[str] = None, cache_hash16: Optional[str] = None):
    """Build a StageTrainingConfig with the cache fields under test, defaults
    elsewhere. dataset_mix has one entry summing to 1.0 (required by validator).
    """
    from src.train.train import StageTrainingConfig, DatasetMix
    return StageTrainingConfig(
        stage_name="test_freeze_encoder",
        epochs=1,
        effective_samples_per_epoch=100,
        batch_size=1,
        max_sequence_length=128,
        lr_dora=5e-4,
        lr_new_modules=3e-4,
        warmup_steps=10,
        schedule="cosine",
        label_smoothing=0.0,
        contour_loss_weight=0.0,
        weight_decay=0.0,
        checkpoint_every_steps=100,
        validate_every_steps=100,
        grad_accumulation_steps=1,
        loraplus_lr_ratio=1.0,
        dataset_mix=(DatasetMix(dataset="synthetic_systems", ratio=1.0, split="train"),),
        stage_b_encoder="radio_h",
        cache_root=cache_root,
        cache_hash16=cache_hash16,
    )


def test_cache_config_freezes_encoder_lora_params():
    """When the stage config has cache_root + cache_hash16 set, every
    encoder-side parameter must end with requires_grad=False — including any
    LoRA adapters PEFT injected into encoder modules."""
    from src.train.train import _prepare_model_for_dora
    model, dora_config = _build_components_and_dora_config()
    stage_config = _stage_config(
        cache_root="data/cache/encoder",
        cache_hash16="ac8948ae4b5be3e9",
    )
    model, _ = _prepare_model_for_dora(model, dora_config, stage_config=stage_config)
    trainable_encoder = [
        name for name, p in model.named_parameters()
        if "encoder" in name and p.requires_grad
    ]
    assert trainable_encoder == [], (
        f"Encoder params should be frozen when cache is configured, but found "
        f"{len(trainable_encoder)} trainable encoder params. First 5: "
        f"{trainable_encoder[:5]}"
    )


def test_no_cache_unfreezes_encoder_lora_params():
    """Mirror of the above — proves the cache-derived gate does real work.
    Without cache_root/cache_hash16, encoder-side LoRA params should be trainable."""
    from src.train.train import _prepare_model_for_dora
    model, dora_config = _build_components_and_dora_config()
    stage_config = _stage_config()  # no cache
    model, _ = _prepare_model_for_dora(model, dora_config, stage_config=stage_config)
    encoder_lora_trainable = [
        name for name, p in model.named_parameters()
        if "encoder" in name and "lora_" in name and p.requires_grad
    ]
    assert len(encoder_lora_trainable) > 0, (
        "Expected encoder-side LoRA params to be trainable when no cache is "
        "configured. The cache+freeze pairing should be the only thing that "
        "triggers the freeze."
    )


def test_decoder_lora_always_trainable():
    """Sanity check: decoder-side LoRA params must be trainable regardless of
    whether the encoder is frozen (decoder always trains during Stage 3)."""
    from src.train.train import _prepare_model_for_dora
    for cache_root, cache_hash16 in ((None, None), ("data/cache/encoder", "ac8948ae4b5be3e9")):
        model, dora_config = _build_components_and_dora_config()
        stage_config = _stage_config(cache_root=cache_root, cache_hash16=cache_hash16)
        model, _ = _prepare_model_for_dora(model, dora_config, stage_config=stage_config)
        decoder_lora_trainable = [
            name for name, p in model.named_parameters()
            if "encoder" not in name and "lora_" in name and p.requires_grad
        ]
        assert len(decoder_lora_trainable) > 0, (
            f"Decoder LoRA should be trainable with cache_root={cache_root!r}. "
            f"Got {len(decoder_lora_trainable)} trainable decoder LoRA params."
        )
