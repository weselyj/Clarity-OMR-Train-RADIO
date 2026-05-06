"""Verify the trainer constructs a fused AdamW optimizer when CUDA is available."""
import pytest
import torch
import torch.nn as nn


def _make_stage():
    """Build a minimal StageTrainingConfig with only the fields _build_optimizer uses."""
    from src.train.train import StageTrainingConfig, DatasetMix

    return StageTrainingConfig(
        stage_name="test",
        epochs=1,
        effective_samples_per_epoch=1,
        batch_size=1,
        max_sequence_length=32,
        lr_dora=1e-4,
        lr_new_modules=1e-3,
        warmup_steps=1,
        schedule="cosine",
        label_smoothing=0.0,
        contour_loss_weight=0.0,
        weight_decay=0.01,
        checkpoint_every_steps=1,
        validate_every_steps=1,
        grad_accumulation_steps=1,
        loraplus_lr_ratio=2.0,
        dataset_mix=(DatasetMix(dataset="dummy", ratio=1.0),),
    )


def _make_model():
    """Return a tiny model with at least one trainable parameter that has a
    non-lora, non-bias name so it lands in the new_module_decay bucket."""
    return nn.Linear(4, 4)


def test_build_optimizer_uses_fused_adamw_on_cuda(monkeypatch):
    """The optimizer factory must request fused=True when CUDA is available."""
    import torch
    from src.train import train as train_mod

    captured = {}
    real_adamw = torch.optim.AdamW

    def spy(params, **kwargs):
        captured["kwargs"] = kwargs
        return real_adamw(params, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = _make_model()
    stage = _make_stage()
    _ = train_mod._build_optimizer(model, stage)

    assert captured.get("kwargs", {}).get("fused") is True


def test_build_optimizer_uses_unfused_adamw_without_cuda(monkeypatch):
    """The factory must NOT request fused=True when CUDA is unavailable
    (fused AdamW only works on CUDA)."""
    import torch
    from src.train import train as train_mod

    captured = {}
    real_adamw = torch.optim.AdamW

    def spy(params, **kwargs):
        captured["kwargs"] = kwargs
        return real_adamw(params, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model = _make_model()
    stage = _make_stage()
    _ = train_mod._build_optimizer(model, stage)

    assert captured.get("kwargs", {}).get("fused") is False
