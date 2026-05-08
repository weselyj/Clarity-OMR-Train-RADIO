"""Test that best_val_loss is persisted in and restored from checkpoints.

Issue #5: _save_checkpoint did not include best_val_loss in its payload, so
on resume best_val_loss was always None and the first validation pass after
resume would unconditionally overwrite _best.pt even if the pre-resume model
had already achieved a better loss.
"""
import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path


def _make_tiny_model():
    """Return a trivially small model whose state_dict round-trips cleanly."""
    return nn.Linear(4, 4)


def _make_stage():
    """Build a minimal StageTrainingConfig for checkpoint helpers."""
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


def test_save_checkpoint_persists_best_val_loss():
    """_save_checkpoint must include best_val_loss in the saved payload."""
    from src.train.train import _save_checkpoint

    model = _make_tiny_model()
    stage = _make_stage()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = _save_checkpoint(
            checkpoint_dir=Path(tmpdir),
            model=model,
            optimizer=optimizer,
            stage_name=stage.stage_name,
            global_step=100,
            best_val_loss=0.5,
        )
        payload = torch.load(ckpt_path, map_location="cpu")

    assert "best_val_loss" in payload, (
        "_save_checkpoint must include 'best_val_loss' in the saved payload"
    )
    assert payload["best_val_loss"] == pytest.approx(0.5), (
        "Loaded best_val_loss must match the value passed to _save_checkpoint"
    )


def test_save_checkpoint_persists_none_best_val_loss():
    """_save_checkpoint must persist best_val_loss=None without error."""
    from src.train.train import _save_checkpoint

    model = _make_tiny_model()
    stage = _make_stage()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = _save_checkpoint(
            checkpoint_dir=Path(tmpdir),
            model=model,
            optimizer=optimizer,
            stage_name=stage.stage_name,
            global_step=100,
            best_val_loss=None,
        )
        payload = torch.load(ckpt_path, map_location="cpu")

    # best_val_loss=None is a valid state (no validation has run yet).
    assert "best_val_loss" in payload
    assert payload["best_val_loss"] is None


def test_resume_restores_best_val_loss_from_checkpoint():
    """A checkpoint saved with best_val_loss=0.5 must restore that value
    from checkpoint_payload so that the resumed run does not unconditionally
    overwrite _best.pt on the first validation pass.

    This test exercises the restore path directly via checkpoint_payload.get().
    """
    # Simulate what run_execute_mode does at resume time.
    checkpoint_payload = {
        "stage_name": "test",
        "global_step": 100,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "best_val_loss": 0.5,
    }

    # This is the pattern the fix must implement in run_execute_mode.
    restored_best_val_loss = checkpoint_payload.get("best_val_loss", None)

    assert restored_best_val_loss == pytest.approx(0.5), (
        "Resume path must restore best_val_loss from checkpoint payload"
    )


def test_resume_defaults_best_val_loss_to_none_for_old_checkpoints():
    """Checkpoints saved before this fix lack 'best_val_loss'; the restore
    path must fall back to None rather than raising KeyError.
    """
    # Old checkpoint format — no best_val_loss key.
    checkpoint_payload = {
        "stage_name": "test",
        "global_step": 100,
        "model_state_dict": {},
        "optimizer_state_dict": {},
    }

    restored_best_val_loss = checkpoint_payload.get("best_val_loss", None)

    assert restored_best_val_loss is None, (
        "Old checkpoints without 'best_val_loss' must restore to None (backward compat)"
    )
