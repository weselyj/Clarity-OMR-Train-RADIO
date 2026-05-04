"""Tests for the vocab-extension hook in src/train/train.py."""
from __future__ import annotations

import pytest
import torch

from src.train.train import _extend_vocab_tensors_for_resume


def _make_state(vocab_size: int, dim: int = 16, *, tied: bool = False) -> dict:
    emb = torch.randn(vocab_size, dim)
    if tied:
        head_w = emb
    else:
        head_w = torch.randn(vocab_size, dim)
    head_b = torch.randn(vocab_size)
    return {
        "token_embedding.weight": emb,
        "lm_head.weight": head_w,
        "lm_head.bias": head_b,
        "decoder_blocks.0.weight": torch.randn(dim, dim),  # unrelated tensor
    }


def test_extension_pads_three_tensors_with_correct_init() -> None:
    src = _make_state(vocab_size=380, dim=16)
    dst = _make_state(vocab_size=388, dim=16)
    out, notes = _extend_vocab_tensors_for_resume(src, dst, init_seed=0)
    assert out["token_embedding.weight"].shape == (388, 16)
    assert out["lm_head.weight"].shape == (388, 16)
    assert out["lm_head.bias"].shape == (388,)
    # Original 380 rows preserved.
    assert torch.equal(out["token_embedding.weight"][:380], src["token_embedding.weight"])
    assert torch.equal(out["lm_head.weight"][:380], src["lm_head.weight"])
    assert torch.equal(out["lm_head.bias"][:380], src["lm_head.bias"])
    # New rows initialized near mean of existing rows.
    expected_emb_mean = src["token_embedding.weight"].mean(dim=0)
    new_rows = out["token_embedding.weight"][380:]
    assert torch.allclose(new_rows.mean(dim=0), expected_emb_mean, atol=0.02)
    # New bias rows are zero.
    assert torch.equal(out["lm_head.bias"][380:], torch.zeros(8))
    # Notes mention each extended tensor.
    assert any("token_embedding.weight" in n for n in notes)


def test_extension_is_idempotent_when_shapes_match() -> None:
    src = _make_state(vocab_size=388, dim=16)
    dst = _make_state(vocab_size=388, dim=16)
    out, notes = _extend_vocab_tensors_for_resume(src, dst, init_seed=0)
    assert torch.equal(out["token_embedding.weight"], src["token_embedding.weight"])
    assert notes == []


def test_extension_rejects_non_appended_diff() -> None:
    """Refuse to handle a checkpoint where the only change is something other than appending rows."""
    src = _make_state(vocab_size=380, dim=16)
    # Different decoder dim => shape diff in column count, not row count.
    dst = _make_state(vocab_size=380, dim=32)

    with pytest.raises(ValueError, match="only \\+N rows at end"):
        _extend_vocab_tensors_for_resume(src, dst, init_seed=0)


def test_extension_handles_weight_tying() -> None:
    src = _make_state(vocab_size=380, dim=16, tied=True)
    dst_emb = torch.randn(388, 16)
    dst = {
        "token_embedding.weight": dst_emb,
        "lm_head.weight": dst_emb,  # same tensor object (tied)
        "lm_head.bias": torch.randn(388),
        "decoder_blocks.0.weight": torch.randn(16, 16),
    }
    out, notes = _extend_vocab_tensors_for_resume(src, dst, init_seed=0)
    # Both keys point at extended tensors.
    assert out["token_embedding.weight"].shape == (388, 16)
    assert out["lm_head.weight"].shape == (388, 16)
    # Same content (tied) in the new rows.
    assert torch.equal(out["token_embedding.weight"], out["lm_head.weight"])


def test_extension_handles_peft_wrapped_names() -> None:
    """PEFT/DoRA wraps vocab-shaped layers into .original_module.X and .modules_to_save.default.X.
    The hook must extend all wrapped variants, not just the bare names."""
    src = {
        "base_model.model.token_embedding.original_module.weight": torch.randn(380, 16),
        "base_model.model.token_embedding.modules_to_save.default.weight": torch.randn(380, 16),
        "base_model.model.lm_head.original_module.weight": torch.randn(380, 16),
        "base_model.model.lm_head.original_module.bias": torch.randn(380),
        "base_model.model.lm_head.modules_to_save.default.weight": torch.randn(380, 16),
        "base_model.model.lm_head.modules_to_save.default.bias": torch.randn(380),
        "base_model.model.decoder_blocks.0.weight": torch.randn(16, 16),  # unrelated
    }
    dst = {
        "base_model.model.token_embedding.original_module.weight": torch.randn(388, 16),
        "base_model.model.token_embedding.modules_to_save.default.weight": torch.randn(388, 16),
        "base_model.model.lm_head.original_module.weight": torch.randn(388, 16),
        "base_model.model.lm_head.original_module.bias": torch.randn(388),
        "base_model.model.lm_head.modules_to_save.default.weight": torch.randn(388, 16),
        "base_model.model.lm_head.modules_to_save.default.bias": torch.randn(388),
        "base_model.model.decoder_blocks.0.weight": torch.randn(16, 16),
    }
    out, notes = _extend_vocab_tensors_for_resume(src, dst, init_seed=0)
    # All 6 vocab-shaped tensors extended.
    for key in src:
        if "token_embedding" in key or "lm_head" in key:
            assert out[key].shape[0] == 388, f"{key} not extended"
    assert len(notes) == 6
