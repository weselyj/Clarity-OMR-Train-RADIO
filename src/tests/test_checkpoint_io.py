"""Tests for the shared checkpoint loader (Bug 1: DoRA-stripping fix).

TDD: these tests were written before the implementation.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers: minimal model that looks like a Stage-B model structurally
# ---------------------------------------------------------------------------

class _MinimalStageB(nn.Module):
    """Minimal model with real nn.Linear layers, no RADIO backbone.

    Using nn.Linear directly so that _prepare_model_for_dora can find
    and wrap the linear layers via isinstance(module, nn.Linear).
    """
    def __init__(self):
        super().__init__()
        self.foo = nn.Linear(4, 4)
        self.bar = nn.Linear(4, 4)

    def encode_staff(self, x):  # pragma: no cover
        return x

    def decode_tokens(self, *a, **kw):  # pragma: no cover
        return None, None, None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_plain_state_dict() -> Dict[str, Any]:
    """State dict with base_model.model. prefix but NO lora keys."""
    model = _MinimalStageB()
    return {
        f"base_model.model.{k}": v.clone()
        for k, v in model.state_dict().items()
    }


def _build_dora_state_dict() -> Dict[str, Any]:
    """State dict that contains both base weights AND synthetic DoRA adapter keys."""
    model = _MinimalStageB()
    base = {f"base_model.model.{k}": v.clone() for k, v in model.state_dict().items()}

    rank = 2
    # Simulate DoRA keys for the foo.weight layer
    dora_extra = {
        "base_model.model.foo.lora_A.default.weight": torch.randn(rank, 4),
        "base_model.model.foo.lora_B.default.weight": torch.randn(4, rank),
        "base_model.model.foo.lora_magnitude_vector.default.weight": torch.ones(4),
        "base_model.model.bar.lora_A.default.weight": torch.randn(rank, 4),
        "base_model.model.bar.lora_B.default.weight": torch.randn(4, rank),
        "base_model.model.bar.lora_magnitude_vector.default.weight": torch.ones(4),
        # modules_to_save style key
        "base_model.model.foo.modules_to_save.default.weight": torch.randn(4, 4),
    }
    return {**base, **dora_extra}


def _save_checkpoint(state_dict: Dict[str, Any], path: Path) -> None:
    torch.save({"model_state_dict": state_dict}, str(path))


# ---------------------------------------------------------------------------
# Test 1: shared helper detects DoRA and raises for coverage < threshold
# on a plain (non-DoRA-wrapped) model receiving DoRA keys
# ---------------------------------------------------------------------------

def test_dora_detection_raises_when_unwrapped_model_loaded_with_dora_keys(tmp_path):
    """The OLD code path: strip prefix, load into unwrapped model.

    With DoRA keys present, most keys become unexpected → coverage falls well
    below 50%.  The new helper must raise RuntimeError mentioning 'coverage'.
    """
    from src.checkpoint_io import load_stage_b_checkpoint

    ckpt = tmp_path / "dora.pt"
    state_dict = _build_dora_state_dict()
    _save_checkpoint(state_dict, ckpt)

    model = _MinimalStageB()
    device = torch.device("cpu")

    # The new helper should detect DoRA but when the model is NOT wrapped (no
    # peft available or force_unwrapped=True) AND coverage drops, it must raise.
    with pytest.raises(RuntimeError, match="coverage"):
        load_stage_b_checkpoint(
            checkpoint_path=ckpt,
            model=model,
            device=device,
            dora_config=None,  # no dora_config → must fail with coverage error
        )


# ---------------------------------------------------------------------------
# Test 2: plain checkpoint loads cleanly (no coverage error)
# ---------------------------------------------------------------------------

def test_plain_checkpoint_loads_with_high_coverage(tmp_path):
    """A plain (non-DoRA) checkpoint should load with ≥ 95% coverage."""
    from src.checkpoint_io import load_stage_b_checkpoint

    model_src = _MinimalStageB()
    model_dst = _MinimalStageB()
    ckpt = tmp_path / "plain.pt"

    state_dict = {f"base_model.model.{k}": v.clone() for k, v in model_src.state_dict().items()}
    _save_checkpoint(state_dict, ckpt)

    device = torch.device("cpu")
    result = load_stage_b_checkpoint(
        checkpoint_path=ckpt,
        model=model_dst,
        device=device,
        dora_config=None,
    )
    assert result["load_ratio"] >= 0.95
    assert result["checkpoint_format"] == "plain"


# ---------------------------------------------------------------------------
# Test 3: DoRA checkpoint with dora_config loads correctly (≥ 50% coverage)
# ---------------------------------------------------------------------------

def test_dora_checkpoint_with_dora_config_loads(tmp_path):
    """When dora_config is supplied and keys look like DoRA, the helper wraps
    the model with PEFT and loads with ≥ 50% coverage."""
    from src.checkpoint_io import load_stage_b_checkpoint

    ckpt = tmp_path / "dora.pt"
    state_dict = _build_dora_state_dict()
    _save_checkpoint(state_dict, ckpt)

    device = torch.device("cpu")
    # Build a minimal dora_config that matches the minimal model's linear layers.
    # _prepare_model_for_dora targets every nn.Linear not in the exclusion list.
    dora_config = {
        "adapter_type": "dora",
        "rank": 2,
        "alpha": 2,
        "dropout": 0.0,
        "target_modules": [],  # empty = match all linears not excluded
    }
    model = _MinimalStageB()
    result = load_stage_b_checkpoint(
        checkpoint_path=ckpt,
        model=model,
        device=device,
        dora_config=dora_config,
    )
    assert result["checkpoint_format"] == "dora_peft"
    assert result["load_ratio"] >= 0.50


# ---------------------------------------------------------------------------
# Test 4: missing-key / unexpected-key SAMPLES are reported in result
# ---------------------------------------------------------------------------

def test_load_result_reports_key_samples(tmp_path):
    """Result dict must include 'missing_key_sample' and 'unexpected_key_sample'
    (or empty lists if there are none)."""
    from src.checkpoint_io import load_stage_b_checkpoint

    model_src = _MinimalStageB()
    model_dst = _MinimalStageB()
    ckpt = tmp_path / "plain.pt"

    state_dict = {f"base_model.model.{k}": v.clone() for k, v in model_src.state_dict().items()}
    _save_checkpoint(state_dict, ckpt)

    device = torch.device("cpu")
    result = load_stage_b_checkpoint(
        checkpoint_path=ckpt,
        model=model_dst,
        device=device,
        dora_config=None,
    )
    assert "missing_key_sample" in result
    assert "unexpected_key_sample" in result
    assert isinstance(result["missing_key_sample"], list)
    assert isinstance(result["unexpected_key_sample"], list)
