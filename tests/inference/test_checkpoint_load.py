"""Unit tests for src.inference.checkpoint_load."""
from __future__ import annotations

from dataclasses import is_dataclass


def test_bundle_is_dataclass_with_expected_fields():
    from src.inference.checkpoint_load import StageBInferenceBundle

    assert is_dataclass(StageBInferenceBundle)
    fields = {f.name for f in StageBInferenceBundle.__dataclass_fields__.values()}
    assert fields == {
        "model",
        "decode_model",
        "vocab",
        "token_to_idx",
        "use_fp16",
        "factory_cfg",
    }
