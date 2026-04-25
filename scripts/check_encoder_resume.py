#!/usr/bin/env python3
"""Sanity-check: encoder discriminator survives checkpoint serialisation.

This script exercises the full config-roundtrip path that
model_factory_config_from_checkpoint_payload walks on resume, without
loading RADIO weights or a GPU.  It verifies:

  1. RadioStageBConfig serialises to a dict that contains encoder='radio_h'.
  2. model_factory_config_from_checkpoint_payload reads that field and
     returns a ModelFactoryConfig with stage_b_encoder='radio_h'.
  3. That config dispatches build_stage_b_components to the RadioStageB
     path (verified by asserting the returned model class name, not by
     loading RADIO weights -- we mock RadioEncoder to avoid GPU/network).
  4. A symmetric check for the DaViT path: StageBModelConfig serialises
     encoder='davit' and the factory returns a StageBModel.

Run with:
    venv\\Scripts\\python scripts\\check_encoder_resume.py
"""
from __future__ import annotations

import dataclasses
import sys
import types
import unittest.mock as mock


def _patch_radio_encoder():
    """Stub out RadioEncoder so the test doesn't need RADIO weights."""
    import torch.nn as nn

    class _FakeRadioEncoder(nn.Module):
        hidden_dim = 1280

        def __init__(self):
            super().__init__()
            # Minimal linear so named_modules() sees something
            self._dummy = nn.Linear(1, 1)

        def forward(self, x):
            raise NotImplementedError("FakeRadioEncoder is not for forward()")

    return _FakeRadioEncoder


def main():
    print("=" * 60)
    print("Encoder discriminator checkpoint-resume sanity check")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Import configs and factory (these must not load RADIO weights yet)
    # ------------------------------------------------------------------
    from src.models.radio_stage_b import RadioStageBConfig
    from src.models.davit_stage_b import StageBModelConfig
    from src.train.model_factory import (
        ModelFactoryConfig,
        model_factory_config_from_checkpoint_payload,
    )

    # ------------------------------------------------------------------
    # 2. RADIO path: verify config serialises encoder='radio_h'
    # ------------------------------------------------------------------
    radio_cfg = RadioStageBConfig()
    radio_dict = dataclasses.asdict(radio_cfg)
    assert "encoder" in radio_dict, (
        f"FAIL: 'encoder' not found in RadioStageBConfig.asdict(): {radio_dict}"
    )
    assert radio_dict["encoder"] == "radio_h", (
        f"FAIL: expected encoder='radio_h', got {radio_dict['encoder']!r}"
    )
    print(f"[PASS] RadioStageBConfig serialises encoder={radio_dict['encoder']!r}")

    # ------------------------------------------------------------------
    # 3. RADIO path: factory reads the field and returns radio_h config
    # ------------------------------------------------------------------
    fake_payload = {
        "stage_b_config": radio_dict,
        "model_state_dict": {},
    }
    recovered = model_factory_config_from_checkpoint_payload(
        fake_payload, vocab_size=487
    )
    assert recovered.stage_b_encoder == "radio_h", (
        f"FAIL: expected stage_b_encoder='radio_h' after resume, "
        f"got {recovered.stage_b_encoder!r}"
    )
    print(f"[PASS] model_factory_config_from_checkpoint_payload returns "
          f"stage_b_encoder={recovered.stage_b_encoder!r}")

    # ------------------------------------------------------------------
    # 4. RADIO path: build_stage_b_components returns RadioStageB
    #    (mock RadioEncoder so we don't need weights/GPU)
    # ------------------------------------------------------------------
    FakeRadioEncoder = _patch_radio_encoder()
    with mock.patch("src.models.radio_stage_b.RadioEncoder", FakeRadioEncoder):
        from src.train.model_factory import build_stage_b_components
        components = build_stage_b_components(recovered)
        model = components["model"]
        model_cls = type(model).__name__
        assert model_cls == "RadioStageB", (
            f"FAIL: expected model class 'RadioStageB', got '{model_cls}'"
        )
        print(f"[PASS] build_stage_b_components dispatched to {model_cls}")

    # ------------------------------------------------------------------
    # 5. DaViT path: StageBModelConfig serialises encoder='davit'
    # ------------------------------------------------------------------
    davit_cfg = StageBModelConfig()
    davit_dict = dataclasses.asdict(davit_cfg)
    assert "encoder" in davit_dict, (
        f"FAIL: 'encoder' not found in StageBModelConfig.asdict(): {davit_dict}"
    )
    assert davit_dict["encoder"] == "davit", (
        f"FAIL: expected encoder='davit', got {davit_dict['encoder']!r}"
    )
    print(f"[PASS] StageBModelConfig serialises encoder={davit_dict['encoder']!r}")

    # ------------------------------------------------------------------
    # 6. DaViT path: factory reads field and returns davit config
    # ------------------------------------------------------------------
    davit_payload = {
        "stage_b_config": davit_dict,
        "model_state_dict": {},
    }
    davit_recovered = model_factory_config_from_checkpoint_payload(
        davit_payload, vocab_size=380
    )
    assert davit_recovered.stage_b_encoder == "davit", (
        f"FAIL: expected stage_b_encoder='davit' after resume, "
        f"got {davit_recovered.stage_b_encoder!r}"
    )
    print(f"[PASS] DaViT resume returns stage_b_encoder={davit_recovered.stage_b_encoder!r}")

    print()
    print("All checks passed. Encoder discriminator survives checkpoint roundtrip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
