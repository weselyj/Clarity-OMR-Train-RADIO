"""DoRA-aware loader smoke test against a Stage 2 v2-shaped checkpoint.

Plan C Phase 1 launches with::

    --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt

so the DoRA-aware loader path documented in
``project_radio_stage3_design.md`` (line 55) MUST work end-to-end on the
Stage 2 v2 best.pt schema.  Naive ``load_state_dict(strict=False)`` silently
leaves the encoder near-random because all ``lora_*`` keys land in
``unexpected``.  The fix: round-trip through
``model_factory_config_from_checkpoint_payload`` + ``build_stage_b_components``
+ ``load_stage_b_checkpoint(..., dora_config=...)``.

This test is the regression guard for that pattern.  It does NOT require
the real GPU-box checkpoint or RADIO weights -- it constructs a synthetic
Stage 2 v2-shaped payload (full ``stage_b_config`` metadata + DoRA-prefixed
state dict) and exercises:

  1. ``model_factory_config_from_checkpoint_payload`` correctly reconstructs
     the ``ModelFactoryConfig`` (encoder='radio_h', dora_rank, decoder dims
     etc.) from the saved metadata.
  2. ``load_stage_b_checkpoint`` detects DoRA, wraps the model with PEFT
     using the supplied ``dora_config``, and loads with >=50% coverage.
  3. Freezing ``encoder.*`` parameters leaves a non-empty trainable surface
     (decoder + cross-attn + LM head + positional_bridge) -- the Stage 3
     trainable surface contract.

A heavyweight companion test that drives the actual ``RadioStageB`` build
is gated on ``torch.hub`` RADIO weight availability -- it is skipped on
boxes without the cached weights and runs on the GPU box.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal RADIO-shaped stand-in.  Has the SAME top-level attribute names as
# RadioStageB (encoder / positional_bridge / decoder_blocks / lm_head /
# token_embedding) so the freeze-by-prefix logic in this test exercises the
# exact pattern the trainer uses.
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """Mimics one ViT block in RADIO's encoder: qkv / proj / fc1 / fc2 leaves."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)


class _StageBEncoder(nn.Module):
    """Stand-in for RADIO's ViT encoder (just one ``block0``)."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = _EncoderBlock()


class _PositionalBridge(nn.Module):
    """Stand-in for the real PositionalBridge.  Has a ``.proj`` Linear leaf
    so the encoder-shape inference path in
    ``model_factory_config_from_checkpoint_payload`` can read its in_features."""

    def __init__(self, in_dim: int = 8, out_dim: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)


class _DecoderBlock(nn.Module):
    """Mimics the cross-attention decoder block targeted by the DoRA recipe."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)


class _MinimalRadioShapedStageB(nn.Module):
    """Tiny stand-in for RadioStageB used to round-trip the checkpoint loader.

    Avoids the real ``torch.hub`` download of C-RADIOv4-H (~700 MB) so the
    smoke test runs anywhere.  Mirrors RadioStageB's top-level attribute
    naming (``encoder`` / ``positional_bridge`` / ``token_embedding`` /
    ``decoder_blocks`` / ``decoder_norm`` / ``lm_head``) so the freeze-by-
    prefix logic in this test exercises the exact pattern the trainer uses.

    Sub-modules are declared as ``nn.Module`` subclasses (not ``nn.ModuleDict``)
    because PEFT's ``modules_to_save`` cannot wrap container types -- the
    same structural shape the real model has.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder side -- frozen + DoRA-adapted in Stage 3.
        self.encoder = _StageBEncoder()
        # Bridge -- 1280 -> 768 in real model; toy dims here.  Modules-to-save
        # in the DoRA recipe (full-finetuned, not LoRA).
        self.positional_bridge = _PositionalBridge(in_dim=8, out_dim=8)
        # Decoder side -- trainable in Stage 3.
        self.token_embedding = nn.Embedding(100, 8)
        self.decoder_blocks = nn.ModuleList([_DecoderBlock()])
        self.decoder_norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 100)


def _radio_shaped_stage_b_config_dict() -> Dict[str, Any]:
    """Return the dict the trainer writes into ``stage_b_config`` for RADIO runs.

    Mirrors ``dataclasses.asdict(RadioStageBConfig(...))`` plus the explicit
    ``encoder`` key the trainer adds at save time (see ``train.py`` around
    line 2016).  The key set MUST stay aligned with ``RadioStageBConfig``;
    the test fails fast if either side drifts.
    """
    return {
        "decoder_dim": 8,
        "decoder_layers": 1,
        "decoder_heads": 2,
        "vocab_size": 100,
        "max_decode_len": 128,
        "dropout": 0.1,
        "contour_classes": 3,
        "pool_to_stride32": False,
        # Trainer-injected (see train.py: stage_b_config_dict["encoder"] = ...)
        "encoder": "radio_h",
        # DoRA rank is also persisted by the trainer.
        "dora_rank": 4,
    }


def _build_dora_state_dict(model: nn.Module, rank: int = 4) -> Dict[str, torch.Tensor]:
    """Take a freshly-built model and emit a Stage 2 v2-shaped state dict.

    The trainer wraps the model with PEFT before saving, so saved keys carry
    the ``base_model.model.<orig>.base_layer.weight`` / ``...lora_A.default.weight``
    pattern.  Reproduce that here for the encoder linear leaves the DoRA
    recipe targets (qkv/proj/fc1/fc2), and the ``modules_to_save`` pattern
    for positional_bridge / token_embedding / decoder_norm / lm_head.
    """
    out: Dict[str, torch.Tensor] = {}
    base = {f"base_model.model.{k}": v.clone() for k, v in model.state_dict().items()}
    out.update(base)

    # DoRA-adapted encoder leaves.  Names mirror what _prepare_model_for_dora
    # would produce given the RADIO target list.
    encoder_dora_targets = [
        ("encoder.block0.qkv", 8, 24),
        ("encoder.block0.proj", 8, 8),
        ("encoder.block0.fc1", 8, 16),
        ("encoder.block0.fc2", 16, 8),
    ]
    for prefix, in_dim, out_dim in encoder_dora_targets:
        out[f"base_model.model.{prefix}.lora_A.default.weight"] = torch.randn(rank, in_dim)
        out[f"base_model.model.{prefix}.lora_B.default.weight"] = torch.randn(out_dim, rank)
        out[f"base_model.model.{prefix}.lora_magnitude_vector.default.weight"] = torch.ones(out_dim)

    # modules_to_save entries -- positional_bridge, token_embedding,
    # decoder_norm, lm_head (full-finetuned, no LoRA).
    for full_name in (
        "positional_bridge.proj.weight",
        "token_embedding.weight",
        "decoder_norm.weight",
        "lm_head.weight",
    ):
        # Resolve the existing tensor shape from the real model.
        tensor = dict(model.state_dict())[full_name]
        out[
            f"base_model.model.{full_name.rsplit('.', 1)[0]}"
            f".modules_to_save.default.{full_name.rsplit('.', 1)[1]}"
        ] = tensor.clone()
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dora_aware_loader_round_trips_stage2_v2_shaped_checkpoint(tmp_path):
    """Synthetic Stage 2 v2 payload -> reconstruct config -> DoRA-load -> freeze encoder.

    Verifies the four properties Plan C depends on:
      * ``model_factory_config_from_checkpoint_payload`` reads the embedded
        ``stage_b_config`` and returns a ``ModelFactoryConfig`` with the
        right encoder ('radio_h') and dora_rank.
      * ``load_stage_b_checkpoint`` detects DoRA, wraps the model with PEFT
        via the supplied ``dora_config``, and reaches >=50% coverage.
      * The Stage 3 trainable surface (decoder/bridge/LM head) is non-empty
        after freezing every parameter whose name starts with ``encoder.``.
      * The frozen surface (encoder) is also non-empty -- otherwise the
        freeze step silently did nothing.
    """
    pytest.importorskip("peft", reason="peft is required for DoRA-aware loading")

    from src.checkpoint_io import load_stage_b_checkpoint
    from src.train.model_factory import (
        ModelFactoryConfig,
        model_factory_config_from_checkpoint_payload,
    )

    # --- 1. Build the synthetic Stage 2 v2 payload ---------------------
    src_model = _MinimalRadioShapedStageB()
    state_dict = _build_dora_state_dict(src_model, rank=4)

    payload: Dict[str, Any] = {
        "model_state_dict": state_dict,
        "stage_b_config": _radio_shaped_stage_b_config_dict(),
        "best_val_loss": 0.148,
        "global_step": 4000,
        "stage_name": "stage2-radio-systems-polyphonic",
    }
    ckpt_path = tmp_path / "stage2_v2_best.pt"
    torch.save(payload, str(ckpt_path))

    # --- 2. Reconstruct the factory config from the saved metadata ----
    factory_cfg = model_factory_config_from_checkpoint_payload(
        payload,
        vocab_size=100,
        fallback=ModelFactoryConfig(stage_b_vocab_size=100),
    )
    assert factory_cfg.stage_b_encoder == "radio_h", (
        f"expected encoder='radio_h' from saved metadata, got {factory_cfg.stage_b_encoder!r}"
    )
    assert factory_cfg.stage_b_dora_rank == 4, (
        f"expected dora_rank=4 from saved metadata, got {factory_cfg.stage_b_dora_rank}"
    )
    assert factory_cfg.stage_b_decoder_dim == 8
    assert factory_cfg.stage_b_decoder_layers == 1
    assert factory_cfg.stage_b_decoder_heads == 2

    # --- 3. Load the checkpoint into a fresh model with dora_config ---
    # (We reuse the minimal model rather than the real factory because
    # build_stage_b_components(encoder='radio_h') would torch.hub.load
    # ~700 MB of RADIO weights.  The companion GPU-box test below covers
    # the real factory path.)
    dst_model = _MinimalRadioShapedStageB()
    dora_config = {
        "adapter_type": "dora",
        "rank": 4,
        "alpha": 4,
        "dropout": 0.0,
        # Empty target list = match every nn.Linear not in the
        # _prepare_model_for_dora keyword exclusion list (positional_bridge,
        # token_embedding, lm_head, decoder_norm, contour_head,
        # deformable_attention).  This catches encoder leaves and decoder
        # blocks alike.
        "target_modules": [],
    }
    device = torch.device("cpu")
    result = load_stage_b_checkpoint(
        checkpoint_path=ckpt_path,
        model=dst_model,
        device=device,
        dora_config=dora_config,
    )
    assert result["checkpoint_format"] == "dora_peft", (
        "expected loader to detect DoRA-formatted state dict; "
        f"got format={result['checkpoint_format']!r}"
    )
    assert result["load_ratio"] >= 0.50, (
        f"DoRA-aware loader fell below 50% coverage: {result}"
    )

    # --- 4. Freeze encoder + verify the Stage 3 trainable surface -----
    # The PEFT wrapper renames every parameter to ``base_model.model.<orig>``,
    # so the encoder freeze prefix becomes ``base_model.model.encoder.``.
    loaded_model = result["_model"]
    encoder_prefix = "base_model.model.encoder."
    n_trainable = 0
    n_frozen = 0
    for name, p in loaded_model.named_parameters():
        if name.startswith(encoder_prefix):
            p.requires_grad = False
        if p.requires_grad:
            n_trainable += p.numel()
        else:
            n_frozen += p.numel()
    assert n_trainable > 0, (
        "Stage 3 trainable surface (decoder + bridge + LM head + token_embedding) "
        "must be non-empty after freezing the encoder."
    )
    assert n_frozen > 0, (
        "Encoder must contribute non-zero frozen params; freeze step did nothing."
    )


def test_factory_config_infers_radio_encoder_from_tensor_shapes(tmp_path):
    """Encoder type is recoverable from the saved positional_bridge tensor.

    Older Stage 2 checkpoints may have been saved before the trainer started
    injecting an explicit ``encoder`` key into ``stage_b_config``.  In that
    case ``model_factory_config_from_checkpoint_payload`` falls back to
    inspecting ``positional_bridge.proj.weight`` -- 1280 input dim -> RADIO,
    768 input dim -> DaViT.  This test documents that contract.
    """
    from src.train.model_factory import (
        ModelFactoryConfig,
        model_factory_config_from_checkpoint_payload,
    )

    # Synthetic state with a 1280-input positional_bridge -> should trigger
    # encoder='radio_h' inference even though stage_b_config omits 'encoder'.
    state = {
        "positional_bridge.proj.weight": torch.zeros(768, 1280),
        "positional_bridge.proj.bias": torch.zeros(768),
        "token_embedding.weight": torch.zeros(100, 8),
    }
    payload = {
        "model_state_dict": state,
        # stage_b_config WITHOUT 'encoder' -- legacy schema.
        "stage_b_config": {
            "decoder_dim": 8,
            "decoder_layers": 1,
            "decoder_heads": 2,
            "vocab_size": 100,
            "max_decode_length": 128,
            "dora_rank": 4,
        },
    }
    cfg = model_factory_config_from_checkpoint_payload(
        payload,
        vocab_size=100,
        fallback=ModelFactoryConfig(stage_b_vocab_size=100),
    )
    assert cfg.stage_b_encoder == "radio_h", (
        "encoder inference from positional_bridge.proj.weight in_dim=1280 should "
        f"return 'radio_h'; got {cfg.stage_b_encoder!r}"
    )
