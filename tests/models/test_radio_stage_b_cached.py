"""Tests for RadioStageB.forward cached_features branch.

These tests run on CPU with a tiny dummy model (vocab_size=10, 2 decoder layers)
to avoid requiring the RADIO hub download. We mock RadioEncoder to return a
deterministic feature map.
"""
from __future__ import annotations

import torch
import pytest


def _build_tiny_model():
    """Build a RadioStageB with stub encoder that doesn't call torch.hub."""
    import torch.nn as nn
    from src.models.radio_stage_b import RadioStageB, RadioStageBConfig

    config = RadioStageBConfig(
        decoder_dim=64,
        decoder_layers=2,
        decoder_heads=4,
        vocab_size=10,
        max_decode_len=16,
        contour_classes=3,
    )
    model = RadioStageB.__new__(RadioStageB)
    nn.Module.__init__(model)
    model.config = config

    # Stub encoder: returns (B, 1280, 2, 4) feature map deterministically
    class _StubEncoder(nn.Module):
        hidden_dim = 1280
        def forward(self, x):
            B = x.shape[0]
            return torch.ones(B, 1280, 2, 4, dtype=x.dtype, device=x.device)

    from src.models.davit_stage_b import DecoderBlock, DeformableContextBlock, PositionalBridge, RMSNorm
    model.encoder = _StubEncoder()
    model.deformable_attention = DeformableContextBlock(dim=1280, heads=4)
    model.positional_bridge = PositionalBridge(encoder_dim=1280, decoder_dim=64)
    model.token_embedding = nn.Embedding(10, 64)
    model.decoder_blocks = nn.ModuleList([DecoderBlock(64, 4) for _ in range(2)])
    model.decoder_norm = RMSNorm(64)
    model.lm_head = nn.Linear(64, 10)
    model.contour_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3))
    model.max_decode_length = 16
    model.eval()
    return model


def test_forward_cached_features_signature_accepted() -> None:
    """forward() must accept cached_features without raising TypeError."""
    model = _build_tiny_model()
    B, seq_tokens, C = 2, 8, 1280
    cached = torch.randn(B, seq_tokens, C)
    tgt = torch.zeros(B, 4, dtype=torch.long)
    with torch.no_grad():
        out = model.forward(cached_features=cached, tgt=tgt, _h16=2, _w16=4)
    assert "logits" in out
    assert "contour_logits" in out


def test_forward_cached_matches_live_to_1e3() -> None:
    """Cached path output must match live path output to ≤ 1e-3 max abs diff.

    Both paths use the same stub encoder (deterministic ones). We:
      1. Run live forward: image → encoder → deformable_attn → bridge → decoder
      2. Capture the encoder output (feature_map from stub: ones tensor)
      3. Flatten to (seq_tokens, 1280), run cached forward
      4. Assert logits match to ≤ 1e-3
    """
    model = _build_tiny_model()
    B, H, W = 1, 32, 64  # dummy image; stub encoder ignores content
    image = torch.rand(B, 1, H, W)
    tgt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        live_out = model.forward(image=image, tgt=tgt)
        # Manually extract encoder output for the cached path
        feature_map = model.encoder(image)  # (B, 1280, 2, 4)
        h16, w16 = feature_map.shape[2], feature_map.shape[3]
        cached_tensor = feature_map.flatten(2).transpose(1, 2)  # (B, 8, 1280)
        cached_out = model.forward(cached_features=cached_tensor, tgt=tgt, _h16=h16, _w16=w16)

    diff = (live_out["logits"].float() - cached_out["logits"].float()).abs().max().item()
    assert diff <= 1e-3, f"max abs diff {diff} exceeds 1e-3 tolerance"


def test_forward_raises_without_image_or_cached() -> None:
    """forward() with neither image nor cached_features must raise ValueError."""
    model = _build_tiny_model()
    tgt = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="requires"):
        model.forward(tgt=tgt)


def test_cached_features_skips_encoder_call() -> None:
    """When cached_features is provided, encoder.forward must NOT be called."""
    model = _build_tiny_model()
    call_count = [0]
    original_forward = model.encoder.forward

    def counting_forward(x):
        call_count[0] += 1
        return original_forward(x)

    model.encoder.forward = counting_forward

    cached = torch.randn(1, 8, 1280)
    tgt = torch.zeros(1, 3, dtype=torch.long)
    with torch.no_grad():
        model.forward(cached_features=cached, tgt=tgt, _h16=2, _w16=4)

    assert call_count[0] == 0, "encoder.forward was called despite cached_features being provided"
