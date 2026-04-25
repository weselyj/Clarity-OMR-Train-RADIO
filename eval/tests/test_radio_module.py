"""Smoke tests for the RADIO Stage B module.

Run before integrating into model_factory."""
import torch
import pytest


def test_radio_module_loads():
    from src.models.radio_stage_b import build_radio_stage_b
    model = build_radio_stage_b(
        decoder_dim=768,
        decoder_layers=8,
        decoder_heads=12,
        vocab_size=487,
        max_decode_len=512,
    )
    assert model is not None


def test_radio_forward_shapes():
    from src.models.radio_stage_b import build_radio_stage_b
    model = build_radio_stage_b(
        decoder_dim=768, decoder_layers=8, decoder_heads=12,
        vocab_size=487, max_decode_len=512,
    ).cuda().eval()
    img = torch.rand(1, 1, 192, 1024).cuda()  # 1-channel grayscale
    tgt = torch.zeros(1, 32, dtype=torch.long).cuda()
    with torch.no_grad():
        out = model(img, tgt)
    assert "logits" in out, f"expected dict with 'logits' key, got: {out.keys() if hasattr(out, 'keys') else type(out)}"
    assert out["logits"].shape == (1, 32, 487), f"unexpected logits shape: {out['logits'].shape}"
    assert out["contour_logits"].shape == (1, 3), f"unexpected contour_logits shape: {out['contour_logits'].shape}"
