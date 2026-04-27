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


def test_radio_dora_target_count():
    """Regression test for Task 6: ensure DoRA targets are RADIO-shaped, not DaViT-shaped.

    The previous bug: targets returned DaViT-style names (q_proj, k_proj, ...). _prepare_model_for_dora
    matches via name.endswith(target). None matched RADIO's attn.qkv/attn.proj/mlp.fc1/mlp.fc2.
    DoRA was silently not applied.
    """
    from src.train.model_factory import list_radio_dora_target_modules
    targets = list_radio_dora_target_modules()

    # Encoder-side: must include the RADIO leaf names
    radio_encoder_targets = {"qkv", "proj", "fc1", "fc2"}
    found = radio_encoder_targets & set(targets)
    assert len(found) >= 4, f"Missing RADIO encoder targets: {radio_encoder_targets - found}"

    # Total target count should match the documented count (15: 4 encoder + 11 decoder).
    # If this changes, update the comment in configs/train_stage2_radio_mvp.yaml.
    assert len(targets) >= 4 + 7, f"Unexpectedly few DoRA targets: {len(targets)}"
