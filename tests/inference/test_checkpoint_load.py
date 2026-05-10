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


def test_load_stage_b_for_inference_chains_existing_helpers(monkeypatch, tmp_path):
    """Verify the loader exercises the real factory_cfg + components chain
    while mocking heavy torch.load + checkpoint loading + decoder prep."""
    from unittest.mock import MagicMock

    import src.inference.checkpoint_load as cl

    # Mock torch.load to return a fake payload
    fake_payload = {"factory_config": {"stage_b_vocab_size": 1234, "decoder_d_model": 512}}
    monkeypatch.setattr(cl, "torch", MagicMock(load=MagicMock(return_value=fake_payload)))

    fake_vocab = MagicMock()
    fake_vocab.size = 1234
    fake_vocab.tokens = ["<bos>", "<eos>", "<staff_end>"]
    monkeypatch.setattr(cl, "build_default_vocabulary", lambda: fake_vocab)

    fake_factory_cfg = MagicMock()
    monkeypatch.setattr(
        cl,
        "model_factory_config_from_checkpoint_payload",
        lambda payload, vocab_size, fallback: fake_factory_cfg,
    )

    fake_model = MagicMock()
    fake_dora_cfg = MagicMock()
    monkeypatch.setattr(
        cl,
        "build_stage_b_components",
        lambda factory_cfg: {"model": fake_model, "dora_config": fake_dora_cfg},
    )

    monkeypatch.setattr(
        cl,
        "load_stage_b_checkpoint",
        lambda **kwargs: {"_model": kwargs["model"], "checkpoint_format": "v1",
                          "loaded_keys": [], "load_ratio": 1.0},
    )

    fake_decode_model = MagicMock()
    monkeypatch.setattr(
        cl,
        "_prepare_model_for_inference",
        lambda model, device, *, use_fp16=False, quantize=False: (fake_decode_model, use_fp16),
    )

    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"")  # exists for path validation

    bundle = cl.load_stage_b_for_inference(ckpt, device="cpu", use_fp16=True)

    assert bundle.model is fake_model
    assert bundle.decode_model is fake_decode_model
    assert bundle.vocab is fake_vocab
    assert bundle.use_fp16 is True
    assert bundle.factory_cfg is fake_factory_cfg
    assert bundle.token_to_idx == {"<bos>": 0, "<eos>": 1, "<staff_end>": 2}
    fake_model.eval.assert_called_once()
