"""Tests for the --torch-compile flag (cu132 plan, Phase 4.1).

Verifies that the flag wires through to torch.compile() applied to the
decoder and positional_bridge submodules — explicitly NOT the encoder
(trust_remote_code RADIO is a known compile hazard).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.train import _maybe_compile_decoder_and_bridge


class _DummyStageB:
    """Minimal stand-in for the Stage-B model with the three submodules
    the helper inspects: encoder (skipped), decoder, positional_bridge."""

    def __init__(self):
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)
        self.positional_bridge = nn.Linear(4, 4)


def test_disabled_returns_model_unchanged():
    m = _DummyStageB()
    enc, dec, pb = m.encoder, m.decoder, m.positional_bridge
    out = _maybe_compile_decoder_and_bridge(m, enabled=False)
    assert out is m
    assert out.encoder is enc
    assert out.decoder is dec
    assert out.positional_bridge is pb


def test_enabled_compiles_only_decoder_and_bridge():
    m = _DummyStageB()
    encoder_orig = m.encoder
    decoder_orig = m.decoder
    bridge_orig = m.positional_bridge

    sentinel_calls: list = []

    def fake_compile(mod, **kwargs):
        sentinel_calls.append((id(mod), kwargs))
        return f"compiled-{id(mod)}"

    with patch("torch.compile", side_effect=fake_compile):
        out = _maybe_compile_decoder_and_bridge(m, enabled=True)

    # encoder must NOT have been touched
    assert m.encoder is encoder_orig

    # decoder + bridge replaced with the compile sentinel
    assert m.decoder == f"compiled-{id(decoder_orig)}"
    assert m.positional_bridge == f"compiled-{id(bridge_orig)}"

    # exactly two compile calls (decoder + bridge), neither for encoder
    assert len(sentinel_calls) == 2
    compiled_ids = {call[0] for call in sentinel_calls}
    assert id(decoder_orig) in compiled_ids
    assert id(bridge_orig) in compiled_ids
    assert id(encoder_orig) not in compiled_ids


def test_missing_attrs_are_tolerated():
    """Helper must not crash if a submodule attr is absent."""

    class Partial:
        decoder = nn.Linear(4, 4)
        # no positional_bridge, no encoder

    p = Partial()
    decoder_orig = p.decoder
    with patch("torch.compile", side_effect=lambda mod, **kwargs: "x"):
        out = _maybe_compile_decoder_and_bridge(p, enabled=True)
    assert p.decoder == "x"
    assert not hasattr(p, "positional_bridge")
    assert not hasattr(p, "encoder")


def test_help_mentions_torch_compile_flag(capsys):
    """The CLI help output advertises the new flag."""
    from src.train.train import parse_args
    with patch.object(sys, "argv", ["train.py", "--help"]):
        with pytest.raises(SystemExit):
            parse_args()
    captured = capsys.readouterr()
    assert "--torch-compile" in captured.out
