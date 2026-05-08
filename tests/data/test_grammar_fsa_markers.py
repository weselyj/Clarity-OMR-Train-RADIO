"""Verify the grammar FSA accepts <staff_idx_N> immediately after <staff_start>."""
from __future__ import annotations

import pytest

from src.decoding.grammar_fsa import GrammarFSA


def test_marker_token_accepted_after_staff_start() -> None:
    fsa = GrammarFSA()
    fsa.validate_sequence(
        [
            "<bos>",
            "<staff_start>",
            "<staff_idx_0>",
            "clef-G2",
            "timeSignature-4/4",
            "<measure_start>",
            "note-C4",
            "_quarter",
            "<measure_end>",
            "<staff_end>",
            "<eos>",
        ],
        strict=True,
    )


def test_marker_token_rejected_if_not_after_staff_start() -> None:
    fsa = GrammarFSA()
    with pytest.raises(ValueError):
        fsa.validate_sequence(["<bos>", "<staff_idx_0>"], strict=True)


def test_two_staff_sequence_with_markers() -> None:
    fsa = GrammarFSA()
    fsa.validate_sequence(
        [
            "<bos>",
            "<staff_start>", "<staff_idx_0>",
            "clef-G2", "timeSignature-4/4",
            "<measure_start>", "note-C4", "_quarter", "<measure_end>",
            "<staff_end>",
            "<staff_start>", "<staff_idx_1>",
            "clef-F4", "timeSignature-4/4",
            "<measure_start>", "note-C2", "_quarter", "<measure_end>",
            "<staff_end>",
            "<eos>",
        ],
        strict=True,
    )
