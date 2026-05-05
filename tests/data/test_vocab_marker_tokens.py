"""Verify the v2 vocab adds <staff_idx_N> tokens at the end without shifting existing IDs."""
from __future__ import annotations

from src.tokenizer.vocab import build_default_vocabulary, build_default_token_list


def test_vocab_size_is_429() -> None:
    """v3 phase 2 adds 21 octave-1 sub-bass tokens on top of the 408 from v3 phase 1."""
    vocab = build_default_vocabulary()
    assert vocab.size == 429


def test_marker_tokens_present() -> None:
    vocab = build_default_vocabulary()
    for i in range(8):
        token = f"<staff_idx_{i}>"
        assert token in vocab.token_to_id, f"{token} missing from vocab"
    # Staff-idx tokens must be in ascending ID order and occupy a contiguous block.
    new_ids = [vocab.token_to_id[f"<staff_idx_{i}>"] for i in range(8)]
    assert new_ids == sorted(new_ids), "marker tokens not in ascending ID order"
    assert new_ids == list(range(new_ids[0], new_ids[0] + 8)), "marker token IDs not contiguous"
    # IDs 380-387 (the v2 block) must still be the staff_idx tokens.
    assert min(new_ids) == 380
    assert max(new_ids) == 387


def test_existing_token_ids_unchanged() -> None:
    """Sanity: pre-existing structural and music tokens keep their original IDs."""
    vocab = build_default_vocabulary()
    # Spot-check: ids 0-16 are STRUCTURAL_TOKENS (existing).
    assert vocab.token_to_id["<bos>"] == 0
    assert vocab.token_to_id["<eos>"] == 1
    assert vocab.token_to_id["<staff_start>"] == 3
    assert vocab.token_to_id["<staff_end>"] == 4
    # <voice_1> stays at id 7 (was at id 7 before).
    assert vocab.token_to_id["<voice_1>"] == 7


def test_marker_tokens_in_structural_set() -> None:
    vocab = build_default_vocabulary()
    for i in range(8):
        assert f"<staff_idx_{i}>" in vocab.structural_tokens


def test_token_list_has_no_duplicates() -> None:
    tokens = build_default_token_list()
    assert len(tokens) == len(set(tokens))
