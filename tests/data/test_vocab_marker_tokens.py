"""Verify the v2 vocab adds <staff_idx_N> tokens at the end without shifting existing IDs."""
from __future__ import annotations

from src.tokenizer.vocab import build_default_vocabulary, build_default_token_list


def test_vocab_size_is_388() -> None:
    vocab = build_default_vocabulary()
    assert vocab.size == 388


def test_marker_tokens_present_at_end() -> None:
    vocab = build_default_vocabulary()
    for i in range(8):
        token = f"<staff_idx_{i}>"
        assert token in vocab.token_to_id, f"{token} missing from vocab"
    # New tokens are at the end (highest IDs).
    new_ids = [vocab.token_to_id[f"<staff_idx_{i}>"] for i in range(8)]
    assert new_ids == sorted(new_ids), "marker tokens not in ascending ID order"
    assert max(new_ids) == vocab.size - 1
    assert min(new_ids) == vocab.size - 8


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
