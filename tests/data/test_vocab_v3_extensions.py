"""Tests for the v3 enharmonic vocab extension (Cb/Fb/B#/E# × 5 octaves = 20 tokens)."""
from __future__ import annotations

from src.tokenizer.vocab import build_default_vocabulary


def test_cb_fb_bsharp_esharp_present_in_vocab() -> None:
    v = build_default_vocabulary()
    expected = (
        [f"note-Cb{o}" for o in range(2, 7)]
        + [f"note-Fb{o}" for o in range(2, 7)]
        + [f"note-B#{o}" for o in range(2, 7)]
        + [f"note-E#{o}" for o in range(2, 7)]
    )
    for tok in expected:
        assert tok in v.token_to_id, f"{tok} missing from vocab"


def test_v3_tokens_appended_at_ids_388_to_407() -> None:
    """v3 extension must append tokens starting at ID 388 (after the v2 staff_idx block)."""
    v = build_default_vocabulary()
    cb2_id = v.token_to_id.get("note-Cb2")
    assert cb2_id == 388, f"note-Cb2 expected at ID 388, got {cb2_id}"
    esharp6_id = v.token_to_id.get("note-E#6")
    assert esharp6_id == 407, f"note-E#6 expected at ID 407, got {esharp6_id}"


def test_existing_vocab_token_ids_preserved() -> None:
    """The v3 extension must append, not insert — existing IDs unchanged."""
    v = build_default_vocabulary()
    assert v.token_to_id["<bos>"] == 0
    assert v.token_to_id["<eos>"] == 1
    assert v.token_to_id["<staff_start>"] == 3
    assert v.token_to_id["<staff_end>"] == 4
    assert v.token_to_id["<voice_1>"] == 7
    # Staff-idx block unchanged at 380-387
    assert v.token_to_id["<staff_idx_0>"] == 380
    assert v.token_to_id["<staff_idx_7>"] == 387


def test_v3_tokens_in_note_tokens_set() -> None:
    """The 20 new tokens must appear in vocab.note_tokens (used by grammar FSA)."""
    v = build_default_vocabulary()
    for tok in [f"note-Cb{o}" for o in range(2, 7)]:
        assert tok in v.note_tokens, f"{tok} missing from note_tokens set"
    for tok in [f"note-Fb{o}" for o in range(2, 7)]:
        assert tok in v.note_tokens, f"{tok} missing from note_tokens set"
    for tok in [f"note-B#{o}" for o in range(2, 7)]:
        assert tok in v.note_tokens, f"{tok} missing from note_tokens set"
    for tok in [f"note-E#{o}" for o in range(2, 7)]:
        assert tok in v.note_tokens, f"{tok} missing from note_tokens set"


def test_octave_1_note_tokens_present() -> None:
    """v3 phase 2: octave-1 sub-bass notes are now in vocab."""
    v = build_default_vocabulary()
    expected = [
        "note-A1", "note-Ab1", "note-A#1",
        "note-B1", "note-Bb1", "note-B#1",
        "note-C1", "note-Cb1", "note-C#1",
        "note-D1", "note-Db1", "note-D#1",
        "note-E1", "note-Eb1", "note-E#1",
        "note-F1", "note-Fb1", "note-F#1",
        "note-G1", "note-Gb1", "note-G#1",
    ]
    for tok in expected:
        assert tok in v.token_to_id, f"{tok} missing from vocab"
    assert len(v.token_to_id) == 408 + 21  # was 408 after v3 enharmonic extension
