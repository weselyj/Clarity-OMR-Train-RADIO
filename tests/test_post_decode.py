"""Unit tests for src/pipeline/post_decode.py.

Pure-logic functions (no torch/CUDA), so located at the top of tests/
rather than tests/pipeline/ to avoid the conftest.py CUDA gate.
"""
from __future__ import annotations

from src.pipeline.post_decode import (
    clean_system_tokens,
    split_system_tokens_into_staves,
    _drop_phantom_chunks,
    _maybe_swap_clef_for_bass_register,
    _median_note_octave,
    _note_event_signature,
)


# --- Helpers --------------------------------------------------------------


def _staff(*body) -> list:
    """Build a complete staff chunk (<staff_start>...<staff_end>)."""
    return ["<staff_start>", *body, "<staff_end>"]


def _grand_staff_system(treble_body, bass_body, *, bass_clef="clef-F4") -> list:
    """Build a 2-staff system: <bos> <treble> <bass> <eos>."""
    return [
        "<bos>",
        *_staff("<staff_idx_0>", "clef-G2", "keySignature-CM", "timeSignature-4/4", *treble_body),
        *_staff("<staff_idx_1>", bass_clef, "keySignature-CM", "timeSignature-4/4", *bass_body),
        "<eos>",
    ]


# --- split_system_tokens_into_staves --------------------------------------


def test_split_returns_one_chunk_per_staff():
    sys = _grand_staff_system(
        treble_body=["note-G4", "_quarter"],
        bass_body=["note-C3", "_quarter"],
    )
    chunks = split_system_tokens_into_staves(sys)
    assert len(chunks) == 2
    assert chunks[0][0] == "<staff_start>"
    assert chunks[0][-1] == "<staff_end>"
    assert chunks[1][0] == "<staff_start>"
    assert chunks[1][-1] == "<staff_end>"


def test_split_handles_missing_final_staff_end():
    """Decoder may truncate before <staff_end> on the final staff."""
    sys = ["<bos>", "<staff_start>", "clef-G2", "note-G4", "_quarter"]
    chunks = split_system_tokens_into_staves(sys)
    assert len(chunks) == 1
    assert "<staff_end>" not in chunks[0]
    assert "note-G4" in chunks[0]


# --- _median_note_octave + _note_event_signature --------------------------


def test_median_note_octave_treble_range():
    chunk = _staff("clef-G2", "note-G4", "_quarter", "note-E5", "_quarter")
    assert _median_note_octave(chunk) == 4.5


def test_median_note_octave_bass_range():
    chunk = _staff("clef-F4", "note-C3", "_quarter", "note-G3", "_quarter")
    assert _median_note_octave(chunk) == 3.0


def test_median_note_octave_none_when_no_notes():
    chunk = _staff("clef-G2", "rest", "_whole")
    assert _median_note_octave(chunk) is None


def test_note_event_signature_strips_clef_and_staff_idx():
    a = _staff("<staff_idx_1>", "clef-G2", "note-C3", "_quarter")
    b = _staff("<staff_idx_2>", "clef-F4", "note-C3", "_quarter")
    assert _note_event_signature(a) == _note_event_signature(b)


# --- _drop_phantom_chunks: real failure modes -----------------------------


def test_drop_phantom_all_rest_chunk_when_sibling_has_notes():
    """TimeMachine sys 3 pattern: chunk 2 is all-rest, chunk 3 has real notes."""
    treble = _staff("clef-G2", "note-G4", "_quarter")
    phantom = _staff("clef-G2", "rest", "_whole", "rest", "_whole")
    real_bass = _staff("clef-F4", "note-C3", "_quarter", "note-G3", "_quarter")
    out = _drop_phantom_chunks([treble, phantom, real_bass])
    assert len(out) == 2
    assert any("note-G4" in c for c in out)
    assert any("note-C3" in c for c in out)
    assert all("rest" not in c or "note-" in " ".join(c) for c in out)


def test_drop_phantom_duplicate_chunks_keeps_bass_clef_for_low_content():
    """TimeMachine sys 1 pattern: same bass content emitted twice under
    different clefs; keep the bass-clef version."""
    treble = _staff("clef-G2", "note-G4", "_quarter")
    duplicate_treble = _staff(
        "clef-G2",
        "rest", "_whole",
        "<chord_start>", "note-C3", "note-G3", "<chord_end>", "_quarter",
    )
    duplicate_bass = _staff(
        "clef-F4",
        "rest", "_whole",
        "<chord_start>", "note-C3", "note-G3", "<chord_end>", "_quarter",
    )
    out = _drop_phantom_chunks([treble, duplicate_treble, duplicate_bass])
    assert len(out) == 2
    # The bass-clef version should be kept, not the treble.
    last = out[-1]
    assert "clef-F4" in last
    assert "clef-G2" not in [t for t in last if t.startswith("clef-")]


def test_no_phantom_drop_when_only_two_staves():
    """Never collapse 2 staves — that's a normal grand-staff system."""
    treble = _staff("clef-G2", "note-G4", "_quarter")
    bass = _staff("clef-F4", "note-C3", "_quarter")
    out = _drop_phantom_chunks([treble, bass])
    assert len(out) == 2


def test_no_phantom_drop_when_three_real_staves():
    """Three legitimate staves (e.g. organ pedal) should NOT be collapsed.
    All have notes, all have distinct signatures."""
    manual1 = _staff("clef-G2", "note-G4", "_quarter")
    manual2 = _staff("clef-G2", "note-E4", "_quarter")
    pedal = _staff("clef-F4", "note-C2", "_quarter")
    out = _drop_phantom_chunks([manual1, manual2, pedal])
    assert len(out) == 3


# --- _maybe_swap_clef_for_bass_register -----------------------------------


def test_swap_treble_to_bass_when_notes_below_middle_c():
    """Bethlehem / TimeMachine sys 2 pattern."""
    chunk = _staff("clef-G2", "note-C3", "_quarter", "note-G3", "_quarter")
    repaired = _maybe_swap_clef_for_bass_register(chunk)
    assert "clef-F4" in repaired
    assert "clef-G2" not in [t for t in repaired if t.startswith("clef-")]


def test_no_swap_when_notes_in_treble_range():
    chunk = _staff("clef-G2", "note-G4", "_quarter", "note-E5", "_quarter")
    repaired = _maybe_swap_clef_for_bass_register(chunk)
    assert "clef-G2" in repaired
    assert "clef-F4" not in repaired


def test_no_swap_when_clef_already_bass():
    chunk = _staff("clef-F4", "note-C3", "_quarter")
    repaired = _maybe_swap_clef_for_bass_register(chunk)
    assert repaired == chunk


def test_no_swap_when_chunk_has_no_notes():
    chunk = _staff("clef-G2", "rest", "_whole")
    repaired = _maybe_swap_clef_for_bass_register(chunk)
    assert repaired == chunk


# --- clean_system_tokens (end-to-end) -------------------------------------


def test_clean_collapses_phantom_and_repairs_clef():
    """TimeMachine sys 1: 3 staves, phantom duplicate of bass under treble clef.
    Bottom (real) staff already has F4 — should pass unchanged."""
    sys = [
        "<bos>",
        *_staff("clef-G2", "note-G4", "_quarter"),
        *_staff("clef-G2", "rest", "_whole",
                "<chord_start>", "note-C3", "note-G3", "<chord_end>", "_quarter"),
        *_staff("clef-F4", "rest", "_whole",
                "<chord_start>", "note-C3", "note-G3", "<chord_end>", "_quarter"),
        "<eos>",
    ]
    cleaned = clean_system_tokens(sys)
    chunks = split_system_tokens_into_staves(cleaned)
    assert len(chunks) == 2
    # Last chunk should be bass-clef (we dropped the duplicate-treble one)
    assert _maybe_swap_clef_for_bass_register  # imported
    assert "clef-F4" in chunks[-1]


def test_clean_is_noop_on_well_formed_grand_staff():
    sys = _grand_staff_system(
        treble_body=["note-G4", "_quarter"],
        bass_body=["note-C3", "_quarter"],
    )
    cleaned = clean_system_tokens(sys)
    assert cleaned == list(sys)


def test_clean_disables_via_kwargs():
    """When both flags are False, the function is a no-op even for known
    failure patterns."""
    sys = _grand_staff_system(
        treble_body=["note-G4", "_quarter"],
        bass_body=["note-C3", "_quarter"],  # below middle C
        bass_clef="clef-G2",  # misread
    )
    cleaned = clean_system_tokens(sys, drop_phantom_staves=False, repair_bass_clef=False)
    assert cleaned == list(sys)


def test_clean_swap_bass_clef_on_bottom_staff_only_when_opted_in():
    """The bass-clef repair fires only when repair_bass_clef=True (it's an
    experimental heuristic, default-off due to pitch-correctness concerns;
    see clean_system_tokens docstring)."""
    sys = _grand_staff_system(
        treble_body=["note-G4", "_quarter"],
        bass_body=["note-C3", "_quarter", "note-G3", "_quarter"],
        bass_clef="clef-G2",  # the misread we want to fix
    )
    # Default off: clef stays G2 (just-rejoined system)
    default = clean_system_tokens(sys)
    default_chunks = split_system_tokens_into_staves(default)
    assert "clef-G2" in default_chunks[-1]
    # Explicit opt-in: clef swapped to F4
    opted = clean_system_tokens(sys, repair_bass_clef=True)
    opted_chunks = split_system_tokens_into_staves(opted)
    assert "clef-F4" in opted_chunks[-1]


def test_clean_does_not_swap_clef_on_top_staff_even_with_opt_in():
    """A treble clef on the TOP staff with low-register notes is rare but
    possible (eg. left-hand alone passage); don't repair the top staff
    even when repair_bass_clef=True."""
    sys = [
        "<bos>",
        *_staff("clef-G2", "note-C3", "_quarter"),  # top with low notes
        *_staff("clef-G2", "note-G4", "_quarter"),  # bottom with high notes
        "<eos>",
    ]
    cleaned = clean_system_tokens(sys, repair_bass_clef=True)
    chunks = split_system_tokens_into_staves(cleaned)
    # Top staff clef unchanged
    assert "clef-G2" in chunks[0]
    assert "clef-F4" not in chunks[0]
