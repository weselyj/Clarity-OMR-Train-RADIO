"""Tests for src/data/convert_tokens.py::convert_kern_file (multi-spine support)."""
from __future__ import annotations

from pathlib import Path

from src.data.convert_tokens import convert_kern_file


def _write_kern(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sample.krn"
    p.write_text(content, encoding="utf-8")
    return p


def test_single_spine_emits_no_marker(tmp_path: Path) -> None:
    """A 1-spine kern file produces a sequence without <staff_idx_*>."""
    krn = _write_kern(
        tmp_path,
        "**kern\n"
        "*clefG2\n"
        "*k[]\n"
        "*M4/4\n"
        "=1\n"
        "4c\n"
        "*-\n",
    )
    tokens = convert_kern_file(krn)
    assert "<bos>" in tokens
    assert "<eos>" in tokens
    assert "<staff_start>" in tokens
    assert "<staff_end>" in tokens
    assert not any(t.startswith("<staff_idx_") for t in tokens)


def test_two_spine_grand_staff_reverses_columns_for_top_down(tmp_path: Path) -> None:
    """A 2-spine kern (bass column 0, treble column 1) emits treble first as <staff_idx_0>."""
    krn = _write_kern(
        tmp_path,
        "**kern\t**kern\n"
        "*clefF4\t*clefG2\n"
        "*k[]\t*k[]\n"
        "*M4/4\t*M4/4\n"
        "=1\t=1\n"
        "4C\t4c\n"
        "*-\t*-\n",
    )
    tokens = convert_kern_file(krn)

    # Two staff_start blocks, top-down (treble first).
    assert tokens.count("<staff_start>") == 2
    assert tokens.count("<staff_end>") == 2
    idx0_pos = tokens.index("<staff_idx_0>")
    idx1_pos = tokens.index("<staff_idx_1>")
    assert idx0_pos < idx1_pos

    # Idx 0 (treble) should appear immediately after first <staff_start>.
    first_start = tokens.index("<staff_start>")
    assert tokens[first_start + 1] == "<staff_idx_0>"

    # Treble's clef is G2; bass clef F4. Verify ordering by clef.
    g2_pos = tokens.index("clef-G2")
    f4_pos = tokens.index("clef-F4")
    assert g2_pos < f4_pos, "treble (G2) block must precede bass (F4) block"


def test_three_spine_top_down_ordering(tmp_path: Path) -> None:
    """3-spine kern emits idx_0, idx_1, idx_2 in top-down (kern column N-1 to 0) order."""
    krn = _write_kern(
        tmp_path,
        "**kern\t**kern\t**kern\n"
        "*clefF4\t*clefC4\t*clefG2\n"
        "*k[]\t*k[]\t*k[]\n"
        "*M4/4\t*M4/4\t*M4/4\n"
        "=1\t=1\t=1\n"
        "4C\t4e\t4g\n"
        "*-\t*-\t*-\n",
    )
    tokens = convert_kern_file(krn)
    assert tokens.count("<staff_start>") == 3
    assert tokens.index("<staff_idx_0>") < tokens.index("<staff_idx_1>") < tokens.index("<staff_idx_2>")
    # Top-down: G2 (rightmost column) first, F4 (leftmost column) last.
    assert tokens.index("clef-G2") < tokens.index("clef-C4") < tokens.index("clef-F4")


def test_sub_spine_split_emits_voices_within_spine(tmp_path: Path) -> None:
    """A spine that splits via *^ produces <voice_N> markers within its <staff_start>...<staff_end>."""
    krn = _write_kern(
        tmp_path,
        "**kern\t**kern\n"
        "*clefF4\t*clefG2\n"
        "*k[]\t*k[]\n"
        "*M4/4\t*M4/4\n"
        "=1\t=1\n"
        "*\t*^\n"
        "4C\t4c\t4e\n"
        "*\t*v\t*v\n"
        "4C\t4c\n"
        "*-\t*-\n",
    )
    tokens = convert_kern_file(krn)
    # Treble (idx_0) should contain voice markers; bass (idx_1) should not.
    treble_start = tokens.index("<staff_idx_0>")
    bass_start = tokens.index("<staff_idx_1>")
    treble_block = tokens[treble_start:bass_start]
    bass_block = tokens[bass_start:]
    assert "<voice_1>" in treble_block
    assert "<voice_2>" in treble_block
    assert "<voice_2>" not in bass_block


def test_voice_index_overflow_drops_excess(tmp_path: Path) -> None:
    """A spine with 5 sub-spines (>MAX_SUPPORTED_VOICE_INDEX=4) drops the 5th."""
    krn = _write_kern(
        tmp_path,
        "**kern\n"
        "*clefG2\n"
        "*k[]\n"
        "*M4/4\n"
        "*^\n"
        "*\t*^\n"
        "*\t*\t*^\n"
        "*\t*\t*\t*^\n"
        "=1\t=1\t=1\t=1\t=1\n"
        "4c\t4d\t4e\t4f\t4g\n"
        "*-\t*-\t*-\t*-\t*-\n",
    )
    tokens = convert_kern_file(krn)
    # We expect voices 1..4 to appear; voice 5 dropped silently (no <voice_5> token).
    assert "<voice_1>" in tokens
    assert "<voice_4>" in tokens
    assert "<voice_5>" not in tokens
    # The dropped pitch ("g") should not appear.
    assert "note-G4" not in tokens


def test_malformed_no_kern_header_returns_empty(tmp_path: Path) -> None:
    """A file without **kern header returns []."""
    krn = _write_kern(
        tmp_path,
        "this is not a kern file\nsome content\n",
    )
    tokens = convert_kern_file(krn)
    assert tokens == []


def test_empty_file_returns_empty(tmp_path: Path) -> None:
    krn = _write_kern(tmp_path, "")
    assert convert_kern_file(krn) == []


def test_simple_kern_token_output_unchanged_after_refactor(tmp_path: Path) -> None:
    """Regression baseline: the simplest 1-note kern produces a known token list."""
    krn = _write_kern(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*-\n",
    )
    tokens = convert_kern_file(krn)
    # Expected: <bos> <staff_start> clef-G2 keySignature-CM timeSignature-4/4 <measure_start> note-C4 _quarter <measure_end> <staff_end> <eos>
    assert tokens[0] == "<bos>"
    assert tokens[-1] == "<eos>"
    assert "<staff_start>" in tokens
    assert "<staff_end>" in tokens
    assert "note-C4" in tokens
    assert "_quarter" in tokens


def test_tie_open_emits_tie_start(tmp_path: Path) -> None:
    krn = _write_kern(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c[\n=2\n4c]\n*-\n",
    )
    tokens = convert_kern_file(krn)
    assert "tie_start" in tokens
    assert "tie_end" in tokens


def test_tie_token_emits_before_pitch(tmp_path: Path) -> None:
    """Canonical order: tie_start emitted before the note tokens it ties to."""
    krn = _write_kern(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c[\n=2\n4c]\n*-\n",
    )
    tokens = convert_kern_file(krn)
    tie_start_idx = tokens.index("tie_start")
    note_c4_first = next(i for i, t in enumerate(tokens) if t == "note-C4")
    assert tie_start_idx < note_c4_first


def test_slur_open_close_emit_tokens(tmp_path: Path) -> None:
    krn = _write_kern(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c(\n4d)\n*-\n",
    )
    tokens = convert_kern_file(krn)
    assert "slur_start" in tokens
    assert "slur_end" in tokens


def test_accent_emits_accent_token(tmp_path: Path) -> None:
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c^\n*-\n")
    tokens = convert_kern_file(krn)
    assert "accent" in tokens


def test_staccato_emits_staccato_token(tmp_path: Path) -> None:
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c'\n*-\n")
    tokens = convert_kern_file(krn)
    assert "staccato" in tokens


def test_fermata_emits_fermata_token(tmp_path: Path) -> None:
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c;\n*-\n")
    tokens = convert_kern_file(krn)
    assert "fermata" in tokens


def test_articulation_emits_before_pitch(tmp_path: Path) -> None:
    """Canonical order: articulations come before pitch (and after ornaments, but no ornaments here)."""
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c^\n*-\n")
    tokens = convert_kern_file(krn)
    accent_idx = tokens.index("accent")
    note_idx = tokens.index("note-C4")
    assert accent_idx < note_idx


def test_trill_emits_trill_token(tmp_path: Path) -> None:
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4ct\n*-\n")
    tokens = convert_kern_file(krn)
    assert "trill" in tokens


def test_mordent_emits_mordent_token(tmp_path: Path) -> None:
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4cm\n*-\n")
    tokens = convert_kern_file(krn)
    assert "mordent" in tokens


def test_ornament_emits_before_articulation(tmp_path: Path) -> None:
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4ct^\n*-\n")
    tokens = convert_kern_file(krn)
    trill_idx = tokens.index("trill")
    accent_idx = tokens.index("accent")
    assert trill_idx < accent_idx


def test_kern_24_is_sixteenth_triplet_not_eighth_triplet(tmp_path: Path) -> None:
    """Kern duration code 24 = sixteenth-triplet (3 in time of 2 sixteenths).
    The math: kern_code = base × N/M = 16 × 3/2 = 24. Inversely, base = 24 × 2/3 = 16."""
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n24c\n*-\n")
    tokens = convert_kern_file(krn)
    assert "<tuplet_3>" in tokens
    assert "_sixteenth" in tokens
    # The buggy version produced _eighth.
    assert "_eighth" not in tokens


def test_kern_48_can_be_thirty_second_tuplet(tmp_path: Path) -> None:
    """Kern duration 48: base = 48 × 2/3 = 32 (32nd note). With tuplet_3 OR tuplet_6 grouping."""
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n48c\n*-\n")
    tokens = convert_kern_file(krn)
    assert "_thirty_second" in tokens
    assert "_sixteenth" not in tokens  # was buggy output


def test_kern_12_is_eighth_triplet(tmp_path: Path) -> None:
    """Kern 12: base = 12 × 2/3 = 8 (eighth note) with tuplet_3."""
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n12c\n*-\n")
    tokens = convert_kern_file(krn)
    assert "<tuplet_3>" in tokens
    assert "_eighth" in tokens


def test_six_kern_48s_emit_tuplet_3(tmp_path: Path) -> None:
    """Six 48-notes in a row are 2 groups of 3 thirty-second triplets (not a sextuplet).
    kern 48 = 32nd-note triplet (3:2 ratio). 6 × (1/12 QL) = 1/2 QL total,
    which is half a beat — not one full quarter beat — so they are NOT a sextuplet.
    The converter emits <tuplet_3> for each and the disambiguator leaves them unchanged.
    """
    krn = _write_kern(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n48c\n48d\n48e\n48f\n48g\n48a\n*-\n",
    )
    tokens = convert_kern_file(krn)
    # Six 32nd triplets stay as tuplet_3 — upgrading to tuplet_6 was incorrect because
    # TUPLET_NORMAL[6]=5 (6:5 ratio) yields wrong QL. Round-trip test confirms tuplet_3.
    assert tokens.count("<tuplet_3>") == 6
    assert tokens.count("<tuplet_6>") == 0


def test_three_kern_24s_in_an_eighth_emit_tuplet_3(tmp_path: Path) -> None:
    """Three kern-24 events in a row = triplet (3 sixteenth-triplets in an eighth beat)."""
    krn = _write_kern(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n24c\n24d\n24e\n*-\n",
    )
    tokens = convert_kern_file(krn)
    assert tokens.count("<tuplet_3>") == 3
    assert tokens.count("<tuplet_6>") == 0


# ---------------------------------------------------------------------------
# v3 enharmonic spelling preservation tests
# ---------------------------------------------------------------------------


def test_cflat_preserved_in_token_output(tmp_path: Path) -> None:
    """kern c- (C-flat) should emit note-Cb4, not note-B3 (enharmonic collapse)."""
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c-\n*-\n")
    tokens = convert_kern_file(krn)
    assert "note-Cb4" in tokens
    assert "note-B3" not in tokens


def test_fflat_preserved_in_token_output(tmp_path: Path) -> None:
    """kern f- (F-flat) should emit note-Fb4, not note-E4 (enharmonic collapse)."""
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4f-\n*-\n")
    tokens = convert_kern_file(krn)
    assert "note-Fb4" in tokens
    assert "note-E4" not in tokens


def test_bsharp_preserved_in_token_output(tmp_path: Path) -> None:
    """kern b# (B-sharp) should emit note-B#4, not note-C5 (enharmonic collapse)."""
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4b#\n*-\n")
    tokens = convert_kern_file(krn)
    assert "note-B#4" in tokens
    assert "note-C5" not in tokens


def test_esharp_preserved_in_token_output(tmp_path: Path) -> None:
    """kern e# (E-sharp) should emit note-E#4, not note-F4 (enharmonic collapse)."""
    krn = _write_kern(tmp_path, "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4e#\n*-\n")
    tokens = convert_kern_file(krn)
    assert "note-E#4" in tokens
    assert "note-F4" not in tokens


# ---------------------------------------------------------------------------
# octave-1 sub-bass clamping bug fix tests
# ---------------------------------------------------------------------------


def test_kern_FFF_emits_F1_not_C2(tmp_path: Path) -> None:
    """kern FFF (3 F's = F1) should emit note-F1, not note-C2 (clamping bug).

    Kern octave encoding for uppercase: octave = 3 - (count - 1), so
    3 uppercase letters → octave 1.  FFFF would be F0 (sub-contrabass).
    """
    krn = _write_kern(tmp_path, "**kern\n*clefF4\n*k[]\n*M4/4\n=1\n4FFF\n*-\n")
    tokens = convert_kern_file(krn)
    assert "note-F1" in tokens
    assert "note-C2" not in tokens


def test_kern_AAA_emits_A1(tmp_path: Path) -> None:
    """kern AAA (3 A's = A1) should emit note-A1."""
    krn = _write_kern(tmp_path, "**kern\n*clefF4\n*k[]\n*M4/4\n=1\n4AAA\n*-\n")
    tokens = convert_kern_file(krn)
    assert "note-A1" in tokens


def test_kern_BBB_minus_emits_Bb1(tmp_path: Path) -> None:
    """kern BBB- (3 B's with flat = Bb1) should emit note-Bb1."""
    krn = _write_kern(tmp_path, "**kern\n*clefF4\n*k[]\n*M4/4\n=1\n4BBB-\n*-\n")
    tokens = convert_kern_file(krn)
    assert "note-Bb1" in tokens
    assert "note-A1" not in tokens  # would be the enharmonic equivalent
