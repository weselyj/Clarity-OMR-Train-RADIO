"""Regression tests: every <part> in the exported MusicXML must carry the
same number of <measure> elements.

Partwise MusicXML defines a shared measure-index timeline across parts. When
the decoder emits a system with fewer staves than the score has parts (e.g.
only the bass staff of a piano grand staff survives Stage A merging), the
"missing" part falls behind on that system's measures. MuseScore 4 rejects
the resulting file as corrupted (music21's parser is permissive and does not).

The fix in export_musicxml.assembled_score_to_music21[_with_diagnostics] pads
any part that didn't receive a staff this system with the system's
canonical_measure_count whole-rest measures.
"""
from __future__ import annotations


def _make_staff(*, sample_id, part_label, tokens, measure_count, page_index=0):
    from src.pipeline.assemble_score import AssembledStaff, StaffLocation

    return AssembledStaff(
        sample_id=sample_id,
        tokens=tokens,
        part_label=part_label,
        measure_count=measure_count,
        clef=None,
        key_signature=None,
        time_signature=None,
        location=StaffLocation(
            page_index=page_index, y_top=0.0, y_bottom=100.0,
            x_left=0.0, x_right=1000.0,
        ),
    )


def _two_measure_staff_tokens():
    """A staff with two minimal measures (quarter-note + whole rest)."""
    return [
        "<measure_start>", "note-C4-quarter", "<measure_end>",
        "<measure_start>", "rest", "_whole", "<measure_end>",
    ]


def _one_measure_staff_tokens(pitch="note-C4-quarter"):
    return ["<measure_start>", pitch, "<measure_end>"]


def test_balanced_grand_staff_produces_matching_part_lengths():
    """Sanity: when every system has both RH+LH, no padding is needed and
    the existing balanced behavior is preserved."""
    from src.pipeline.assemble_score import AssembledScore, AssembledSystem
    from src.pipeline.export_musicxml import assembled_score_to_music21
    from music21 import stream as m21_stream

    system_a = AssembledSystem(
        page_index=0, system_index=0,
        staves=[
            _make_staff(sample_id="s0_rh", part_label="piano_right_hand",
                        tokens=_two_measure_staff_tokens(), measure_count=2),
            _make_staff(sample_id="s0_lh", part_label="piano_left_hand",
                        tokens=_two_measure_staff_tokens(), measure_count=2),
        ],
        canonical_measure_count=2,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    score = AssembledScore(
        systems=[system_a],
        part_order=["piano_right_hand", "piano_left_hand"],
    )

    out = assembled_score_to_music21(score)
    measure_counts = [
        len(list(p.getElementsByClass(m21_stream.Measure)))
        for p in out.parts
    ]
    assert len(measure_counts) == 2
    assert measure_counts[0] == measure_counts[1] == 2


def test_trailing_orphan_staff_pads_the_missing_part():
    """The decoder emits a final system with only one staff (e.g. only the
    bass staff). The other part must be padded so both have the same count."""
    from src.pipeline.assemble_score import AssembledScore, AssembledSystem
    from src.pipeline.export_musicxml import assembled_score_to_music21
    from music21 import stream as m21_stream

    # System 0: both staves, 2 measures each.
    system_a = AssembledSystem(
        page_index=0, system_index=0,
        staves=[
            _make_staff(sample_id="s0_rh", part_label="piano_right_hand",
                        tokens=_two_measure_staff_tokens(), measure_count=2),
            _make_staff(sample_id="s0_lh", part_label="piano_left_hand",
                        tokens=_two_measure_staff_tokens(), measure_count=2),
        ],
        canonical_measure_count=2,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    # System 1: ONLY the left-hand staff (orphan after _merge_undersized_systems
    # failed to find a sibling). Canonical measure count = 1.
    system_b = AssembledSystem(
        page_index=0, system_index=1,
        staves=[
            _make_staff(sample_id="s1_lh", part_label="piano_left_hand",
                        tokens=_one_measure_staff_tokens("note-C3-quarter"),
                        measure_count=1),
        ],
        canonical_measure_count=1,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    score = AssembledScore(
        systems=[system_a, system_b],
        part_order=["piano_right_hand", "piano_left_hand"],
    )

    out = assembled_score_to_music21(score)
    measure_counts = {
        p.id: len(list(p.getElementsByClass(m21_stream.Measure)))
        for p in out.parts
    }
    # Both parts must have measure_count(system_a) + measure_count(system_b)
    # = 2 + 1 = 3 measures, even though system_b only provided a LH staff.
    assert measure_counts["piano_right_hand"] == 3
    assert measure_counts["piano_left_hand"] == 3


def test_leading_orphan_staff_pads_the_missing_part():
    """The first system has only one staff; later systems are complete.
    The padding for the missing part must precede the real content."""
    from src.pipeline.assemble_score import AssembledScore, AssembledSystem
    from src.pipeline.export_musicxml import assembled_score_to_music21
    from music21 import stream as m21_stream

    system_a = AssembledSystem(
        page_index=0, system_index=0,
        staves=[
            _make_staff(sample_id="s0_rh", part_label="piano_right_hand",
                        tokens=_one_measure_staff_tokens(), measure_count=1),
        ],
        canonical_measure_count=1,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    system_b = AssembledSystem(
        page_index=0, system_index=1,
        staves=[
            _make_staff(sample_id="s1_rh", part_label="piano_right_hand",
                        tokens=_two_measure_staff_tokens(), measure_count=2),
            _make_staff(sample_id="s1_lh", part_label="piano_left_hand",
                        tokens=_two_measure_staff_tokens(), measure_count=2),
        ],
        canonical_measure_count=2,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    score = AssembledScore(
        systems=[system_a, system_b],
        part_order=["piano_right_hand", "piano_left_hand"],
    )

    out = assembled_score_to_music21(score)
    measure_counts = {
        p.id: len(list(p.getElementsByClass(m21_stream.Measure)))
        for p in out.parts
    }
    assert measure_counts["piano_right_hand"] == 3
    assert measure_counts["piano_left_hand"] == 3


def test_with_diagnostics_variant_also_pads():
    """The diagnostics-collecting export path must apply the same padding."""
    from src.pipeline.assemble_score import AssembledScore, AssembledSystem
    from src.pipeline.export_musicxml import (
        StageDExportDiagnostics,
        assembled_score_to_music21_with_diagnostics,
    )
    from music21 import stream as m21_stream

    system_a = AssembledSystem(
        page_index=0, system_index=0,
        staves=[
            _make_staff(sample_id="s0_rh", part_label="piano_right_hand",
                        tokens=_two_measure_staff_tokens(), measure_count=2),
            _make_staff(sample_id="s0_lh", part_label="piano_left_hand",
                        tokens=_two_measure_staff_tokens(), measure_count=2),
        ],
        canonical_measure_count=2,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    system_b = AssembledSystem(
        page_index=0, system_index=1,
        staves=[
            _make_staff(sample_id="s1_lh", part_label="piano_left_hand",
                        tokens=_one_measure_staff_tokens("note-C3-quarter"),
                        measure_count=1),
        ],
        canonical_measure_count=1,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    score = AssembledScore(
        systems=[system_a, system_b],
        part_order=["piano_right_hand", "piano_left_hand"],
    )

    diags = StageDExportDiagnostics()
    out = assembled_score_to_music21_with_diagnostics(score, diags)
    measure_counts = {
        p.id: len(list(p.getElementsByClass(m21_stream.Measure)))
        for p in out.parts
    }
    assert measure_counts["piano_right_hand"] == 3
    assert measure_counts["piano_left_hand"] == 3
    # The orphan LH-only system caused 1 measure of padding on piano_right_hand.
    assert diags.padded_measures == 1
    # Padding uses well-formed measure tokens, so no silent-skip counters fire.
    assert diags.unknown_tokens == 0
    assert diags.malformed_spans == 0
