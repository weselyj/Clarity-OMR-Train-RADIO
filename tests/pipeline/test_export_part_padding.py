"""Per-system part padding in assembled_score_to_music21*.

Regression test for the bug where systems with fewer staves than the score has
parts caused MuseScore to reject the produced MusicXML as corrupt: each
`<part>` in score-partwise must share the same measure-index timeline, but the
export loop appended only the staves the system contained, so parts diverged.

These tests build an AssembledScore directly (skipping the YOLO + decoder
pipeline) so the padding behaviour can be exercised deterministically.
"""
from __future__ import annotations


def _tokens_n_measures(n: int) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        out.extend(["<measure_start>", "rest", "_whole", "<measure_end>"])
    return out


def _make_assembled_score():
    """Two-system AssembledScore where the second system is missing the
    right-hand staff — exactly the shape that produced the corrupt files.

    System 1 contributes 2 measures to each of {piano_right_hand, piano_left_hand}.
    System 2 contributes 2 measures to piano_left_hand only — without padding,
    piano_right_hand would end at 2 measures while piano_left_hand reaches 4.
    """
    from src.pipeline.assemble_score import (
        AssembledScore,
        AssembledStaff,
        AssembledSystem,
        StaffLocation,
    )

    def staff(label: str, n_measures: int, sample_id: str) -> AssembledStaff:
        return AssembledStaff(
            sample_id=sample_id,
            tokens=_tokens_n_measures(n_measures),
            part_label=label,
            measure_count=n_measures,
            clef="clef-G2" if label == "piano_right_hand" else "clef-F4",
            key_signature=None,
            time_signature=None,
            location=StaffLocation(0, 0.0, 100.0, 0.0, 1000.0),
        )

    sys1 = AssembledSystem(
        page_index=0,
        system_index=0,
        staves=[
            staff("piano_right_hand", 2, "p0s0r"),
            staff("piano_left_hand", 2, "p0s0l"),
        ],
        canonical_measure_count=2,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    sys2 = AssembledSystem(
        page_index=0,
        system_index=1,
        staves=[
            staff("piano_left_hand", 2, "p0s1l"),
        ],
        canonical_measure_count=2,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    return AssembledScore(
        systems=[sys1, sys2],
        part_order=["piano_right_hand", "piano_left_hand"],
    )


def _count_measures_per_part(music21_score) -> list[int]:
    from music21 import stream

    return [
        len(part.getElementsByClass(stream.Measure))
        for part in music21_score.parts
    ]


def test_pad_in_undiagnosed_export():
    """assembled_score_to_music21 (no diagnostics): missing staves get padded."""
    from src.pipeline.export_musicxml import assembled_score_to_music21

    score = _make_assembled_score()
    m21 = assembled_score_to_music21(score)

    counts = _count_measures_per_part(m21)
    # Both parts must end with the same total measure count.
    assert len(set(counts)) == 1, f"parts diverged: {counts}"
    # Total = sum of canonical_measure_count over systems = 2 + 2 = 4.
    assert counts[0] == 4, f"expected 4 measures per part, got {counts}"


def test_pad_in_diagnosed_export_increments_counter():
    """The diagnostics variant increments `padded_measures` by the number of
    rest-measures it inserts (one part missing 2 measures => +2)."""
    from src.pipeline.export_musicxml import (
        StageDExportDiagnostics,
        assembled_score_to_music21_with_diagnostics,
    )

    score = _make_assembled_score()
    diags = StageDExportDiagnostics()
    m21 = assembled_score_to_music21_with_diagnostics(score, diags)

    counts = _count_measures_per_part(m21)
    assert len(set(counts)) == 1, f"parts diverged: {counts}"
    assert counts[0] == 4, f"expected 4 measures per part, got {counts}"
    assert diags.padded_measures == 2, (
        f"expected padded_measures=2, got {diags.padded_measures}"
    )


def test_no_padding_when_all_systems_have_full_staves():
    """When every system contains both staves, no padding should be inserted
    and the counter should remain at zero."""
    from src.pipeline.assemble_score import (
        AssembledScore,
        AssembledStaff,
        AssembledSystem,
        StaffLocation,
    )
    from src.pipeline.export_musicxml import (
        StageDExportDiagnostics,
        assembled_score_to_music21_with_diagnostics,
    )

    def staff(label: str, n: int, sid: str) -> AssembledStaff:
        return AssembledStaff(
            sample_id=sid,
            tokens=_tokens_n_measures(n),
            part_label=label,
            measure_count=n,
            clef="clef-G2" if label == "piano_right_hand" else "clef-F4",
            key_signature=None,
            time_signature=None,
            location=StaffLocation(0, 0.0, 100.0, 0.0, 1000.0),
        )

    sys = AssembledSystem(
        page_index=0, system_index=0,
        staves=[
            staff("piano_right_hand", 3, "rh"),
            staff("piano_left_hand", 3, "lh"),
        ],
        canonical_measure_count=3,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    score = AssembledScore(
        systems=[sys],
        part_order=["piano_right_hand", "piano_left_hand"],
    )
    diags = StageDExportDiagnostics()
    m21 = assembled_score_to_music21_with_diagnostics(score, diags)

    counts = _count_measures_per_part(m21)
    assert counts == [3, 3], f"expected [3, 3], got {counts}"
    assert diags.padded_measures == 0
