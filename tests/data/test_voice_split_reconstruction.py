"""Regression tests for mid-measure *^ (voice-split) reconstruction in export_musicxml.py.

When a kern `*^` fires mid-measure (after some notes have already been emitted on
voice 1), `append_tokens_to_part` must place voice-2's first event at the elapsed
offset within the measure — not at measure offset 0.

Bug: line 506 of export_musicxml.py called voices.setdefault(active_voice, ...) without
     any padding, so new voices always started at offset 0 regardless of elapsed time.
Fix: insert a hidden rest of duration == elapsed offset into voice N before its first note.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.kern_validation import compare_via_music21


def _write_kern(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sample.krn"
    p.write_text(content, encoding="utf-8")
    return p


def test_mid_measure_voice_split_reconstructs_at_correct_offset(tmp_path: Path) -> None:
    """When *^ fires mid-measure, voice 2 must start at the elapsed offset, not 0.

    Kern structure (treble staff only, 3/4):
      measure 1:
        voice 1: quarter C4  (offset 0)
        *^ fires here — voice 1 is at 1.0 QL elapsed
        voice 1: quarter D4  (offset 1.0)
        voice 2: quarter E4  (offset 1.0 — must NOT land at 0.0)
        voice 1: quarter C4  (offset 2.0)
        voice 2: quarter F4  (offset 2.0)
        *v *v (merge)

    Expected: voice-2's E4 lands at score offset 1.0, voice-2's F4 at score offset 2.0.
    """
    krn = _write_kern(
        tmp_path,
        "**kern\n"
        "*clefG2\n"
        "*k[]\n"
        "*M3/4\n"
        "=1\n"
        "4c\n"          # voice 1: C4 quarter at offset 0
        "*^\n"          # split fires HERE — voice 1 elapsed = 1.0 QL
        "4d\t4e\n"      # voice 1: D4 quarter | voice 2: E4 quarter
        "4c\t4f\n"      # voice 1: C4 quarter | voice 2: F4 quarter
        "*v\t*v\n"      # merge
        "=2\n"
        "*-\n",
    )
    result = compare_via_music21(krn)
    note_rest_divs = [d for d in result.divergences if d.kind in ("note", "rest")]
    assert len(note_rest_divs) == 0, (
        f"voice-split divergences after mid-measure *^:\n"
        + "\n".join(
            f"  staff={d.staff_idx} offset={d.offset_ql} kind={d.kind} "
            f"ref={d.ref_value!r} our={d.our_value!r}"
            for d in note_rest_divs
        )
    )


def test_voice_split_at_measure_start_unaffected(tmp_path: Path) -> None:
    """When *^ fires at the very start of a measure (elapsed=0), voice 2 still starts at 0.

    No hidden rest should be inserted (it would be zero-length anyway), and the
    fix must not introduce a spurious rest at offset 0.
    """
    krn = _write_kern(
        tmp_path,
        "**kern\n"
        "*clefG2\n"
        "*k[]\n"
        "*M4/4\n"
        "=1\n"
        "*^\n"          # split at measure start — elapsed = 0
        "4c\t4e\n"
        "4d\t4f\n"
        "4c\t4e\n"
        "4d\t4f\n"
        "*v\t*v\n"
        "=2\n"
        "*-\n",
    )
    result = compare_via_music21(krn)
    note_rest_divs = [d for d in result.divergences if d.kind in ("note", "rest")]
    assert len(note_rest_divs) == 0, (
        f"unexpected divergences when *^ fires at measure start:\n"
        + "\n".join(
            f"  staff={d.staff_idx} offset={d.offset_ql} kind={d.kind} "
            f"ref={d.ref_value!r} our={d.our_value!r}"
            for d in note_rest_divs
        )
    )


def test_header_level_voice_split_reconstructs_correctly(tmp_path: Path) -> None:
    """Header-level *^ (immediately after spine declarations) must not drop staff-1 content.

    When *^ appears in the header before any body events, both sub-spines should
    reconstruct without missing notes.  This covers the bug where an
    AccidentalException in _parse_pitch_token (double-flat bb notation) caused
    append_tokens_to_part to silently bail, producing an empty Part for staff 1.

    Kern structure (grand staff, 3/4): spine 1 splits *^ at the header level.
      Bass staff voices: voice1=quarter chord + half, voice2=half + quarter
    """
    krn = _write_kern(
        tmp_path,
        "**kern\t**kern\n"
        "*^\t*\n"                          # header-level split on bass spine
        "*clefF4\t*clefF4\t*clefG2\n"
        "*k[]\t*k[]\t*k[]\n"
        "*M3/4\t*M3/4\t*M3/4\n"
        "=1\t=1\t=1\n"
        "4C 4E\t2G\t4c\n"                  # bass voice1: chord C3/E3 q | bass voice2: G2 half | treble: C4 q
        "4D\t.\t4d\n"                       # bass voice1: D3 q       |             (continues) | treble: D4 q
        "4E\t4F\t4e\n"                      # bass voice1: E3 q       | bass voice2: F2 q       | treble: E4 q
        "*v\t*v\t*\n"                       # merge bass voices
        "=2\t=2\n"
        "*-\t*-\n",
    )
    result = compare_via_music21(krn)
    note_rest_divs = [d for d in result.divergences if d.kind in ("note", "rest", "chord")]
    assert len(note_rest_divs) == 0, (
        f"divergences with header-level *^:\n"
        + "\n".join(
            f"  staff={d.staff_idx} offset={d.offset_ql} kind={d.kind} "
            f"ref={d.ref_value!r} our={d.our_value!r}"
            for d in note_rest_divs
        )
    )


def test_double_flat_pitch_roundtrip(tmp_path: Path) -> None:
    """Double-flat pitch tokens (e.g. Bbb2) must survive the token→music21 round-trip.

    The RADIO vocab encodes double-flats as ``bb`` (e.g. ``Bbb2``) but music21
    requires ``--`` (e.g. ``B--2``).  The _parse_pitch_token helper must translate
    between the two formats.  A note encoded as ``Bbb2`` in kern (B double-flat)
    should round-trip to the same canonical event.
    """
    krn = _write_kern(
        tmp_path,
        "**kern\n"
        "*clefF4\n"
        "*k[b-e-a-d-g-]\n"
        "*M4/4\n"
        "=1\n"
        "4BB--\n"    # B double-flat 2 (kern: BB--)
        "4BB-\n"
        "4BB\n"
        "4BB-\n"
        "=2\n"
        "*-\n",
    )
    result = compare_via_music21(krn)
    note_rest_divs = [d for d in result.divergences if d.kind in ("note", "rest")]
    assert len(note_rest_divs) == 0, (
        f"double-flat pitch divergences:\n"
        + "\n".join(
            f"  staff={d.staff_idx} offset={d.offset_ql} kind={d.kind} "
            f"ref={d.ref_value!r} our={d.our_value!r}"
            for d in note_rest_divs
        )
    )


def test_sample_div_krn_divergences_drop(tmp_path: Path) -> None:
    """The concrete beethoven sample file that triggered this bug must have 0 note/rest divergences."""
    sample = Path("/tmp/sample_div.krn")
    if not sample.exists():
        pytest.skip("sample_div.krn not available — run the scp step first")
    result = compare_via_music21(sample)
    note_rest_divs = [d for d in result.divergences if d.kind in ("note", "rest")]
    assert len(note_rest_divs) == 0, (
        f"note/rest divergences remain on sample_div.krn:\n"
        + "\n".join(
            f"  staff={d.staff_idx} offset={d.offset_ql} kind={d.kind} "
            f"ref={d.ref_value!r} our={d.our_value!r}"
            for d in note_rest_divs
        )
    )
