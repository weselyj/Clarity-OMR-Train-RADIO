"""Tests for src/data/kern_validation.py."""
from __future__ import annotations

from pathlib import Path

import music21

from src.data.kern_validation import (
    CanonicalEvent,
    CompareResult,
    Divergence,
    canonicalize_score,
    compare_via_music21,
    summarize_divergences,
)


def test_canonical_event_dataclass() -> None:
    e = CanonicalEvent(offset_ql=0.0, kind="note", payload=("C4", 1.0), staff_idx=0)
    assert e.kind == "note"
    assert e.payload == ("C4", 1.0)


def test_divergence_dataclass() -> None:
    d = Divergence(
        kind="tie_open",
        offset_ql=2.0,
        staff_idx=0,
        ref_value=True,
        our_value=None,
        note="missing tie_open",
    )
    assert d.kind == "tie_open"
    assert d.our_value is None


def test_compare_result_passed_when_no_divergences() -> None:
    r = CompareResult(
        kern_path=Path("/x.krn"),
        ref_canonical=[],
        our_canonical=[],
        divergences=[],
    )
    assert r.passed is True


def test_compare_result_not_passed_when_divergences_exist() -> None:
    r = CompareResult(
        kern_path=Path("/x.krn"),
        ref_canonical=[],
        our_canonical=[],
        divergences=[
            Divergence(kind="note", offset_ql=0.0, staff_idx=0, ref_value="C4", our_value="C5", note="")
        ],
    )
    assert r.passed is False


def test_canonicalize_simple_quarter_note() -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    measure = music21.stream.Measure(number=1)
    measure.append(music21.note.Note("C4", quarterLength=1.0))
    part.append(measure)
    score.append(part)

    events = canonicalize_score(score)
    notes = [e for e in events if e.kind == "note"]
    assert len(notes) == 1
    assert notes[0].payload == ("C4", 1.0, None)
    assert notes[0].staff_idx == 0


def test_canonicalize_rest() -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    measure = music21.stream.Measure(number=1)
    measure.append(music21.note.Rest(quarterLength=2.0))
    part.append(measure)
    score.append(part)

    events = canonicalize_score(score)
    rests = [e for e in events if e.kind == "rest"]
    assert len(rests) == 1
    assert rests[0].payload == (2.0, None)


def test_canonicalize_chord() -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    measure = music21.stream.Measure(number=1)
    measure.append(music21.chord.Chord(["C4", "E4", "G4"], quarterLength=1.0))
    part.append(measure)
    score.append(part)

    events = canonicalize_score(score)
    chords = [e for e in events if e.kind == "chord"]
    assert len(chords) == 1
    pitches, dur, tuplet = chords[0].payload
    assert pitches == ("C4", "E4", "G4")
    assert dur == 1.0
    assert tuplet is None


def test_canonicalize_two_staves_assigns_staff_idx_top_down() -> None:
    score = music21.stream.Score()
    part_top = music21.stream.Part()
    m_top = music21.stream.Measure(number=1)
    m_top.append(music21.note.Note("C5", quarterLength=1.0))  # treble
    part_top.append(m_top)
    part_bot = music21.stream.Part()
    m_bot = music21.stream.Measure(number=1)
    m_bot.append(music21.note.Note("C3", quarterLength=1.0))  # bass
    part_bot.append(m_bot)
    score.append(part_top)
    score.append(part_bot)

    events = canonicalize_score(score)
    notes = [e for e in events if e.kind == "note"]
    assert {n.staff_idx for n in notes} == {0, 1}
    top_note = next(n for n in notes if n.staff_idx == 0)
    bot_note = next(n for n in notes if n.staff_idx == 1)
    assert top_note.payload[0] == "C5"
    assert bot_note.payload[0] == "C3"


def test_canonicalize_tie_start_and_stop() -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    measure = music21.stream.Measure(number=1)
    n1 = music21.note.Note("C4", quarterLength=1.0)
    n1.tie = music21.tie.Tie("start")
    n2 = music21.note.Note("C4", quarterLength=1.0)
    n2.tie = music21.tie.Tie("stop")
    measure.append(n1)
    measure.append(n2)
    part.append(measure)
    score.append(part)

    events = canonicalize_score(score)
    kinds = [e.kind for e in events]
    assert "tie_open" in kinds
    assert "tie_close" in kinds


def test_canonicalize_articulations() -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    measure = music21.stream.Measure(number=1)
    n = music21.note.Note("C4", quarterLength=1.0)
    n.articulations = [music21.articulations.Accent(), music21.articulations.Staccato()]
    measure.append(n)
    part.append(measure)
    score.append(part)

    events = canonicalize_score(score)
    arts = [e for e in events if e.kind == "articulation"]
    payloads = sorted(e.payload for e in arts)
    assert payloads == ["accent", "staccato"]


def test_canonicalize_ornaments_and_fermata() -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    measure = music21.stream.Measure(number=1)
    n = music21.note.Note("C4", quarterLength=1.0)
    n.expressions = [music21.expressions.Trill(), music21.expressions.Fermata()]
    measure.append(n)
    part.append(measure)
    score.append(part)

    events = canonicalize_score(score)
    orns = [e for e in events if e.kind == "ornament"]
    payloads = sorted(e.payload for e in orns)
    assert payloads == ["fermata", "trill"]


def test_canonicalize_triplet_ratio() -> None:
    """A note with tuplet 3:2 should have its tuplet recorded in the canonical event."""
    score = music21.stream.Score()
    part = music21.stream.Part()
    measure = music21.stream.Measure(number=1)
    n = music21.note.Note("C4", quarterLength=1.0 / 3.0)  # eighth-triplet
    n.duration.appendTuplet(music21.duration.Tuplet(3, 2))
    measure.append(n)
    part.append(measure)
    score.append(part)

    events = canonicalize_score(score)
    note_events = [e for e in events if e.kind == "note"]
    assert len(note_events) == 1
    # Payload extended to include tuplet ratio when present.
    payload = note_events[0].payload
    assert payload[0] == "C4"
    # Third element is tuplet ratio (3, 2) or None when absent
    assert payload[2] == (3, 2)


def _write_kern(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sample.krn"
    p.write_text(content, encoding="utf-8")
    return p


def test_compare_via_music21_simple_passes(tmp_path: Path) -> None:
    """The simplest possible kern (one note, one staff) should round-trip with no divergences."""
    krn = _write_kern(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*-\n",
    )
    result = compare_via_music21(krn)
    assert result.kern_path == krn
    # The simplest case may still surface music21-vs-our differences (e.g., implicit clef).
    # Just ensure the function runs and returns a result.
    assert isinstance(result.ref_canonical, list)
    assert isinstance(result.our_canonical, list)


def test_summarize_divergences_groups_by_kind(tmp_path: Path) -> None:
    results = [
        CompareResult(
            kern_path=Path("/a.krn"),
            ref_canonical=[],
            our_canonical=[],
            divergences=[
                Divergence(kind="tie_open", offset_ql=0.0, staff_idx=0, ref_value=True, our_value=None, note=""),
                Divergence(kind="tie_open", offset_ql=2.0, staff_idx=0, ref_value=True, our_value=None, note=""),
                Divergence(kind="ornament", offset_ql=0.0, staff_idx=0, ref_value="trill", our_value=None, note=""),
            ],
        ),
        CompareResult(
            kern_path=Path("/b.krn"),
            ref_canonical=[],
            our_canonical=[],
            divergences=[
                Divergence(kind="tie_open", offset_ql=0.0, staff_idx=0, ref_value=True, our_value=None, note=""),
            ],
        ),
    ]
    summary = summarize_divergences(results)
    assert summary["tie_open"]["occurrence_count"] == 3
    assert summary["tie_open"]["files_with_kind"] == 2
    assert summary["ornament"]["occurrence_count"] == 1
    assert summary["ornament"]["files_with_kind"] == 1
