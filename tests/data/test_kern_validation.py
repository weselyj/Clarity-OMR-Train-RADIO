"""Tests for src/data/kern_validation.py."""
from __future__ import annotations

from src.data.kern_validation import CanonicalEvent, Divergence, CompareResult


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
        kern_path=__import__("pathlib").Path("/x.krn"),
        ref_canonical=[],
        our_canonical=[],
        divergences=[],
    )
    assert r.passed is True


def test_compare_result_not_passed_when_divergences_exist() -> None:
    from pathlib import Path
    r = CompareResult(
        kern_path=Path("/x.krn"),
        ref_canonical=[],
        our_canonical=[],
        divergences=[
            Divergence(kind="note", offset_ql=0.0, staff_idx=0, ref_value="C4", our_value="C5", note="")
        ],
    )
    assert r.passed is False


import music21

from src.data.kern_validation import canonicalize_score


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
    assert notes[0].payload == ("C4", 1.0)
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
    assert rests[0].payload == 2.0


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
    pitches, dur = chords[0].payload
    assert pitches == ("C4", "E4", "G4")
    assert dur == 1.0


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
