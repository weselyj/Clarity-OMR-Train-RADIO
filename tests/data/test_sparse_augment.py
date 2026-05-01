"""Unit tests for src/data/sparse_augment.py — MusicXML transformations."""
from pathlib import Path

import music21
import pytest


@pytest.fixture
def two_part_score(tmp_path: Path) -> Path:
    """Minimal 2-part score: vocal (all rests) + piano (notes throughout)."""
    s = music21.stream.Score()
    vocal = music21.stream.Part()
    vocal.partName = "Voice"
    piano = music21.stream.Part()
    piano.partName = "Piano"
    for _ in range(8):
        m_v = music21.stream.Measure()
        m_v.append(music21.note.Rest(quarterLength=4))
        vocal.append(m_v)
        m_p = music21.stream.Measure()
        m_p.append(music21.note.Note("C4", quarterLength=4))
        piano.append(m_p)
    s.append(vocal)
    s.append(piano)
    p = tmp_path / "score.musicxml"
    s.write("musicxml", fp=str(p))
    return p


@pytest.fixture
def vocal_with_notes_score(tmp_path: Path) -> Path:
    """Vocal part has multiple notes; piano has notes throughout."""
    s = music21.stream.Score()
    vocal = music21.stream.Part()
    vocal.partName = "Voice"
    piano = music21.stream.Part()
    piano.partName = "Piano"
    for i in range(8):
        m_v = music21.stream.Measure()
        m_v.append(music21.note.Note("E4", quarterLength=4))
        vocal.append(m_v)
        m_p = music21.stream.Measure()
        m_p.append(music21.note.Note("C4", quarterLength=4))
        piano.append(m_p)
    s.append(vocal)
    s.append(piano)
    p = tmp_path / "vocal_score.musicxml"
    s.write("musicxml", fp=str(p))
    return p


def test_strip_silent_intro_drops_fully_silent_part(two_part_score: Path, tmp_path: Path):
    """If a part has only rests for the entire piece, it should be removed
    in v1 (per the plan's documented limitation).
    """
    from src.data.sparse_augment import strip_silent_intro

    out = tmp_path / "stripped.musicxml"
    strip_silent_intro(two_part_score, out, intro_measures=4)

    s = music21.converter.parse(str(out))
    # Voice part is all rests (entire piece) → should be removed
    assert all(p.partName != "Voice" for p in s.parts), \
        f"Expected Voice part to be removed; got parts: {[p.partName for p in s.parts]}"
    # Piano survives
    assert any(p.partName == "Piano" for p in s.parts)


def test_strip_silent_intro_preserves_part_with_any_notes(vocal_with_notes_score: Path, tmp_path: Path):
    """If the part has notes anywhere in the piece, it should NOT be removed."""
    from src.data.sparse_augment import strip_silent_intro

    out = tmp_path / "kept.musicxml"
    strip_silent_intro(vocal_with_notes_score, out, intro_measures=4)

    s = music21.converter.parse(str(out))
    assert any(p.partName == "Voice" for p in s.parts)
    assert any(p.partName == "Piano" for p in s.parts)


def test_add_multibar_rest_replaces_first_n_measures(vocal_with_notes_score: Path, tmp_path: Path):
    """add_multibar_rest replaces measures 1..n in the named part with whole rests."""
    from src.data.sparse_augment import add_multibar_rest

    out = tmp_path / "rest.musicxml"
    add_multibar_rest(out_path=out, source=vocal_with_notes_score, part_name="Voice", n_measures=4)

    s = music21.converter.parse(str(out))
    voice = next(p for p in s.parts if p.partName == "Voice")
    measures = list(voice.getElementsByClass("Measure"))
    # First 4 measures contain only rests
    for i in range(4):
        notes = list(measures[i].notes)
        rests = list(measures[i].notesAndRests.stream().getElementsByClass("Rest"))
        assert len(notes) == 0, f"Measure {i+1} should have no notes; got {len(notes)}"
        assert len(rests) >= 1, f"Measure {i+1} should have at least one rest"
    # Measure 5+ retains the original note(s)
    notes_in_m5 = list(measures[4].notes)
    assert len(notes_in_m5) >= 1


def test_thin_vocal_part_keeps_only_n_notes(vocal_with_notes_score: Path, tmp_path: Path):
    """thin_vocal_part replaces all notes EXCEPT the first N with rests."""
    from src.data.sparse_augment import thin_vocal_part

    out = tmp_path / "thin.musicxml"
    thin_vocal_part(vocal_with_notes_score, out, part_name="Voice", keep_n_notes=1)

    s = music21.converter.parse(str(out))
    voice = next(p for p in s.parts if p.partName == "Voice")
    notes = list(voice.recurse().notes)
    assert len(notes) == 1, f"Expected exactly 1 note in Voice; got {len(notes)}"


def test_part_name_not_found_raises(two_part_score: Path, tmp_path: Path):
    """Passing a non-existent part name should produce a clear error, not silent no-op."""
    from src.data.sparse_augment import add_multibar_rest

    out = tmp_path / "nope.musicxml"
    with pytest.raises(ValueError, match="not in score"):
        add_multibar_rest(out_path=out, source=two_part_score, part_name="Trumpet", n_measures=2)
