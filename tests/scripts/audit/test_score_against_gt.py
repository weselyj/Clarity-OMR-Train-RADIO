"""Tests for the Bethlehem-vs-GT scoring harness.

Builds tiny synthetic MusicXML via music21, writes to tmp_path, scores.
CPU-only, deterministic.
"""
import music21
import pytest

from scripts.audit.score_against_gt import score


def _two_part_score(tmp_path, name, n_measures, bass_clef_sign="F", bass_clef_line=4):
    """A 2-part score: part0 treble G2 (one C5/measure), part1 bass (one C3/measure)."""
    sc = music21.stream.Score()
    p0 = music21.stream.Part(); p0.id = "P0"
    p1 = music21.stream.Part(); p1.id = "P1"
    p0.append(music21.clef.TrebleClef())
    bc = music21.clef.Clef(); bc.sign = bass_clef_sign; bc.line = bass_clef_line
    p1.append(bc)
    for m in range(n_measures):
        m0 = music21.stream.Measure(number=m + 1)
        m0.append(music21.note.Note("C5", quarterLength=4))
        p0.append(m0)
        m1 = music21.stream.Measure(number=m + 1)
        m1.append(music21.note.Note("C3", quarterLength=4))
        p1.append(m1)
    sc.append(p0); sc.append(p1)
    path = tmp_path / f"{name}.musicxml"
    sc.write("musicxml", fp=str(path))
    return str(path)


def test_perfect_match_scores_one(tmp_path):
    gt = _two_part_score(tmp_path, "gt", 4)
    pred = _two_part_score(tmp_path, "pred", 4)
    r = score(pred, gt)
    assert r["measure_recall"] == pytest.approx(1.0)
    assert r["clef_accuracy"] == pytest.approx(1.0)
    assert r["note_onset_f1"] == pytest.approx(1.0)


def test_missing_measures_lowers_recall(tmp_path):
    gt = _two_part_score(tmp_path, "gt", 4)
    pred = _two_part_score(tmp_path, "pred", 2)  # 2 of 4 measures
    r = score(pred, gt)
    assert r["measure_recall"] == pytest.approx(0.5, abs=0.01)


def test_wrong_bass_clef_lowers_clef_accuracy(tmp_path):
    gt = _two_part_score(tmp_path, "gt", 2)
    # pred bass staff mislabeled as treble G2 (the Bethlehem failure)
    pred = _two_part_score(tmp_path, "pred", 2, bass_clef_sign="G", bass_clef_line=2)
    r = score(pred, gt)
    assert r["clef_accuracy"] == pytest.approx(0.5, abs=0.01)  # 1 of 2 parts correct


def test_mixed_clefs_not_masked_by_majority(tmp_path):
    """A bass part with clefs [F4, F4, G2] must score per-occurrence (2/3 on
    that part), NOT 1.0 via majority voting. Guards the masking bug the
    Bethlehem smoke exposed."""
    sc = music21.stream.Score()
    p0 = music21.stream.Part(); p0.id = "P0"
    p1 = music21.stream.Part(); p1.id = "P1"
    p0.append(music21.clef.TrebleClef())              # part0: one G2 (correct)
    bass_signs = [("F", 4), ("F", 4), ("G", 2)]       # part1: F4, F4, G2 (one flip)
    for idx, (sign, line) in enumerate(bass_signs):
        m = music21.stream.Measure(number=idx + 1)
        c = music21.clef.Clef(); c.sign = sign; c.line = line
        m.append(c)
        m.append(music21.note.Note("C3", quarterLength=4))
        p1.append(m)
    for idx in range(3):
        m = music21.stream.Measure(number=idx + 1)
        m.append(music21.note.Note("C5", quarterLength=4))
        p0.append(m)
    sc.append(p0); sc.append(p1)
    gt = _two_part_score(tmp_path, "gt", 3)
    pred_path = str(tmp_path / "pred_mixed.musicxml")
    sc.write("musicxml", fp=pred_path)

    r = score(pred_path, gt)
    # part0: 1/1 G2 correct; part1: 2/3 F4 correct -> 3/4 = 0.75, and NOT 1.0
    assert r["clef_accuracy"] == pytest.approx(0.75, abs=0.02)
    assert r["clef_accuracy"] < 1.0
