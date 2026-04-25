from pathlib import Path
import pytest
from eval.playback import playback_f

FIXTURES = Path(__file__).parent / "fixtures"


def test_identity_playback_f_is_one():
    """Scoring a file against itself yields F ~= 1.0."""
    f = FIXTURES / "simple_cmajor.musicxml"
    result = playback_f(pred=f, gt=f)
    assert result["f"] == pytest.approx(1.0, abs=0.01)


def test_missing_note_drops_f():
    """Predicted missing one of 4 ground-truth notes -> F = 2*(3/3)*(3/4) / ((3/3)+(3/4)) ~= 0.857."""
    gt = FIXTURES / "simple_cmajor.musicxml"
    pred = FIXTURES / "simple_cmajor_missing_note.musicxml"
    r = playback_f(pred=pred, gt=gt)
    assert r["f"] == pytest.approx(0.857, abs=0.01)


def test_enharmonic_predicted_matches_ground_truth():
    """Enharmonic rewrite (Fb vs E) does not affect playback metric (same MIDI pitch)."""
    gt = FIXTURES / "simple_cmajor.musicxml"
    pred = FIXTURES / "simple_cmajor_enharmonic.musicxml"
    r = playback_f(pred=pred, gt=gt)
    assert r["f"] == pytest.approx(1.0, abs=0.01)


def test_different_divisions_scores_identical():
    """Quarter notes at divisions=4 vs divisions=480 must score the same vs a shared GT."""
    gt = FIXTURES / "simple_cmajor.musicxml"
    a = playback_f(pred=FIXTURES / "simple_cmajor.musicxml", gt=gt)
    b = playback_f(pred=FIXTURES / "simple_cmajor_different_divisions.musicxml", gt=gt)
    assert a["f"] == pytest.approx(b["f"], abs=0.001)


def test_multi_measure_onsets_correct():
    """Multi-measure prediction against single-measure GT: extra-note fixture has 5 notes
    (original 4 in measure 1 + G4 in measure 2 at absolute 4.0 quarters = 2.0 seconds at 120 BPM).

    Against GT's 4 notes: 4 TP, 1 FP, 0 FN -> P = 4/5 = 0.8, R = 1.0, F = 2*0.8*1.0/(0.8+1.0) ~= 0.889.
    """
    gt = FIXTURES / "simple_cmajor.musicxml"
    pred = FIXTURES / "simple_cmajor_extra_note.musicxml"
    r = playback_f(pred=pred, gt=gt)
    assert r["precision"] == pytest.approx(0.8, abs=0.01)
    assert r["recall"] == pytest.approx(1.0, abs=0.01)
    assert r["f"] == pytest.approx(0.889, abs=0.01)
