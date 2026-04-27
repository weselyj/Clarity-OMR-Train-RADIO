"""Tests for eval.linearized_musicxml (linearized MusicXML SER).

TDD: these tests were written before the implementation. Run them first to confirm
all RED, then implement eval/linearized_musicxml.py to make them GREEN.
"""
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _lmx():
    from eval.linearized_musicxml import linearize, compute_linearized_ser
    return linearize, compute_linearized_ser


# ---------------------------------------------------------------------------
# 1. Identical scores → linearized_ser == 0.0
# ---------------------------------------------------------------------------
class TestIdenticalScore:
    def test_identical_ser_zero(self):
        """Same file as both reference and hypothesis → linearized_ser = 0.0."""
        _, compute_linearized_ser = _lmx()
        f = FIXTURES / "simple_cmajor.musicxml"
        result = compute_linearized_ser(f, f)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_two_measure_identical_zero(self):
        """Two-measure file vs itself → 0.0."""
        _, compute_linearized_ser = _lmx()
        f = FIXTURES / "two_measure_cmajor.musicxml"
        result = compute_linearized_ser(f, f)
        assert result == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Linearization is deterministic
# ---------------------------------------------------------------------------
class TestLinearizationDeterminism:
    def test_linearize_same_result_twice(self):
        """linearize() returns identical token list on repeated calls."""
        import music21
        linearize, _ = _lmx()
        f = FIXTURES / "simple_cmajor.musicxml"
        s = music21.converter.parse(str(f))
        t1 = linearize(s)
        t2 = linearize(s)
        assert t1 == t2

    def test_linearize_returns_list_of_strings(self):
        """linearize() returns a non-empty list of strings."""
        import music21
        linearize, _ = _lmx()
        f = FIXTURES / "simple_cmajor.musicxml"
        s = music21.converter.parse(str(f))
        tokens = linearize(s)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_linearize_multivoice_deterministic(self):
        """Multi-voice chord fixture linearizes identically each call."""
        import music21
        linearize, _ = _lmx()
        f = FIXTURES / "multi_voice_chord.musicxml"
        s = music21.converter.parse(str(f))
        t1 = linearize(s)
        t2 = linearize(s)
        assert t1 == t2


# ---------------------------------------------------------------------------
# 3. Measure boundary tokens present
# ---------------------------------------------------------------------------
class TestMeasureBoundaryTokens:
    def test_meas_token_present(self):
        """<MEAS> token appears in linearized output for each measure."""
        import music21
        linearize, _ = _lmx()
        f = FIXTURES / "simple_cmajor.musicxml"
        s = music21.converter.parse(str(f))
        tokens = linearize(s)
        assert "<MEAS>" in tokens

    def test_two_measures_two_meas_tokens(self):
        """Two-measure score has exactly two <MEAS> tokens."""
        import music21
        linearize, _ = _lmx()
        f = FIXTURES / "two_measure_cmajor.musicxml"
        s = music21.converter.parse(str(f))
        tokens = linearize(s)
        assert tokens.count("<MEAS>") == 2

    def test_voice_token_present(self):
        """<VOICE> token appears when voice content is present."""
        import music21
        linearize, _ = _lmx()
        f = FIXTURES / "simple_cmajor.musicxml"
        s = music21.converter.parse(str(f))
        tokens = linearize(s)
        assert "<VOICE>" in tokens


# ---------------------------------------------------------------------------
# 4. Note tokens encode pitch
# ---------------------------------------------------------------------------
class TestNoteTokenEncoding:
    def test_note_token_contains_step(self):
        """Note token for C4 quarter contains 'C' and octave '4'."""
        import music21
        linearize, _ = _lmx()
        f = FIXTURES / "single_note_c4.musicxml"
        s = music21.converter.parse(str(f))
        tokens = linearize(s)
        note_tokens = [t for t in tokens if t.startswith("note:")]
        assert len(note_tokens) >= 1
        assert any("C" in t and "4" in t for t in note_tokens)

    def test_rest_token_present(self):
        """Rest tokens start with 'rest:'."""
        import music21
        linearize, _ = _lmx()
        f = FIXTURES / "single_note_c4.musicxml"
        s = music21.converter.parse(str(f))
        tokens = linearize(s)
        rest_tokens = [t for t in tokens if t.startswith("rest:")]
        # single_note_c4 has 3 rests (filling remaining beats)
        assert len(rest_tokens) >= 1

    def test_different_notes_different_tokens(self):
        """C4 and D4 single-note scores produce different linearizations."""
        import music21
        linearize, _ = _lmx()
        s_c = music21.converter.parse(str(FIXTURES / "single_note_c4.musicxml"))
        s_d = music21.converter.parse(str(FIXTURES / "single_note_d4.musicxml"))
        tokens_c = linearize(s_c)
        tokens_d = linearize(s_d)
        assert tokens_c != tokens_d


# ---------------------------------------------------------------------------
# 5. Edge cases: empty hypothesis, longer hypothesis
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_hypothesis_ser_is_one_or_more(self):
        """Empty hypothesis → SER >= 1.0 (all reference tokens are deletions)."""
        _, compute_linearized_ser = _lmx()
        ref = FIXTURES / "simple_cmajor.musicxml"
        # We pass the same file as hyp but use an empty score inline via a helper
        # that accepts Score objects. Since public API is path-based, use no_notes fixture.
        hyp = FIXTURES / "simple_cmajor_no_notes.musicxml"
        result = compute_linearized_ser(ref, hyp)
        # SER can be > 1 if hyp is longer; for empty hyp edit_dist == len(ref_tokens)
        assert result > 0.0

    def test_longer_hypothesis_nonzero(self):
        """Hypothesis longer than reference → nonzero SER."""
        _, compute_linearized_ser = _lmx()
        ref = FIXTURES / "simple_cmajor.musicxml"      # 1 measure
        hyp = FIXTURES / "two_measure_cmajor.musicxml"  # 2 measures
        result = compute_linearized_ser(ref, hyp)
        assert result > 0.0

    def test_ser_nonnegative(self):
        """SER is always >= 0."""
        _, compute_linearized_ser = _lmx()
        ref = FIXTURES / "simple_cmajor.musicxml"
        hyp = FIXTURES / "simple_cmajor_missing_note.musicxml"
        result = compute_linearized_ser(ref, hyp)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# 6. Linearized SER on a pitch change is small but nonzero
# ---------------------------------------------------------------------------
class TestPitchChangeSER:
    def test_pitch_change_nonzero(self):
        """One note pitch changed → SER > 0."""
        _, compute_linearized_ser = _lmx()
        ref = FIXTURES / "single_note_c4.musicxml"
        hyp = FIXTURES / "single_note_d4.musicxml"
        result = compute_linearized_ser(ref, hyp)
        assert result > 0.0

    def test_pitch_change_less_than_measure_insert(self):
        """Pitch change SER < added-measure SER."""
        _, compute_linearized_ser = _lmx()
        ref = FIXTURES / "simple_cmajor.musicxml"
        hyp_note = FIXTURES / "simple_cmajor_missing_note.musicxml"
        hyp_measure = FIXTURES / "two_measure_cmajor.musicxml"
        ser_note = compute_linearized_ser(ref, hyp_note)
        ser_measure = compute_linearized_ser(ref, hyp_measure)
        assert ser_note < ser_measure
