"""Tests for eval.tedn (Tree Edit Distance, normalized).

TDD: these tests were written before the implementation. Run them first to confirm
all RED, then implement eval/tedn.py to make them GREEN.
"""
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper: import under test (deferred so test collection always works)
# ---------------------------------------------------------------------------
def _tedn():
    from eval.tedn import compute_tedn, score_to_tree
    return compute_tedn, score_to_tree


# ---------------------------------------------------------------------------
# 1. Identical scores → tedn == 0.0
# ---------------------------------------------------------------------------
class TestIdenticalScore:
    def test_identical_paths_zero(self):
        """Same file as both reference and hypothesis → TEDn = 0.0."""
        compute_tedn, _ = _tedn()
        f = FIXTURES / "simple_cmajor.musicxml"
        result = compute_tedn(f, f)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_identical_two_measure_zero(self):
        """Two-measure score vs itself → TEDn = 0.0."""
        compute_tedn, _ = _tedn()
        f = FIXTURES / "two_measure_cmajor.musicxml"
        result = compute_tedn(f, f)
        assert result == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Single-note pitch difference → small tedn proportional to 1/tree_size
# ---------------------------------------------------------------------------
class TestSinglePitchDiff:
    def test_one_note_changed_gives_small_tedn(self):
        """C4 single-note vs D4 single-note: one leaf label differs → tedn = 1/N_ref."""
        compute_tedn, score_to_tree = _tedn()
        ref = FIXTURES / "single_note_c4.musicxml"
        hyp = FIXTURES / "single_note_d4.musicxml"
        result = compute_tedn(ref, hyp)
        # One node label differs out of the whole tree — must be strictly between 0 and 1
        assert 0.0 < result < 1.0

    def test_one_note_changed_less_than_inserted_measure(self):
        """A single-pitch-change TEDn must be smaller than inserting an entire measure."""
        compute_tedn, _ = _tedn()
        ref = FIXTURES / "simple_cmajor.musicxml"          # 1 measure, 4 notes
        hyp_note_diff = FIXTURES / "single_note_c4.musicxml"
        hyp_measure_insert = FIXTURES / "two_measure_cmajor.musicxml"
        tedn_note = compute_tedn(ref, hyp_note_diff)
        tedn_measure = compute_tedn(ref, hyp_measure_insert)
        assert tedn_note < tedn_measure


# ---------------------------------------------------------------------------
# 3. Inserted measure → elevated tedn
# ---------------------------------------------------------------------------
class TestInsertedMeasure:
    def test_inserted_measure_nonzero(self):
        """Hypothesis with an extra measure → TEDn > 0."""
        compute_tedn, _ = _tedn()
        ref = FIXTURES / "simple_cmajor.musicxml"
        hyp = FIXTURES / "two_measure_cmajor.musicxml"
        result = compute_tedn(ref, hyp)
        assert result > 0.0

    def test_inserted_measure_larger_than_per_note_edit(self):
        """Inserting a full measure costs more than changing one note label."""
        compute_tedn, _ = _tedn()
        ref = FIXTURES / "simple_cmajor.musicxml"
        # single pitch diff — one note label changed
        hyp_note = FIXTURES / "simple_cmajor_missing_note.musicxml"
        hyp_measure = FIXTURES / "two_measure_cmajor.musicxml"
        tedn_note = compute_tedn(ref, hyp_note)
        tedn_measure = compute_tedn(ref, hyp_measure)
        assert tedn_measure > tedn_note


# ---------------------------------------------------------------------------
# 4. Deleted measure → elevated tedn
# ---------------------------------------------------------------------------
class TestDeletedMeasure:
    def test_deleted_measure_nonzero(self):
        """Hypothesis missing a measure → TEDn > 0."""
        compute_tedn, _ = _tedn()
        # reference has 2 measures; hypothesis has 1
        ref = FIXTURES / "two_measure_cmajor.musicxml"
        hyp = FIXTURES / "simple_cmajor.musicxml"
        result = compute_tedn(ref, hyp)
        assert result > 0.0

    def test_deleted_measure_larger_than_single_pitch_edit(self):
        """Deleting a full measure costs more than changing one note within a same-size ref.

        We compare TEDn of (two_measure_ref vs single_note_c4) — a radical mismatch
        due to structural deletion — against TEDn of (single_note_c4 vs single_note_d4)
        which is just one leaf label changed.
        """
        compute_tedn, _ = _tedn()
        # Structural mismatch: 2-measure ref vs 1-note hyp
        ref_two = FIXTURES / "two_measure_cmajor.musicxml"
        hyp_tiny = FIXTURES / "single_note_c4.musicxml"
        tedn_deleted = compute_tedn(ref_two, hyp_tiny)
        # Minimal mismatch: same-size ref vs same-size hyp, one leaf label differs
        ref_one = FIXTURES / "single_note_c4.musicxml"
        hyp_one = FIXTURES / "single_note_d4.musicxml"
        tedn_one_note = compute_tedn(ref_one, hyp_one)
        assert tedn_deleted > tedn_one_note


# ---------------------------------------------------------------------------
# 5. Multi-voice / chord — tree is deterministic and tedn behaves under perturbation
# ---------------------------------------------------------------------------
class TestMultiVoiceChord:
    def test_multivoice_identity(self):
        """Multi-voice chord fixture scored against itself → 0.0."""
        compute_tedn, _ = _tedn()
        f = FIXTURES / "multi_voice_chord.musicxml"
        result = compute_tedn(f, f)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_score_to_tree_deterministic(self):
        """score_to_tree called twice on the same file produces equal label sequences."""
        import music21
        _, score_to_tree = _tedn()
        f = FIXTURES / "multi_voice_chord.musicxml"
        s = music21.corpus.parse(str(f)) if False else music21.converter.parse(str(f))
        t1 = score_to_tree(s)
        t2 = score_to_tree(s)

        def collect_labels(node):
            from eval.tedn import Node
            labels = [node.label]
            for child in node.children:
                labels.extend(collect_labels(child))
            return labels

        assert collect_labels(t1) == collect_labels(t2)

    def test_multivoice_perturbation_nonzero(self):
        """Multi-voice score vs single-voice score → TEDn > 0."""
        compute_tedn, _ = _tedn()
        ref = FIXTURES / "multi_voice_chord.musicxml"
        hyp = FIXTURES / "simple_cmajor.musicxml"
        result = compute_tedn(ref, hyp)
        assert result > 0.0


# ---------------------------------------------------------------------------
# 6. score_to_tree public API — basic node structure
# ---------------------------------------------------------------------------
class TestScoreToTree:
    def test_root_label_is_score(self):
        """Root node label should be 'score'."""
        import music21
        _, score_to_tree = _tedn()
        f = FIXTURES / "simple_cmajor.musicxml"
        s = music21.converter.parse(str(f))
        tree = score_to_tree(s)
        assert tree.label == "score"

    def test_children_are_parts(self):
        """Root node children should represent parts."""
        import music21
        _, score_to_tree = _tedn()
        f = FIXTURES / "simple_cmajor.musicxml"
        s = music21.converter.parse(str(f))
        tree = score_to_tree(s)
        assert len(tree.children) >= 1
        assert tree.children[0].label.startswith("part:")

    def test_note_labels_encode_pitch(self):
        """Note leaf labels should contain step and octave."""
        import music21
        _, score_to_tree = _tedn()
        f = FIXTURES / "single_note_c4.musicxml"
        s = music21.converter.parse(str(f))
        tree = score_to_tree(s)

        def find_note_labels(node):
            from eval.tedn import Node
            result = []
            if node.label.startswith("note:"):
                result.append(node.label)
            for child in node.children:
                result.extend(find_note_labels(child))
            return result

        note_labels = find_note_labels(tree)
        assert len(note_labels) >= 1
        assert any("C" in lbl and "4" in lbl for lbl in note_labels)
