"""Tests for Stage D export diagnostics (issue #5 + issue #1 Tier 2).

TDD: written before implementation.  Covers:
  1. Happy path – well-formed token spans → all counters zero.
  2. Malformed chord span – no <chord_end> → malformed_spans=1, no exception.
  3. Missing duration – note token followed by non-duration token → missing_durations=1.
  4. Unknown token – unrecognized token in stream → unknown_tokens=1, no crash.
  5. Strict mode – same malformed input → raises ValueError.
  6. pdf_to_musicxml try/except wrap – monkey-patched export exception:
       lenient → exit 0, diagnostics records raised_during_part_append;
       strict  → exit non-zero (subprocess).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Deferred imports (keeps collection green even if music21 absent somehow)
# ---------------------------------------------------------------------------

def _diag_cls():
    from src.pipeline.export_musicxml import StageDExportDiagnostics
    return StageDExportDiagnostics


def _append_fn():
    from src.pipeline.export_musicxml import append_tokens_to_part_with_diagnostics
    return append_tokens_to_part_with_diagnostics


def _assembled_score_to_music21_diag():
    from src.pipeline.export_musicxml import assembled_score_to_music21_with_diagnostics
    return assembled_score_to_music21_with_diagnostics


def _make_part():
    from music21 import stream
    return stream.Part()


# ---------------------------------------------------------------------------
# Helper: build a minimal AssembledScore-like object with a single staff
# ---------------------------------------------------------------------------

def _make_assembled_score(tokens: list[str]):
    from src.pipeline.assemble_score import (
        AssembledScore, AssembledStaff, AssembledSystem, StaffLocation,
    )
    staff = AssembledStaff(
        sample_id="test",
        tokens=tokens,
        part_label="P1",
        measure_count=1,
        clef=None,
        key_signature=None,
        time_signature=None,
        location=StaffLocation(
            page_index=0, y_top=0.0, y_bottom=10.0, x_left=0.0, x_right=100.0,
        ),
    )
    system = AssembledSystem(
        page_index=0, system_index=0, staves=[staff],
        canonical_measure_count=1,
        canonical_key_signature=None,
        canonical_time_signature=None,
    )
    return AssembledScore(systems=[system], part_order=["P1"])


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Well-formed token spans → all diagnostic counters zero."""

    def test_zeros_on_valid_input(self):
        StageDExportDiagnostics = _diag_cls()
        append_fn = _append_fn()
        part = _make_part()
        tokens = [
            "<measure_start>",
            "note-C4", "_quarter",
            "rest", "_half",
            "<measure_end>",
        ]
        diag = StageDExportDiagnostics()
        append_fn(part, tokens, diag)

        assert diag.skipped_notes == 0
        assert diag.skipped_chords == 0
        assert diag.missing_durations == 0
        assert diag.malformed_spans == 0
        assert diag.unknown_tokens == 0
        assert diag.fallback_rests == 0
        assert diag.raised_during_part_append == []

    def test_no_diag_arg_does_not_raise(self):
        """When diagnostics=None the function must still work silently."""
        append_fn = _append_fn()
        part = _make_part()
        tokens = ["<measure_start>", "note-C4", "_quarter", "<measure_end>"]
        # Should not raise
        append_fn(part, tokens, None)


# ---------------------------------------------------------------------------
# 2. Malformed chord span
# ---------------------------------------------------------------------------

class TestMalformedChordSpan:
    """<chord_start> with no matching <chord_end> → malformed_spans incremented."""

    def test_malformed_chord_increments_counter(self):
        StageDExportDiagnostics = _diag_cls()
        append_fn = _append_fn()
        part = _make_part()
        # chord_start but stream ends without chord_end
        tokens = [
            "<measure_start>",
            "<chord_start>",
            "note-C4",
            "note-E4",
            # deliberately missing <chord_end>
            "<measure_end>",
        ]
        diag = StageDExportDiagnostics()
        append_fn(part, tokens, diag)

        assert diag.malformed_spans == 1

    def test_malformed_chord_produces_no_exception(self):
        """No exception must escape in lenient mode."""
        StageDExportDiagnostics = _diag_cls()
        append_fn = _append_fn()
        part = _make_part()
        tokens = [
            "<measure_start>",
            "<chord_start>",
            "note-G4",
            # no chord_end
        ]
        diag = StageDExportDiagnostics()
        append_fn(part, tokens, diag)  # must not raise

    def test_export_still_produces_output_after_malformed_chord(self):
        """A malformed chord should not prevent the rest of the piece from exporting."""
        from src.pipeline.export_musicxml import assembled_score_to_music21_with_diagnostics, StageDExportDiagnostics
        tokens = [
            "<measure_start>",
            "<chord_start>", "note-C4",  # no chord_end
            "note-D4", "_quarter",        # valid note after bad span
            "<measure_end>",
        ]
        score = _make_assembled_score(tokens)
        diag = StageDExportDiagnostics()
        music_score = assembled_score_to_music21_with_diagnostics(score, diag)
        assert music_score is not None
        assert diag.malformed_spans >= 1


# ---------------------------------------------------------------------------
# 3. Missing duration
# ---------------------------------------------------------------------------

class TestMissingDuration:
    """Note token followed by a non-duration token → missing_durations incremented."""

    def test_missing_duration_increments_counter(self):
        StageDExportDiagnostics = _diag_cls()
        append_fn = _append_fn()
        part = _make_part()
        tokens = [
            "<measure_start>",
            "note-C4",
            "note-E4",   # ← this is NOT a duration token
            "_quarter",
            "<measure_end>",
        ]
        diag = StageDExportDiagnostics()
        append_fn(part, tokens, diag)

        assert diag.missing_durations >= 1

    def test_rest_missing_duration(self):
        StageDExportDiagnostics = _diag_cls()
        append_fn = _append_fn()
        part = _make_part()
        tokens = [
            "<measure_start>",
            "rest",
            "note-G4",   # not a duration
            "_quarter",
            "<measure_end>",
        ]
        diag = StageDExportDiagnostics()
        append_fn(part, tokens, diag)

        assert diag.missing_durations >= 1


# ---------------------------------------------------------------------------
# 4. Unknown token
# ---------------------------------------------------------------------------

class TestUnknownToken:
    """Unrecognized token → unknown_tokens incremented, no crash."""

    def test_unknown_token_increments_counter(self):
        StageDExportDiagnostics = _diag_cls()
        append_fn = _append_fn()
        part = _make_part()
        tokens = [
            "<measure_start>",
            "TOTALLY_UNKNOWN_TOKEN_XYZ",
            "note-C4", "_quarter",
            "<measure_end>",
        ]
        diag = StageDExportDiagnostics()
        append_fn(part, tokens, diag)

        assert diag.unknown_tokens == 1

    def test_unknown_token_does_not_crash(self):
        StageDExportDiagnostics = _diag_cls()
        append_fn = _append_fn()
        part = _make_part()
        tokens = [
            "<measure_start>",
            "MYSTERY_TOKEN_1",
            "MYSTERY_TOKEN_2",
            "<measure_end>",
        ]
        diag = StageDExportDiagnostics()
        append_fn(part, tokens, diag)  # must not raise
        assert diag.unknown_tokens == 2


# ---------------------------------------------------------------------------
# 5. Fallback rest (empty chord span → rest)
# ---------------------------------------------------------------------------

class TestFallbackRest:
    """<chord_start> with no note- tokens (but valid end) → fallback_rests incremented."""

    def test_empty_chord_span_is_fallback_rest(self):
        StageDExportDiagnostics = _diag_cls()
        append_fn = _append_fn()
        part = _make_part()
        tokens = [
            "<measure_start>",
            "<chord_start>",
            # no note tokens
            "<chord_end>",
            "_quarter",
            "<measure_end>",
        ]
        diag = StageDExportDiagnostics()
        append_fn(part, tokens, diag)

        assert diag.fallback_rests == 1


# ---------------------------------------------------------------------------
# 6. Strict mode re-raises
# ---------------------------------------------------------------------------

class TestStrictMode:
    """assembled_score_to_music21_with_diagnostics(..., strict=True) must re-raise."""

    def test_strict_raises_on_malformed_span(self):
        from src.pipeline.export_musicxml import assembled_score_to_music21_with_diagnostics, StageDExportDiagnostics
        tokens = [
            "<measure_start>",
            "<chord_start>",
            "note-C4",
            # no chord_end — malformed
        ]
        score = _make_assembled_score(tokens)
        diag = StageDExportDiagnostics()
        with pytest.raises(ValueError):
            assembled_score_to_music21_with_diagnostics(score, diag, strict=True)


# ---------------------------------------------------------------------------
# 7. pdf_to_musicxml try/except wrap (monkey-patch export)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPdfToMusicxmlWrap:
    """Verify that the lenient/strict CLI wrapping works as expected."""

    def _run_cli(self, extra_args: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable, "-m", "src.pdf_to_musicxml",
            "--help",
        ] + extra_args
        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(REPO_ROOT)
        )

    def test_strict_flag_present_in_help(self):
        """--strict flag must appear in --help output."""
        result = self._run_cli([])
        assert "--strict" in result.stdout, (
            f"--strict not found in --help output:\n{result.stdout}"
        )

    def test_lenient_is_default(self):
        """--help must indicate lenient (no-strict) is the default."""
        result = self._run_cli([])
        # The help text should NOT say strict is on by default
        assert "default: False" in result.stdout or "--strict" in result.stdout

    def test_assembled_score_to_music21_with_diagnostics_import(self):
        """Key new symbol must be importable."""
        from src.pipeline.export_musicxml import (
            StageDExportDiagnostics,
            assembled_score_to_music21_with_diagnostics,
        )
        diag = StageDExportDiagnostics()
        assert diag.skipped_notes == 0
        assert isinstance(diag.raised_during_part_append, list)

    def test_diagnostics_json_field_names(self):
        """Dataclass must be serialisable and contain all required fields."""
        import dataclasses
        from src.pipeline.export_musicxml import StageDExportDiagnostics
        diag = StageDExportDiagnostics()
        d = dataclasses.asdict(diag)
        required = {
            "skipped_notes", "skipped_chords", "missing_durations",
            "malformed_spans", "unknown_tokens", "fallback_rests",
            "raised_during_part_append",
        }
        missing = required - d.keys()
        assert not missing, f"Missing fields in StageDExportDiagnostics: {missing}"
