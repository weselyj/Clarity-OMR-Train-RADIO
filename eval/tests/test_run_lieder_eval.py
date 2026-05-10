"""Tests for eval/run_lieder_eval.py.

Runs without launching a real inference pipeline: SystemInferencePipeline is
never instantiated (main() is not exercised), so torch/GPU imports are not
triggered by these tests.  Only stdlib + the pure-Python helpers of
run_lieder_eval are exercised.

Run with:
    python -m pytest eval/tests/test_run_lieder_eval.py -v
"""
from __future__ import annotations

import json
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module bootstrap — stub out heavy dependencies so import works without torch
# ---------------------------------------------------------------------------

# Minimal piece-like object used by the stubs and tests.
class _PieceStub:
    def __init__(self, name: str):
        self.stem = name


# --- eval.lieder_split stub ------------------------------------------------
_lieder_split_stub = types.ModuleType("eval.lieder_split")
_lieder_split_stub.get_eval_pieces = lambda: [_PieceStub("piece_001"), _PieceStub("piece_002")]
_lieder_split_stub.split_hash = lambda: "8e7d206f53ae3976"

# Register stubs BEFORE importing the module under test.
sys.modules.setdefault("eval.lieder_split", _lieder_split_stub)

# Now import the module under test as a proper package member.
import eval.run_lieder_eval as rle  # noqa: E402


# ---------------------------------------------------------------------------
# A. test_run_inference_uses_in_process_pipeline_not_subprocess
# ---------------------------------------------------------------------------
class TestRunInferenceInProcess(unittest.TestCase):
    """run_inference() must call pipeline.run_pdf + pipeline.export_musicxml and
    must NOT fall back to subprocess.run."""

    def test_run_inference_uses_in_process_pipeline_not_subprocess(self):
        import tempfile

        # Fake pipeline
        fake_score = object()
        fake_pipeline = MagicMock()
        fake_pipeline.run_pdf.return_value = fake_score

        # Patch subprocess so any accidental .run call raises immediately.
        def _no_subprocess(*args, **kwargs):
            raise AssertionError("run_inference must NOT call subprocess.run")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "score.pdf"
            p.touch()
            out = Path(td) / "out.musicxml"
            work = Path(td) / "work"

            with patch("subprocess.run", side_effect=_no_subprocess):
                rle.run_inference(
                    pipeline=fake_pipeline,
                    pdf=p,
                    out=out,
                    work_dir=work,
                )

        fake_pipeline.run_pdf.assert_called_once_with(p, diagnostics=unittest.mock.ANY)
        fake_pipeline.export_musicxml.assert_called_once()
        # First positional arg to export_musicxml must be the score returned by run_pdf.
        call_args = fake_pipeline.export_musicxml.call_args
        self.assertIs(call_args[0][0], fake_score)


# ---------------------------------------------------------------------------
# B. test_is_cache_valid_*
# ---------------------------------------------------------------------------
class TestIsCacheValid(unittest.TestCase):
    def test_valid_file_passes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "score.musicxml"
            content = b"<?xml version='1.0'?><score-partwise version='3.1'>" + b"x" * 2000
            p.write_bytes(content)
            valid, reason = rle._is_cache_valid(p, min_bytes=1024)
            self.assertTrue(valid, reason)

    def test_too_small_file_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "score.musicxml"
            p.write_bytes(b"<score-partwise>tiny</score-partwise>")
            valid, reason = rle._is_cache_valid(p, min_bytes=1024)
            self.assertFalse(valid)
            self.assertIn("too small", reason)

    def test_missing_xml_marker_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "score.musicxml"
            p.write_bytes(b"x" * 2000)  # large but no XML marker
            valid, reason = rle._is_cache_valid(p, min_bytes=100)
            self.assertFalse(valid)
            self.assertIn("score-partwise", reason)


# ---------------------------------------------------------------------------
# C. test_format_eta_*
# ---------------------------------------------------------------------------
class TestFormatETA(unittest.TestCase):
    def test_empty_deque_returns_unknown(self):
        eta = rle._format_eta(5, deque())
        self.assertEqual(eta, "unknown")

    def test_zero_remaining_returns_unknown(self):
        eta = rle._format_eta(0, deque([60.0]))
        self.assertEqual(eta, "unknown")

    def test_eta_with_durations(self):
        # mean([60, 120, 90]) = 90; 5 * 90 = 450s = 00:07:30
        durations = deque([60.0, 120.0, 90.0], maxlen=10)
        eta = rle._format_eta(5, durations)
        self.assertRegex(eta, r"^\d{2}:\d{2}:\d{2}$")
        self.assertEqual(eta, "00:07:30")


# ---------------------------------------------------------------------------
# D. test_status_jsonl_has_required_fields
# ---------------------------------------------------------------------------
class TestStatusJsonlRequiredFields(unittest.TestCase):
    REQUIRED_FIELDS = {
        "index", "stem", "pdf", "prediction_path", "work_dir", "status",
        "duration_sec", "output_size_bytes", "checkpoint", "config",
        "beam_width", "max_decode_steps", "device", "diagnostics_sidecar_present",
        "error",
    }

    def _make_args(self, force=False, min_bytes=10):
        args = MagicMock()
        args.force = force
        args.min_output_bytes = min_bytes
        args.checkpoint = Path("/fake/ckpt.pt")
        args.config = Path("/fake/config.yaml")
        args.beam_width = 1
        args.max_decode_steps = 256
        args.stage_b_device = "cuda"
        return args

    def test_status_record_has_required_fields_on_failure(self):
        """pipeline.run_pdf raises → status is 'failed:*'; record still complete."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            status_jsonl = td / "status.jsonl"

            piece = _PieceStub("fail_piece")
            pdf = td / "fail_piece.pdf"
            pdf.touch()
            pred = td / "fail_piece.musicxml"
            work_dir = td / "work"

            fake_pipeline = MagicMock()
            fake_pipeline.run_pdf.side_effect = RuntimeError("boom")

            args = self._make_args()

            rec = rle._run_piece(
                i=1,
                n_total=1,
                piece=piece,
                pdf=pdf,
                pred=pred,
                work_dir=work_dir,
                args=args,
                pipeline=fake_pipeline,
                status_jsonl=status_jsonl,
            )

            missing = self.REQUIRED_FIELDS - set(rec.keys())
            self.assertFalse(missing, f"Status record missing fields: {missing}")
            self.assertTrue(rec["status"].startswith("failed:"))

            # JSONL was written.
            self.assertTrue(status_jsonl.exists())
            lines = status_jsonl.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            missing_jsonl = self.REQUIRED_FIELDS - set(parsed.keys())
            self.assertFalse(missing_jsonl, f"JSONL missing fields: {missing_jsonl}")

    def test_status_record_has_required_fields_on_success(self):
        """pipeline runs, output file written → status 'done'; record complete."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            status_jsonl = td / "status.jsonl"

            piece = _PieceStub("ok_piece")
            pdf = td / "ok_piece.pdf"
            pdf.touch()
            pred = td / "ok_piece.musicxml"
            work_dir = td / "work"

            def fake_run_inference(*, pipeline, pdf, out, work_dir):
                # Write a valid-looking musicxml output + diagnostics sidecar.
                out.write_bytes(b"<score-partwise>" + b"x" * 2000)
                diag = out.with_suffix(out.suffix + ".diagnostics.json")
                diag.write_text("{}")

            args = self._make_args()

            with patch.object(rle, "run_inference", side_effect=fake_run_inference):
                rec = rle._run_piece(
                    i=1,
                    n_total=1,
                    piece=piece,
                    pdf=pdf,
                    pred=pred,
                    work_dir=work_dir,
                    args=args,
                    pipeline=MagicMock(),
                    status_jsonl=status_jsonl,
                )

            missing = self.REQUIRED_FIELDS - set(rec.keys())
            self.assertFalse(missing, f"Status record missing fields: {missing}")
            self.assertEqual(rec["status"], "done")


# ---------------------------------------------------------------------------
# E. test_run_piece_writes_diagnostics_sidecar_check
# ---------------------------------------------------------------------------
class TestDiagnosticsSidecarCheck(unittest.TestCase):
    """When output .musicxml is written but no .diagnostics.json, the status
    record's diagnostics_sidecar_present must be False."""

    def _make_args(self):
        args = MagicMock()
        args.force = True
        args.min_output_bytes = 10
        args.checkpoint = Path("/fake/ckpt.pt")
        args.config = Path("/fake/config.yaml")
        args.beam_width = 1
        args.max_decode_steps = 256
        args.stage_b_device = "cuda"
        return args

    def test_missing_sidecar_flagged_false(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            status_jsonl = td / "status.jsonl"

            piece = _PieceStub("nosidecar")
            pdf = td / "nosidecar.pdf"
            pdf.touch()
            pred = td / "nosidecar.musicxml"
            work_dir = td / "work"

            def fake_run_inference(*, pipeline, pdf, out, work_dir):
                # Write musicxml but deliberately omit the .diagnostics.json sidecar.
                out.write_bytes(b"<score-partwise>" + b"x" * 2000)

            args = self._make_args()

            with patch.object(rle, "run_inference", side_effect=fake_run_inference):
                rec = rle._run_piece(
                    i=1,
                    n_total=1,
                    piece=piece,
                    pdf=pdf,
                    pred=pred,
                    work_dir=work_dir,
                    args=args,
                    pipeline=MagicMock(),
                    status_jsonl=status_jsonl,
                )

            self.assertEqual(rec["status"], "done")
            self.assertFalse(
                rec["diagnostics_sidecar_present"],
                "diagnostics_sidecar_present should be False when .diagnostics.json absent",
            )

    def test_present_sidecar_flagged_true(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            status_jsonl = td / "status.jsonl"

            piece = _PieceStub("withsidecar")
            pdf = td / "withsidecar.pdf"
            pdf.touch()
            pred = td / "withsidecar.musicxml"
            work_dir = td / "work"

            def fake_run_inference(*, pipeline, pdf, out, work_dir):
                out.write_bytes(b"<score-partwise>" + b"x" * 2000)
                out.with_suffix(out.suffix + ".diagnostics.json").write_text("{}")

            args = self._make_args()

            with patch.object(rle, "run_inference", side_effect=fake_run_inference):
                rec = rle._run_piece(
                    i=1,
                    n_total=1,
                    piece=piece,
                    pdf=pdf,
                    pred=pred,
                    work_dir=work_dir,
                    args=args,
                    pipeline=MagicMock(),
                    status_jsonl=status_jsonl,
                )

            self.assertEqual(rec["status"], "done")
            self.assertTrue(rec["diagnostics_sidecar_present"])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
