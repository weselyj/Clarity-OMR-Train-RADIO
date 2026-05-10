"""Tests for eval/run_lieder_eval.py.

Runs without torch: all inference subprocesses are mocked.  Only stdlib and
the functions/CLI of run_lieder_eval itself are exercised.

Run with:
    python -m pytest tests/test_run_lieder_eval.py -v
or just:
    python tests/test_run_lieder_eval.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module bootstrap — stub out heavy dependencies so import works without torch
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Stubs loaded once at module level
_scoring_utils_stub = types.ModuleType("eval._scoring_utils")
_scoring_utils_stub._resolve_venv_python = lambda explicit=None: explicit or Path(sys.executable)

_lieder_split_stub = types.ModuleType("eval.lieder_split")

# Minimal piece-like object
class _PieceStub:
    def __init__(self, name: str):
        self.stem = name

_lieder_split_stub.get_eval_pieces = lambda: [_PieceStub("piece_001"), _PieceStub("piece_002")]
_lieder_split_stub.split_hash = lambda: "8e7d206f53ae3976"

# Register stubs before importing the module under test
for _name, _mod in [
    ("eval._scoring_utils", _scoring_utils_stub),
    ("eval.lieder_split", _lieder_split_stub),
]:
    sys.modules.setdefault(_name, _mod)

# Load run_lieder_eval without running main()
_spec = importlib.util.spec_from_file_location(
    "run_lieder_eval",
    str(_REPO_ROOT / "eval" / "run_lieder_eval.py"),
)
run_lieder_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_lieder_eval)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_completed(returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode)


# ---------------------------------------------------------------------------
# 1. --config is NOT passed to the subprocess
# ---------------------------------------------------------------------------
class TestConfigNotPassedDownstream(unittest.TestCase):
    """src.pdf_to_musicxml has no --config flag.
    Verify run_inference() never includes '--config' in the subprocess command.
    """

    def test_config_absent_from_subprocess_cmd(self):
        import tempfile
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            # Don't actually run subprocess; signal success so we can inspect cmd
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Create a real yolo.pt file so the exists() check passes
            dummy_yolo = td / "yolo.pt"
            dummy_yolo.touch()
            dummy_ckpt = td / "ckpt.pt"
            dummy_ckpt.touch()
            dummy_config = td / "config.yaml"
            dummy_config.touch()
            dummy_pdf = td / "input.pdf"
            dummy_pdf.touch()
            dummy_out = td / "out.musicxml"
            dummy_work = td / "work"

            with patch("subprocess.run", side_effect=fake_run):
                run_lieder_eval.run_inference(
                    python=Path(sys.executable),
                    checkpoint=dummy_ckpt,
                    config=dummy_config,
                    pdf=dummy_pdf,
                    out=dummy_out,
                    work_dir=dummy_work,
                    stage_a_yolo=dummy_yolo,
                    stdout_log=None,
                    stderr_log=None,
                )

        self.assertIn("--stage-b-checkpoint", captured_cmd)
        self.assertNotIn("--config", captured_cmd,
                         "--config must NOT be forwarded: src.pdf_to_musicxml has no such flag")


# ---------------------------------------------------------------------------
# 2. _resolve_venv_python fallback chain (via _scoring_utils stub)
# ---------------------------------------------------------------------------
class TestResolveVenvPython(unittest.TestCase):
    def test_explicit_path_returned_when_exists(self, tmp_path=None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            fake_python = Path(td) / "python"
            fake_python.touch()
            # The real _resolve_venv_python should be importable from _scoring_utils
            from eval._scoring_utils import _resolve_venv_python
            result = _resolve_venv_python(fake_python)
            self.assertEqual(result, fake_python)

    def test_sys_executable_returned_when_no_explicit(self):
        from eval._scoring_utils import _resolve_venv_python

        # The stub just returns sys.executable when explicit is None.
        result = _resolve_venv_python(None)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# 3. _is_cache_valid
# ---------------------------------------------------------------------------
class TestIsCacheValid(unittest.TestCase):
    def test_valid_file_passes(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "score.musicxml"
            content = b"<?xml version='1.0'?><score-partwise version='3.1'>" + b"x" * 2000
            p.write_bytes(content)
            valid, reason = run_lieder_eval._is_cache_valid(p, min_bytes=1024)
            self.assertTrue(valid, reason)

    def test_too_small_file_fails(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "score.musicxml"
            p.write_bytes(b"<score-partwise>tiny</score-partwise>")
            valid, reason = run_lieder_eval._is_cache_valid(p, min_bytes=1024)
            self.assertFalse(valid)
            self.assertIn("too small", reason)

    def test_missing_xml_marker_fails(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "score.musicxml"
            p.write_bytes(b"x" * 2000)  # Large but no XML marker
            valid, reason = run_lieder_eval._is_cache_valid(p, min_bytes=100)
            self.assertFalse(valid)
            self.assertIn("score-partwise", reason)


# ---------------------------------------------------------------------------
# 4. Failure accounting: subprocess succeeds but no output → missing_output
# ---------------------------------------------------------------------------
class TestFailureAccounting(unittest.TestCase):
    def _make_args(self, tmp_path, force=False, min_bytes=1024):
        args = MagicMock()
        args.force = force
        args.min_output_bytes = min_bytes
        args.checkpoint = Path("/fake/ckpt.pt")
        args.config = Path("/fake/config.yaml")
        args.beam_width = 1
        args.max_decode_steps = 256
        args.stage_a_weights = Path("/fake/yolo.pt")
        args.timeout_sec = 30
        return args

    def test_missing_output_not_counted_as_ok(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            args = self._make_args(td)
            status_jsonl = td / "status.jsonl"
            logs_dir = td / "logs"
            logs_dir.mkdir()

            piece = _PieceStub("test_piece")
            pdf = td / "test_piece.pdf"
            pdf.touch()
            pred = td / "test_piece.musicxml"
            work_dir = td / "work"

            # run_inference returns success (returncode=0) but pred never created
            fake_result = _make_completed(0)
            durations = deque(maxlen=10)
            import threading
            lock = threading.Lock()

            with patch.object(run_lieder_eval, "run_inference", return_value=fake_result):
                rec = run_lieder_eval._run_piece(
                    i=1,
                    n_total=1,
                    piece=piece,
                    pdf=pdf,
                    pred=pred,
                    work_dir=work_dir,
                    args=args,
                    python=Path(sys.executable),
                    status_jsonl=status_jsonl,
                    logs_dir=logs_dir,
                    device="cuda",
                    rolling_durations=durations,
                    durations_lock=lock,
                )

            self.assertEqual(rec["status"], "missing_output",
                             f"Expected missing_output, got {rec['status']}")
            self.assertNotEqual(rec["status"], "done")

    def test_subprocess_failure_counted_as_failed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            args = self._make_args(td)
            status_jsonl = td / "status.jsonl"
            logs_dir = td / "logs"
            logs_dir.mkdir()

            piece = _PieceStub("test_piece2")
            pdf = td / "test_piece2.pdf"
            pdf.touch()
            pred = td / "test_piece2.musicxml"
            work_dir = td / "work2"

            fake_result = _make_completed(1)  # non-zero exit
            durations = deque(maxlen=10)
            import threading
            lock = threading.Lock()

            with patch.object(run_lieder_eval, "run_inference", return_value=fake_result):
                rec = run_lieder_eval._run_piece(
                    i=1,
                    n_total=1,
                    piece=piece,
                    pdf=pdf,
                    pred=pred,
                    work_dir=work_dir,
                    args=args,
                    python=Path(sys.executable),
                    status_jsonl=status_jsonl,
                    logs_dir=logs_dir,
                    device="cuda",
                    rolling_durations=durations,
                    durations_lock=lock,
                )

            self.assertEqual(rec["status"], "failed")


# ---------------------------------------------------------------------------
# 5. Status JSONL fields per piece
# ---------------------------------------------------------------------------
class TestStatusJsonl(unittest.TestCase):
    REQUIRED_FIELDS = {
        "index", "stem", "pdf", "prediction_path", "work_dir", "status",
        "duration_sec", "output_size_bytes", "checkpoint", "beam_width",
        "max_decode_steps", "device", "error", "config",
    }

    def test_status_jsonl_has_required_fields(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            args = MagicMock()
            args.force = False
            args.min_output_bytes = 10
            args.checkpoint = Path("/fake/ckpt.pt")
            args.config = Path("/fake/config.yaml")
            args.beam_width = 1
            args.max_decode_steps = 256
            args.stage_a_weights = Path("/fake/yolo.pt")
            args.timeout_sec = 30

            status_jsonl = td / "status.jsonl"
            logs_dir = td / "logs"
            logs_dir.mkdir()

            piece = _PieceStub("field_test")
            pdf = td / "field_test.pdf"
            pdf.touch()
            pred = td / "field_test.musicxml"
            # Write a valid-looking pred so it's cached
            pred.write_bytes(b"<score-partwise>" + b"x" * 2000)
            work_dir = td / "work"

            durations = deque(maxlen=10)
            import threading
            lock = threading.Lock()

            rec = run_lieder_eval._run_piece(
                i=1,
                n_total=1,
                piece=piece,
                pdf=pdf,
                pred=pred,
                work_dir=work_dir,
                args=args,
                python=Path(sys.executable),
                status_jsonl=status_jsonl,
                logs_dir=logs_dir,
                device="cuda:0",
                rolling_durations=durations,
                durations_lock=lock,
            )

            # Should be cached (file exists, large enough, has marker)
            missing = self.REQUIRED_FIELDS - set(rec.keys())
            self.assertFalse(missing, f"Status record missing fields: {missing}")

            # Verify JSONL was written
            self.assertTrue(status_jsonl.exists())
            lines = status_jsonl.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            missing_jsonl = self.REQUIRED_FIELDS - set(parsed.keys())
            self.assertFalse(missing_jsonl, f"JSONL missing fields: {missing_jsonl}")


# ---------------------------------------------------------------------------
# 6. Per-piece log files created
# ---------------------------------------------------------------------------
class TestPerPieceLogs(unittest.TestCase):
    def test_log_files_created_after_inference(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            args = MagicMock()
            args.force = True  # force rerun
            args.min_output_bytes = 10
            args.checkpoint = Path("/fake/ckpt.pt")
            args.config = Path("/fake/config.yaml")
            args.beam_width = 1
            args.max_decode_steps = 256
            args.stage_a_weights = Path("/fake/yolo.pt")
            args.timeout_sec = 30

            status_jsonl = td / "status.jsonl"
            logs_dir = td / "logs"
            logs_dir.mkdir()

            piece = _PieceStub("log_test")
            pdf = td / "log_test.pdf"
            pdf.touch()
            pred = td / "log_test.musicxml"
            work_dir = td / "work"

            # run_inference returns success but creates no pred — triggers missing_output
            fake_result = _make_completed(0)
            durations = deque(maxlen=10)
            import threading
            lock = threading.Lock()

            with patch.object(run_lieder_eval, "run_inference", return_value=fake_result) as mock_inf:
                run_lieder_eval._run_piece(
                    i=1,
                    n_total=1,
                    piece=piece,
                    pdf=pdf,
                    pred=pred,
                    work_dir=work_dir,
                    args=args,
                    python=Path(sys.executable),
                    status_jsonl=status_jsonl,
                    logs_dir=logs_dir,
                    device="cuda",
                    rolling_durations=durations,
                    durations_lock=lock,
                )
                # Verify that run_inference was called with stdout_log + stderr_log paths
                call_kwargs = mock_inf.call_args[1] if mock_inf.call_args.kwargs else {}
                if not call_kwargs:
                    call_kwargs = mock_inf.call_args[1]
                stdout_log = call_kwargs.get("stdout_log")
                stderr_log = call_kwargs.get("stderr_log")
                self.assertIsNotNone(stdout_log, "stdout_log should be passed to run_inference")
                self.assertIsNotNone(stderr_log, "stderr_log should be passed to run_inference")
                self.assertIn("log_test", str(stdout_log))
                self.assertIn("log_test", str(stderr_log))


# ---------------------------------------------------------------------------
# 7. --force triggers rerun even when cached output exists
# ---------------------------------------------------------------------------
class TestForceFlag(unittest.TestCase):
    def test_force_reruns_cached_output(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            args = MagicMock()
            args.force = True
            args.min_output_bytes = 10
            args.checkpoint = Path("/fake/ckpt.pt")
            args.config = Path("/fake/config.yaml")
            args.beam_width = 1
            args.max_decode_steps = 256
            args.stage_a_weights = Path("/fake/yolo.pt")
            args.timeout_sec = 30

            status_jsonl = td / "status.jsonl"
            logs_dir = td / "logs"
            logs_dir.mkdir()

            piece = _PieceStub("force_test")
            pdf = td / "force_test.pdf"
            pdf.touch()
            pred = td / "force_test.musicxml"
            # Valid-looking cached output exists
            pred.write_bytes(b"<score-partwise>" + b"x" * 2000)
            work_dir = td / "work"

            call_count = [0]

            def fake_inference(**kwargs):
                call_count[0] += 1
                return _make_completed(0)

            durations = deque(maxlen=10)
            import threading
            lock = threading.Lock()

            with patch.object(run_lieder_eval, "run_inference", side_effect=fake_inference):
                run_lieder_eval._run_piece(
                    i=1,
                    n_total=1,
                    piece=piece,
                    pdf=pdf,
                    pred=pred,
                    work_dir=work_dir,
                    args=args,
                    python=Path(sys.executable),
                    status_jsonl=status_jsonl,
                    logs_dir=logs_dir,
                    device="cuda",
                    rolling_durations=durations,
                    durations_lock=lock,
                )

            self.assertEqual(call_count[0], 1, "--force should trigger run_inference even if cache valid")


# ---------------------------------------------------------------------------
# 8. --min-output-bytes rejects too-small cached files
# ---------------------------------------------------------------------------
class TestMinOutputBytes(unittest.TestCase):
    def test_small_file_treated_as_corrupted_and_rerun(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            args = MagicMock()
            args.force = False
            args.min_output_bytes = 10_000  # require 10 KB
            args.checkpoint = Path("/fake/ckpt.pt")
            args.config = Path("/fake/config.yaml")
            args.beam_width = 1
            args.max_decode_steps = 256
            args.stage_a_weights = Path("/fake/yolo.pt")
            args.timeout_sec = 30

            status_jsonl = td / "status.jsonl"
            logs_dir = td / "logs"
            logs_dir.mkdir()

            piece = _PieceStub("small_cache_test")
            pdf = td / "small_cache_test.pdf"
            pdf.touch()
            pred = td / "small_cache_test.musicxml"
            # Cached file with valid XML marker but only 100 bytes (below 10 KB min)
            pred.write_bytes(b"<score-partwise>" + b"x" * 100)
            work_dir = td / "work"

            call_count = [0]

            def fake_inference(**kwargs):
                call_count[0] += 1
                return _make_completed(0)

            durations = deque(maxlen=10)
            import threading
            lock = threading.Lock()

            with patch.object(run_lieder_eval, "run_inference", side_effect=fake_inference):
                run_lieder_eval._run_piece(
                    i=1,
                    n_total=1,
                    piece=piece,
                    pdf=pdf,
                    pred=pred,
                    work_dir=work_dir,
                    args=args,
                    python=Path(sys.executable),
                    status_jsonl=status_jsonl,
                    logs_dir=logs_dir,
                    device="cuda",
                    rolling_durations=durations,
                    durations_lock=lock,
                )

            self.assertEqual(call_count[0], 1,
                             "--min-output-bytes should cause rerun of undersized cache")


# ---------------------------------------------------------------------------
# 9. --jobs + --devices device assignment (smoke test)
# ---------------------------------------------------------------------------
class TestJobsAndDevices(unittest.TestCase):
    def test_two_jobs_two_devices_each_piece_gets_device(self):
        """With --jobs 2 --devices cuda:0,cuda:1, two pieces should each get a device."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            # We'll capture device assignments per piece
            device_assignments: dict[str, str] = {}
            assigned_lock = __import__("threading").Lock()

            def fake_run_piece(*, piece, device, **kwargs):
                with assigned_lock:
                    device_assignments[piece.stem] = device
                # Build minimal valid record
                status_jsonl = kwargs["status_jsonl"]
                run_lieder_eval._write_status_record(status_jsonl, {
                    "stem": piece.stem, "status": "done",
                })
                return {"stem": piece.stem, "status": "done"}

            pieces = [_PieceStub("p1"), _PieceStub("p2")]
            _lieder_split_stub.get_eval_pieces = lambda: pieces

            # Build a device queue with cuda:0 and cuda:1 (1 slot each for 2 jobs)
            from queue import Queue
            device_queue: Queue[str] = Queue()
            device_list = ["cuda:0", "cuda:1"]
            for d in device_list:
                device_queue.put(d)

            collected: list[dict] = []

            def _dispatch(i_piece_tuple):
                i, piece = i_piece_tuple
                device = device_queue.get()
                try:
                    rec = fake_run_piece(
                        i=i,
                        n_total=2,
                        piece=piece,
                        pdf=td / f"{piece.stem}.pdf",
                        pred=td / f"{piece.stem}.musicxml",
                        work_dir=td / f"work_{piece.stem}",
                        args=MagicMock(),
                        python=Path(sys.executable),
                        status_jsonl=td / "status.jsonl",
                        logs_dir=td / "logs",
                        device=device,
                        rolling_durations=deque(maxlen=10),
                        durations_lock=__import__("threading").Lock(),
                    )
                    return rec
                finally:
                    device_queue.put(device)

            from concurrent.futures import ThreadPoolExecutor, as_completed
            runnable = [(1, pieces[0]), (2, pieces[1])]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(_dispatch, ip) for ip in runnable]
                for fut in as_completed(futures):
                    collected.append(fut.result())

            self.assertEqual(len(collected), 2)
            # Each piece should have gotten exactly one device
            self.assertEqual(len(device_assignments), 2)
            used_devices = set(device_assignments.values())
            self.assertTrue(used_devices.issubset({"cuda:0", "cuda:1"}))


# ---------------------------------------------------------------------------
# 10. ETA smoke test
# ---------------------------------------------------------------------------
class TestETA(unittest.TestCase):
    def test_eta_format(self):
        durations = deque([60.0, 120.0, 90.0], maxlen=10)
        eta = run_lieder_eval._format_eta(5, durations)
        # Should be HH:MM:SS and roughly 5 * mean(60,120,90)=90 = 450s = 00:07:30
        self.assertRegex(eta, r"^\d{2}:\d{2}:\d{2}$")
        # 450 seconds = 7:30 (0 hours)
        self.assertEqual(eta, "00:07:30")

    def test_eta_empty_durations(self):
        eta = run_lieder_eval._format_eta(5, deque())
        self.assertEqual(eta, "unknown")

    def test_eta_zero_remaining(self):
        durations = deque([60.0], maxlen=10)
        eta = run_lieder_eval._format_eta(0, durations)
        self.assertEqual(eta, "unknown")


# ---------------------------------------------------------------------------
# 11. --help smoke test (verifies argparse and module-level imports all OK)
# ---------------------------------------------------------------------------
class TestHelp(unittest.TestCase):
    def test_help_exits_cleanly(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        buf_out = io.StringIO()
        buf_err = io.StringIO()

        with self.assertRaises(SystemExit) as cm:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                sys.argv = ["run_lieder_eval", "--help"]
                run_lieder_eval.main()

        self.assertEqual(cm.exception.code, 0)
        help_text = buf_out.getvalue() + buf_err.getvalue()
        self.assertIn("--checkpoint", help_text)
        self.assertIn("--config", help_text)
        self.assertIn("metadata-only", help_text,
                      "--config help text should mention it's metadata-only")
        self.assertIn("--force", help_text)
        self.assertIn("--min-output-bytes", help_text)
        self.assertIn("--jobs", help_text)
        self.assertIn("--devices", help_text)
        self.assertIn("--stage-a-weights", help_text)
        self.assertIn("--stage-b-device", help_text)
        self.assertIn("--timeout-sec", help_text)
        self.assertIn("--output-dir", help_text)
        self.assertIn("--work-dir", help_text)
        self.assertIn("--python", help_text)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Restore get_eval_pieces for the main test run
    _lieder_split_stub.get_eval_pieces = lambda: [_PieceStub("piece_001"), _PieceStub("piece_002")]
    unittest.main(verbosity=2)
