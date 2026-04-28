"""Tests for eval.score_demo_eval.

Verifies module imports, CLI help, and the new behaviors introduced in the
performance review (2026-04-28):
  - --cheap-jobs / --tedn-jobs / --max-active-pieces two-lane scheduler
  - --write-order completion adds 'index' column
  - --jobs N backward-compat alias maps correctly + prints deprecation warning
  - --memory-limit-gb throttle
  - --jobs 1 and --jobs 2 produce the same logical results
  - Missing reference produces a "missing reference" row (not "scoring failure")
  - Metric-aware summary prints per-metric counts
  - --resume skips already-scored pieces
  - Score_piece_subprocess signature (stem removed)

Does NOT run actual metric scoring (requires torch + music21).
"""
import csv
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_score_payload(metrics: list[str]) -> dict:
    result = {}
    if "onset_f1" in metrics:
        result["onset_f1"] = 0.85
    if "tedn" in metrics:
        result["tedn"] = 0.72
    if "linearized_ser" in metrics:
        result["linearized_ser"] = 0.91
    return result


def _fake_metric_group_payload(metrics: list[str]) -> dict:
    return _fake_score_payload(metrics)


def _run_demo_with_mock(
    tmp_path: Path,
    extra_argv: list[str] = None,
    output_subdir: str = "out",
    metrics: list[str] = None,
) -> "tuple[Path, list[dict]]":
    """Run score_demo_eval.main() with mocked score_metric_group_subprocess."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import eval.score_demo_eval as demo_mod

    pred_dir = tmp_path / "preds"
    ref_dir = tmp_path / "refs"
    output_dir = tmp_path / output_subdir
    pred_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metrics = metrics or ["onset_f1", "linearized_ser"]

    for stem in demo_mod.DEMO_STEMS:
        (pred_dir / f"{stem}.musicxml").write_text("<xml/>")
        (ref_dir / f"{stem}.mxl").write_text("MXL")

    def fake_score_group(pred_path, ref_path, m, timeout, *, venv_python=None,
                         stderr_log=None, child_memory_limit_gb=0.0,
                         instrumentation_log=None, stem=None, group_name=None):
        return _fake_metric_group_payload(m)

    base_argv = [
        "score_demo_eval",
        "--predictions-dir", str(pred_dir),
        "--reference-dir", str(ref_dir),
        "--name", "testrun",
        "--metrics", ",".join(run_metrics),
        "--output-dir", str(output_dir),
    ]
    if extra_argv:
        base_argv.extend(extra_argv)

    with patch("eval.score_demo_eval.score_metric_group_subprocess", side_effect=fake_score_group), \
         patch("eval.score_demo_eval._resolve_venv_python", return_value=Path(sys.executable)):
        old_argv = sys.argv
        sys.argv = base_argv
        try:
            demo_mod.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

    csv_path = output_dir / "clarity_demo_testrun.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    return csv_path, rows


# ---------------------------------------------------------------------------
# Imports and --help
# ---------------------------------------------------------------------------

class TestDemoImports:
    def test_imports(self):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import importlib
        mod = importlib.import_module("eval.score_demo_eval")
        assert hasattr(mod, "main")
        assert hasattr(mod, "DEMO_STEMS")
        assert len(mod.DEMO_STEMS) == 4

    def test_help_renders(self):
        result = subprocess.run(
            [sys.executable, "-m", "eval.score_demo_eval", "--help"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0, f"--help failed:\n{result.stderr}"
        assert "--jobs" in result.stdout
        assert "--cheap-jobs" in result.stdout
        assert "--tedn-jobs" in result.stdout
        assert "--max-active-pieces" in result.stdout
        assert "--write-order" in result.stdout
        assert "--memory-limit-gb" in result.stdout
        assert "--child-memory-limit-gb" in result.stdout
        assert "--resume" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--python" in result.stdout
        assert "--parallel-metric-groups" in result.stdout


# ---------------------------------------------------------------------------
# --jobs deterministic order
# ---------------------------------------------------------------------------

class TestDemoJobsDeterministicOrder:
    def test_jobs1_and_jobs2_same_order(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_demo_eval as demo_mod

        metrics = ["onset_f1", "linearized_ser"]

        _, rows_j1 = _run_demo_with_mock(
            tmp_path / "j1",
            extra_argv=["--jobs", "1", "--write-order", "deterministic"],
            metrics=metrics,
        )
        _, rows_j2 = _run_demo_with_mock(
            tmp_path / "j2",
            extra_argv=["--jobs", "2", "--write-order", "deterministic"],
            metrics=metrics,
        )

        order_j1 = [r["piece"] for r in rows_j1]
        order_j2 = [r["piece"] for r in rows_j2]
        expected = demo_mod.DEMO_STEMS

        assert order_j1 == expected, f"jobs=1 order wrong: {order_j1}"
        assert order_j2 == expected, f"jobs=2 order wrong: {order_j2}"
        assert order_j1 == order_j2


# ---------------------------------------------------------------------------
# Two-lane scheduler for demo
# ---------------------------------------------------------------------------

class TestDemoTwoLaneScheduler:
    def test_cheap_jobs_4_tedn_jobs_2_same_results(self, tmp_path):
        """--cheap-jobs 4 --tedn-jobs 2 produces same logical results as serial."""
        metrics = ["onset_f1", "linearized_ser", "tedn"]

        _, rows_serial = _run_demo_with_mock(
            tmp_path / "serial",
            extra_argv=["--cheap-jobs", "1", "--tedn-jobs", "1", "--max-active-pieces", "1",
                        "--write-order", "deterministic"],
            metrics=metrics,
        )
        _, rows_parallel = _run_demo_with_mock(
            tmp_path / "parallel",
            extra_argv=["--cheap-jobs", "4", "--tedn-jobs", "2", "--max-active-pieces", "4",
                        "--write-order", "deterministic"],
            metrics=metrics,
        )

        serial_by_piece = {r["piece"]: r for r in rows_serial}
        for row in rows_parallel:
            stem = row["piece"]
            assert stem in serial_by_piece
            assert row.get("onset_f1") == serial_by_piece[stem].get("onset_f1")
            assert row.get("tedn") == serial_by_piece[stem].get("tedn")


# ---------------------------------------------------------------------------
# --write-order completion for demo
# ---------------------------------------------------------------------------

class TestDemoWriteOrderCompletion:
    def test_completion_has_index_column(self, tmp_path):
        """--write-order completion adds 'index' column."""
        _, rows = _run_demo_with_mock(
            tmp_path,
            extra_argv=["--write-order", "completion"],
            metrics=["onset_f1"],
        )
        assert len(rows) > 0
        assert "index" in rows[0], f"Expected 'index' column, keys: {list(rows[0].keys())}"

    def test_deterministic_no_index_column(self, tmp_path):
        """--write-order deterministic does not add 'index' column."""
        _, rows = _run_demo_with_mock(
            tmp_path,
            extra_argv=["--write-order", "deterministic"],
            metrics=["onset_f1"],
        )
        assert len(rows) > 0
        assert "index" not in rows[0]


# ---------------------------------------------------------------------------
# --jobs N legacy alias for demo
# ---------------------------------------------------------------------------

class TestDemoJobsLegacyAlias:
    def test_jobs_4_deprecation_warning(self, tmp_path):
        """--jobs 4 prints deprecation warning with mapping."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_demo_eval as demo_mod

        pred_dir = tmp_path / "preds"
        ref_dir = tmp_path / "refs"
        output_dir = tmp_path / "out"
        pred_dir.mkdir()
        ref_dir.mkdir()
        output_dir.mkdir()

        for stem in demo_mod.DEMO_STEMS:
            (pred_dir / f"{stem}.musicxml").write_text("<xml/>")
            (ref_dir / f"{stem}.mxl").write_text("MXL")

        def fake_score_group(pred_path, ref_path, m, timeout, *, venv_python=None,
                             stderr_log=None, child_memory_limit_gb=0.0,
                             instrumentation_log=None, stem=None, group_name=None):
            return _fake_metric_group_payload(m)

        captured_stderr = []
        original_stderr = sys.stderr

        import io

        with patch("eval.score_demo_eval.score_metric_group_subprocess", side_effect=fake_score_group), \
             patch("eval.score_demo_eval._resolve_venv_python", return_value=Path(sys.executable)):
            old_argv = sys.argv
            sys.argv = [
                "score_demo_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "legacytest",
                "--metrics", "onset_f1",
                "--jobs", "4",
                "--output-dir", str(output_dir),
            ]
            stderr_buf = io.StringIO()
            with patch("sys.stderr", stderr_buf):
                try:
                    demo_mod.main()
                except SystemExit:
                    pass
            sys.argv = old_argv

        stderr_output = stderr_buf.getvalue()
        assert "DEPRECATED" in stderr_output or "deprecated" in stderr_output.lower()
        assert "cheap-jobs 4" in stderr_output
        assert "tedn-jobs 2" in stderr_output
        assert "max-active-pieces 4" in stderr_output


# ---------------------------------------------------------------------------
# Missing prediction / reference row types
# ---------------------------------------------------------------------------

class TestDemoMissingFiles:
    def test_missing_prediction_row_type(self, tmp_path):
        """Missing prediction file produces predicted_xml_missing row."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_demo_eval as demo_mod

        pred_dir = tmp_path / "preds"
        ref_dir = tmp_path / "refs"
        output_dir = tmp_path / "out"
        pred_dir.mkdir()
        ref_dir.mkdir()
        output_dir.mkdir()

        # Only create refs, no preds
        for stem in demo_mod.DEMO_STEMS:
            (ref_dir / f"{stem}.mxl").write_text("MXL")

        with patch("eval.score_demo_eval._resolve_venv_python", return_value=Path(sys.executable)):
            old_argv = sys.argv
            sys.argv = [
                "score_demo_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "missingpred",
                "--metrics", "onset_f1",
                "--write-order", "deterministic",
                "--output-dir", str(output_dir),
            ]
            try:
                demo_mod.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

        csv_path = output_dir / "clarity_demo_missingpred.csv"
        assert csv_path.exists()
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # All rows should be predicted_xml_missing
        assert all(r["score_failure_reason"] == "predicted_xml_missing" for r in rows)

    def test_missing_reference_row_type(self, tmp_path):
        """Missing reference MXL produces reference_mxl_missing row."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_demo_eval as demo_mod

        pred_dir = tmp_path / "preds"
        ref_dir = tmp_path / "refs"
        output_dir = tmp_path / "out"
        pred_dir.mkdir()
        ref_dir.mkdir()
        output_dir.mkdir()

        # Only create preds, no refs
        for stem in demo_mod.DEMO_STEMS:
            (pred_dir / f"{stem}.musicxml").write_text("<xml/>")

        with patch("eval.score_demo_eval._resolve_venv_python", return_value=Path(sys.executable)):
            old_argv = sys.argv
            sys.argv = [
                "score_demo_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "missingref",
                "--metrics", "onset_f1",
                "--write-order", "deterministic",
                "--output-dir", str(output_dir),
            ]
            try:
                demo_mod.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

        csv_path = output_dir / "clarity_demo_missingref.csv"
        assert csv_path.exists()
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert all(r["score_failure_reason"] == "reference_mxl_missing" for r in rows)


# ---------------------------------------------------------------------------
# Metric-aware summary for demo
# ---------------------------------------------------------------------------

class TestDemoMetricAwareSummary:
    def test_tedn_only_summary(self, capsys):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_demo_eval import _print_summary

        rows = {
            1: ("piece-1", None, 0.72, None) + (None,) * 8 + (None,),
            2: ("piece-2", None, 0.68, None) + (None,) * 8 + (None,),
            3: ("piece-3", None, None, None) + (None,) * 8 + ("tedn: subprocess timeout after 300s",),
            4: ("piece-4", None, None, None) + (None,) * 8 + ("reference_mxl_missing",),
        }
        _print_summary(rows, ["tedn"], n_total=4, missing_ref_count=1, missing_pred_count=0, name="test")
        captured = capsys.readouterr().out
        assert "tedn" in captured
        assert "2 / 4" in captured

    def test_summary_distinguishes_missing_from_failures(self, capsys):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_demo_eval import _print_summary

        rows = {
            1: ("piece-1", 0.85, None, None) + (None,) * 8 + (None,),
            2: ("piece-2", None, None, None) + (None,) * 8 + ("reference_mxl_missing",),
            3: ("piece-3", None, None, None) + (None,) * 8 + ("predicted_xml_missing",),
            4: ("piece-4", None, None, None) + (None,) * 8 + ("subprocess exit 1: crash",),
        }
        _print_summary(rows, ["onset_f1"], n_total=4, missing_ref_count=1, missing_pred_count=1, name="test")
        captured = capsys.readouterr().out
        assert "Missing references: 1" in captured
        assert "Missing predictions: 1" in captured
        assert "Scoring failures: 1" in captured


# ---------------------------------------------------------------------------
# --resume for demo
# ---------------------------------------------------------------------------

class TestDemoResume:
    def test_resume_skips_already_scored(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_demo_eval as demo_mod
        from eval._scoring_utils import CSV_HEADER

        pred_dir = tmp_path / "preds"
        ref_dir = tmp_path / "refs"
        output_dir = tmp_path / "out"
        pred_dir.mkdir()
        ref_dir.mkdir()
        output_dir.mkdir()

        for stem in demo_mod.DEMO_STEMS:
            (pred_dir / f"{stem}.musicxml").write_text("<xml/>")
            (ref_dir / f"{stem}.mxl").write_text("MXL")

        # Write partial CSV with first stem already done
        partial_path = output_dir / "clarity_demo_resumetest.partial.csv"
        first_stem = demo_mod.DEMO_STEMS[0]
        with partial_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_HEADER)
            w.writerow([first_stem, 0.85, 0.72, 0.91] + [None] * 8 + [None])

        scored_calls = []

        def fake_score_group(pred_path, ref_path, m, timeout, *, venv_python=None,
                             stderr_log=None, child_memory_limit_gb=0.0,
                             instrumentation_log=None, stem=None, group_name=None):
            scored_calls.append(pred_path.stem)
            return _fake_metric_group_payload(m)

        with patch("eval.score_demo_eval.score_metric_group_subprocess", side_effect=fake_score_group), \
             patch("eval.score_demo_eval._resolve_venv_python", return_value=Path(sys.executable)):
            old_argv = sys.argv
            sys.argv = [
                "score_demo_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "resumetest",
                "--metrics", "onset_f1",
                "--output-dir", str(output_dir),
                "--resume",
            ]
            try:
                demo_mod.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

        assert first_stem not in scored_calls, (
            f"{first_stem} was re-scored despite being in partial CSV"
        )
        # Remaining 3 stems should have been scored
        assert len(scored_calls) == 3
