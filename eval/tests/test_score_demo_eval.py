"""Tests for eval.score_demo_eval.

Verifies module imports, CLI help, and the new behaviors introduced in the
performance review (2026-04-28):
  - --jobs 1 and --jobs 2 produce the same deterministic CSV order
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
from unittest.mock import patch

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
        assert "--resume" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--python" in result.stdout
        assert "--parallel-metric-groups" in result.stdout


# ---------------------------------------------------------------------------
# --jobs deterministic order
# ---------------------------------------------------------------------------

class TestDemoJobsDeterministicOrder:
    def _run_with_mocked_scoring(
        self, tmp_path: Path, jobs: int, metrics: list[str]
    ) -> list[str]:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_demo_eval as demo_mod

        # Create fake predictions and refs for all 4 DEMO_STEMS
        pred_dir = tmp_path / f"preds_{jobs}"
        ref_dir = tmp_path / f"refs_{jobs}"
        output_dir = tmp_path / f"out_{jobs}"
        pred_dir.mkdir()
        ref_dir.mkdir()
        output_dir.mkdir()

        for stem in demo_mod.DEMO_STEMS:
            (pred_dir / f"{stem}.musicxml").write_text("<xml/>")
            (ref_dir / f"{stem}.mxl").write_text("MXL")

        def fake_score(pred_path, ref_path, metrics_list, *, venv_python=None,
                       cheap_timeout=60, tedn_timeout=300, parallel_metric_groups=False):
            return _fake_score_payload(metrics_list)

        with patch("eval.score_demo_eval.score_piece_subprocess", side_effect=fake_score), \
             patch("eval.score_demo_eval._resolve_venv_python", return_value=Path(sys.executable)):
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = [
                "score_demo_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", f"test_jobs{jobs}",
                "--metrics", ",".join(metrics),
                "--jobs", str(jobs),
                "--output-dir", str(output_dir),
            ]
            try:
                demo_mod.main()
            except SystemExit:
                pass
            finally:
                _sys.argv = old_argv

        csv_path = output_dir / f"clarity_demo_test_jobs{jobs}.csv"
        if not csv_path.exists():
            return []
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            return [row["piece"] for row in reader]

    def test_jobs1_and_jobs2_same_order(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_demo_eval as demo_mod

        metrics = ["onset_f1", "linearized_ser"]
        order_j1 = self._run_with_mocked_scoring(tmp_path, jobs=1, metrics=metrics)
        order_j2 = self._run_with_mocked_scoring(tmp_path, jobs=2, metrics=metrics)

        expected = demo_mod.DEMO_STEMS
        assert order_j1 == expected, f"jobs=1 order wrong: {order_j1}"
        assert order_j2 == expected, f"jobs=2 order wrong: {order_j2}"
        assert order_j1 == order_j2


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
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = [
                "score_demo_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "missingpred",
                "--metrics", "onset_f1",
                "--output-dir", str(output_dir),
            ]
            try:
                demo_mod.main()
            except SystemExit:
                pass
            finally:
                _sys.argv = old_argv

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
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = [
                "score_demo_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "missingref",
                "--metrics", "onset_f1",
                "--output-dir", str(output_dir),
            ]
            try:
                demo_mod.main()
            except SystemExit:
                pass
            finally:
                _sys.argv = old_argv

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

        def fake_score(pred_path, ref_path, metrics_list, *, venv_python=None,
                       cheap_timeout=60, tedn_timeout=300, parallel_metric_groups=False):
            scored_calls.append(pred_path.stem)
            return _fake_score_payload(metrics_list)

        with patch("eval.score_demo_eval.score_piece_subprocess", side_effect=fake_score), \
             patch("eval.score_demo_eval._resolve_venv_python", return_value=Path(sys.executable)):
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = [
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
                _sys.argv = old_argv

        assert first_stem not in scored_calls, (
            f"{first_stem} was re-scored despite being in partial CSV"
        )
        # Remaining 3 stems should have been scored
        assert len(scored_calls) == 3
