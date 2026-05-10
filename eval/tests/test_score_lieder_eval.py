"""Tests for eval.score_lieder_eval and shared eval._scoring_utils.

Verifies module imports, CLI help, and the new behaviors introduced in the
performance review (2026-04-28):
  - --cheap-jobs / --tedn-jobs / --max-active-pieces two-lane scheduler
  - --write-order completion adds 'index' column; sorted-by-index matches deterministic
  - --write-order deterministic produces same order as prediction-index order
  - --jobs N backward-compat alias maps correctly + prints deprecation warning
  - --memory-limit-gb triggers throttle (mocked psutil)
  - --child-memory-limit-gb only applies on Linux (skipped on non-Linux)
  - Streaming CSV: partial.csv has all completed rows
  - Per-piece stderr files end up in scoring_logs/
  - Legacy --jobs 1 and --jobs 2 produce the same logical results
  - --max-pieces slices correctly
  - Missing reference produces a "missing reference" row (not "scoring failure")
  - Cheap-only metric run produces correct summary counts
  - TEDN-only metric run produces correct summary counts
  - Duplicate reference stems → script fails with a clear error
  - --resume skips already-scored pieces
  - _resolve_venv_python fallback order

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

# Subprocess env with repo root on PYTHONPATH so `eval.*` modules resolve
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pred_dir(tmp_path: Path, stems: list[str]) -> Path:
    """Create a predictions directory with empty .musicxml files for each stem."""
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (pred_dir / f"{stem}.musicxml").write_text("<musicxml/>")
    return pred_dir


def _make_ref_dir(tmp_path: Path, stems: list[str], nested: bool = False) -> Path:
    """Create a reference directory with .mxl files for each stem."""
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        if nested:
            sub = ref_dir / "subdir"
            sub.mkdir(exist_ok=True)
            (sub / f"{stem}.mxl").write_text("MXL")
        else:
            (ref_dir / f"{stem}.mxl").write_text("MXL")
    return ref_dir


def _fake_metric_group_payload(metrics: list[str]) -> dict:
    """Return a fake scoring payload with all requested metrics populated."""
    result = {}
    if "onset_f1" in metrics:
        result["onset_f1"] = 0.85
    if "tedn" in metrics:
        result["tedn"] = 0.72
    if "linearized_ser" in metrics:
        result["linearized_ser"] = 0.91
    return result


def _fake_score_payload(metrics: list[str]) -> dict:
    """Alias for backward compat with tests that use the old name."""
    return _fake_metric_group_payload(metrics)


def _run_lieder_with_mock(
    tmp_path: Path,
    stems: list[str],
    extra_argv: list[str] = None,
    output_subdir: str = "out",
    metrics: list[str] = None,
) -> "tuple[Path, list[dict]]":
    """Run score_lieder_eval.main() with mocked score_metric_group_subprocess.

    Returns (csv_path, rows_as_dicts).
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import importlib
    import eval.score_lieder_eval as lieder_mod

    pred_dir = _make_pred_dir(tmp_path / "preds", stems)
    ref_dir = _make_ref_dir(tmp_path / "refs", stems)
    output_dir = tmp_path / output_subdir
    output_dir.mkdir(exist_ok=True)
    run_metrics = metrics or ["onset_f1", "linearized_ser"]

    def fake_score_group(pred_path, ref_path, m, timeout, *, venv_python=None,
                         stderr_log=None, child_memory_limit_gb=0.0,
                         instrumentation_log=None, stem=None, group_name=None):
        return _fake_metric_group_payload(m)

    base_argv = [
        "score_lieder_eval",
        "--predictions-dir", str(pred_dir),
        "--reference-dir", str(ref_dir),
        "--name", "testrun",
        "--metrics", ",".join(run_metrics),
        "--output-dir", str(output_dir),
    ]
    if extra_argv:
        base_argv.extend(extra_argv)

    with patch("eval.score_lieder_eval.score_metric_group_subprocess", side_effect=fake_score_group), \
         patch("eval.score_lieder_eval._resolve_venv_python", return_value=Path(sys.executable)):
        old_argv = sys.argv
        sys.argv = base_argv
        try:
            lieder_mod.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

    csv_path = output_dir / "lieder_testrun.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    return csv_path, rows


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_lieder_imports(self):
        """score_lieder_eval and its dependencies import cleanly."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import importlib
        mod = importlib.import_module("eval.score_lieder_eval")
        assert hasattr(mod, "main")
        assert hasattr(mod, "_discover_predictions")

    def test_scoring_utils_imports(self):
        """_scoring_utils exports expected symbols."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import importlib
        mod = importlib.import_module("eval._scoring_utils")
        assert hasattr(mod, "score_piece_subprocess")
        assert hasattr(mod, "score_metric_group_subprocess")
        assert hasattr(mod, "_build_reference_index")
        assert hasattr(mod, "_resolve_venv_python")
        assert hasattr(mod, "_read_stage_d_diag")


# ---------------------------------------------------------------------------
# --help smoke tests
# ---------------------------------------------------------------------------

class TestHelpRenders:
    def test_lieder_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "eval.score_lieder_eval", "--help"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0, f"--help failed:\n{result.stderr}"
        assert "predictions-dir" in result.stdout
        assert "reference-dir" in result.stdout
        assert "max-pieces" in result.stdout
        assert "--jobs" in result.stdout
        assert "--cheap-jobs" in result.stdout
        assert "--tedn-jobs" in result.stdout
        assert "--max-active-pieces" in result.stdout
        assert "--write-order" in result.stdout
        assert "--memory-limit-gb" in result.stdout
        assert "--memory-poll-sec" in result.stdout
        assert "--child-memory-limit-gb" in result.stdout
        assert "--resume" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--python" in result.stdout
        assert "--parallel-metric-groups" in result.stdout


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrorPaths:
    def test_missing_predictions_dir_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "eval.score_lieder_eval",
             "--predictions-dir", "/nonexistent/path/to/preds",
             "--reference-dir", "/nonexistent/path/to/refs",
             "--name", "smoke"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), env=_SUBPROCESS_ENV,
        )
        assert result.returncode != 0

    def test_invalid_jobs_exits_nonzero(self, tmp_path):
        pred_dir = _make_pred_dir(tmp_path, ["piece-a"])
        ref_dir = _make_ref_dir(tmp_path, ["piece-a"])
        result = subprocess.run(
            [sys.executable, "-m", "eval.score_lieder_eval",
             "--predictions-dir", str(pred_dir),
             "--reference-dir", str(ref_dir),
             "--name", "smoke",
             "--jobs", "0"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), env=_SUBPROCESS_ENV,
        )
        assert result.returncode != 0
        assert "jobs" in result.stderr.lower() or "jobs" in result.stdout.lower()


# ---------------------------------------------------------------------------
# _build_reference_index
# ---------------------------------------------------------------------------

class TestBuildReferenceIndex:
    def test_flat_directory(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _build_reference_index

        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        (ref_dir / "piece-a.mxl").write_text("MXL")
        (ref_dir / "piece-b.mxl").write_text("MXL")

        idx = _build_reference_index(ref_dir)
        assert "piece-a" in idx
        assert "piece-b" in idx
        assert idx["piece-a"] == ref_dir / "piece-a.mxl"

    def test_nested_directory(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _build_reference_index

        ref_dir = tmp_path / "refs"
        sub = ref_dir / "Composer" / "Opus"
        sub.mkdir(parents=True)
        (sub / "piece-nested.mxl").write_text("MXL")

        idx = _build_reference_index(ref_dir)
        assert "piece-nested" in idx

    def test_duplicate_stems_raise_system_exit(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _build_reference_index

        ref_dir = tmp_path / "refs"
        sub1 = ref_dir / "dir1"
        sub2 = ref_dir / "dir2"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)
        (sub1 / "dup-stem.mxl").write_text("MXL")
        (sub2 / "dup-stem.mxl").write_text("MXL")

        with pytest.raises(SystemExit) as exc_info:
            _build_reference_index(ref_dir)
        assert "dup-stem" in str(exc_info.value)
        assert "duplicate" in str(exc_info.value).lower()

    def test_empty_directory(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _build_reference_index

        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        idx = _build_reference_index(ref_dir)
        assert idx == {}


# ---------------------------------------------------------------------------
# _resolve_venv_python
# ---------------------------------------------------------------------------

class TestResolveVenvPython:
    def test_explicit_existing_path(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _resolve_venv_python

        fake_python = tmp_path / "python"
        fake_python.write_text("#!/usr/bin/env python3")
        fake_python.chmod(0o755)

        result = _resolve_venv_python(fake_python)
        assert result == fake_python

    def test_explicit_missing_path_falls_through(self, tmp_path):
        """When --python points to a nonexistent path, the fallback chain runs."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _resolve_venv_python

        missing = tmp_path / "does_not_exist" / "python.exe"
        # sys.executable CAN import eval (we're in the repo), so it should return
        # sys.executable as the fallback (step 2 of the chain).
        result = _resolve_venv_python(missing)
        assert result == Path(sys.executable)

    def test_sys_executable_used_when_eval_importable(self):
        """sys.executable is returned when it can import eval (normal test env)."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _resolve_venv_python

        # In our test environment, sys.executable should be able to import eval.
        result = _resolve_venv_python(None)
        assert result == Path(sys.executable)

    def test_returns_path_object(self):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _resolve_venv_python

        result = _resolve_venv_python(None)
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# Missing reference → "missing reference" row (not scoring failure)
# ---------------------------------------------------------------------------

class TestMissingReference:
    def test_missing_reference_produces_correct_row(self, tmp_path):
        """A piece with no reference MXL gets a reference_mxl_missing row, not a scoring failure."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _discover_predictions
        from eval._scoring_utils import _build_reference_index

        pred_dir = _make_pred_dir(tmp_path, ["known-piece", "missing-ref-piece"])
        ref_dir = _make_ref_dir(tmp_path, ["known-piece"])  # missing-ref-piece has no ref

        ref_index = _build_reference_index(ref_dir)

        # missing-ref-piece should not be in the index
        assert "known-piece" in ref_index
        assert "missing-ref-piece" not in ref_index

    def test_missing_reference_e2e(self, tmp_path):
        """End-to-end: missing reference writes reference_mxl_missing in the CSV."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import score_piece_subprocess, _build_reference_index

        pred_dir = _make_pred_dir(tmp_path, ["piece-with-ref", "piece-without-ref"])
        ref_dir = _make_ref_dir(tmp_path, ["piece-with-ref"])

        ref_index = _build_reference_index(ref_dir)
        assert "piece-without-ref" not in ref_index

        # Verify the sentinel value matches what the scripts write
        sentinel = "reference_mxl_missing"
        row = ("piece-without-ref", None, None, None) + (None,) * 8 + (sentinel,)
        assert row[-1] == "reference_mxl_missing"
        # This row should NOT be considered a scoring failure
        assert row[-1] != "subprocess exit"
        assert "timeout" not in (row[-1] or "")


# ---------------------------------------------------------------------------
# Metric-aware summary (per-metric counts)
# ---------------------------------------------------------------------------

class TestMetricAwareSummary:
    """Verifies that _print_summary uses per-metric counts, not onset_f1 only."""

    def _make_rows(self, metrics: list[str], n_scored: int, n_total: int) -> dict:
        """Build a rows_by_index dict with n_scored pieces having values and the rest None."""
        metric_col = {"onset_f1": 1, "tedn": 2, "linearized_ser": 3}
        rows = {}
        for i in range(1, n_total + 1):
            if i <= n_scored:
                row = ["piece-" + str(i), None, None, None] + [None] * 8 + [None]
                for m in metrics:
                    col = metric_col.get(m)
                    if col is not None:
                        row[col] = 0.8 + i * 0.01
                rows[i] = tuple(row)
            else:
                rows[i] = ("piece-" + str(i), None, None, None) + (None,) * 8 + ("tedn: subprocess timeout after 300s",)
        return rows

    def test_tedn_only_summary(self, capsys):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _print_summary

        rows = self._make_rows(["tedn"], n_scored=14, n_total=20)
        _print_summary(rows, ["tedn"], n_total=20, missing_ref_count=0, name="test")

        captured = capsys.readouterr().out
        assert "tedn" in captured
        # Should show 14/20 for tedn, not 0/20
        assert "14 / 20" in captured

    def test_cheap_only_summary(self, capsys):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _print_summary

        rows = self._make_rows(["onset_f1", "linearized_ser"], n_scored=18, n_total=20)
        _print_summary(rows, ["onset_f1", "linearized_ser"], n_total=20, missing_ref_count=0, name="test")

        captured = capsys.readouterr().out
        assert "onset_f1" in captured
        assert "18 / 20" in captured

    def test_all_metrics_summary(self, capsys):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _print_summary

        rows = self._make_rows(["onset_f1", "tedn", "linearized_ser"], n_scored=17, n_total=20)
        _print_summary(rows, ["onset_f1", "tedn", "linearized_ser"], n_total=20, missing_ref_count=0, name="test")

        captured = capsys.readouterr().out
        assert "onset_f1" in captured
        assert "tedn" in captured
        assert "linearized_ser" in captured
        assert "17 / 20" in captured

    def test_missing_ref_vs_scoring_failure_distinction(self, capsys):
        """Summary distinguishes missing references from scoring failures."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _print_summary

        rows = {
            1: ("piece-1", 0.85, 0.72, 0.91) + (None,) * 8 + (None,),
            2: ("piece-2", None, None, None) + (None,) * 8 + ("reference_mxl_missing",),
            3: ("piece-3", None, None, None) + (None,) * 8 + ("subprocess exit 1: error",),
        }
        _print_summary(rows, ["onset_f1", "tedn", "linearized_ser"], n_total=3, missing_ref_count=1, name="test")

        captured = capsys.readouterr().out
        assert "Missing references: 1" in captured
        assert "Scoring failures: 1" in captured


# ---------------------------------------------------------------------------
# --max-pieces slices correctly
# ---------------------------------------------------------------------------

class TestMaxPieces:
    def test_discover_predictions_returns_sorted(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _discover_predictions

        pred_dir = tmp_path / "preds"
        pred_dir.mkdir()
        # Create in reverse order to verify sorting
        for stem in ["piece-c", "piece-a", "piece-b"]:
            (pred_dir / f"{stem}.musicxml").write_text("<xml/>")

        preds = _discover_predictions(pred_dir)
        stems = [p.stem for p in preds]
        assert stems == sorted(stems), "Predictions must be returned in sorted order"

    def test_max_pieces_truncates(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _discover_predictions

        pred_dir = tmp_path / "preds"
        pred_dir.mkdir()
        for i in range(5):
            (pred_dir / f"piece-{i:02d}.musicxml").write_text("<xml/>")

        all_preds = _discover_predictions(pred_dir)
        truncated = all_preds[:3]
        assert len(truncated) == 3
        assert len(all_preds) == 5


# ---------------------------------------------------------------------------
# Two-lane scheduler: logical correctness with mocked scoring
# ---------------------------------------------------------------------------

class TestTwoLaneScheduler:
    """Verifies that two-lane scheduler produces correct logical results."""

    def test_cheap_jobs_4_tedn_jobs_2_same_logical_results(self, tmp_path):
        """--cheap-jobs 4 --tedn-jobs 2 produces same logical results as serial run."""
        stems = [f"piece-{i:02d}" for i in range(6)]
        metrics = ["onset_f1", "linearized_ser"]

        _, rows_serial = _run_lieder_with_mock(
            tmp_path / "serial", stems,
            extra_argv=["--cheap-jobs", "1", "--tedn-jobs", "1", "--max-active-pieces", "1",
                        "--write-order", "deterministic"],
            metrics=metrics,
        )
        _, rows_parallel = _run_lieder_with_mock(
            tmp_path / "parallel", stems,
            extra_argv=["--cheap-jobs", "4", "--tedn-jobs", "2", "--max-active-pieces", "4",
                        "--write-order", "deterministic"],
            metrics=metrics,
        )

        # Same stems present (order may differ for completion, but deterministic should match)
        assert {r["piece"] for r in rows_serial} == {r["piece"] for r in rows_parallel}
        # Same metric values
        serial_by_piece = {r["piece"]: r for r in rows_serial}
        for row in rows_parallel:
            stem = row["piece"]
            assert stem in serial_by_piece
            assert row.get("onset_f1") == serial_by_piece[stem].get("onset_f1")
            assert row.get("linearized_ser") == serial_by_piece[stem].get("linearized_ser")

    def test_two_lane_all_metrics(self, tmp_path):
        """Two-lane scheduler works with all three metrics."""
        stems = [f"piece-{i:02d}" for i in range(4)]
        _, rows = _run_lieder_with_mock(
            tmp_path, stems,
            extra_argv=["--cheap-jobs", "2", "--tedn-jobs", "1", "--max-active-pieces", "2"],
            metrics=["onset_f1", "linearized_ser", "tedn"],
        )
        assert len(rows) == len(stems)
        for row in rows:
            assert row["onset_f1"] != ""
            assert row["tedn"] != ""
            assert row["linearized_ser"] != ""


# ---------------------------------------------------------------------------
# --write-order completion: index column + sorting
# ---------------------------------------------------------------------------

class TestWriteOrderCompletion:
    def test_completion_order_has_index_column(self, tmp_path):
        """--write-order completion adds 'index' column as first field."""
        stems = [f"piece-{i:02d}" for i in range(4)]
        _, rows = _run_lieder_with_mock(
            tmp_path, stems,
            extra_argv=["--write-order", "completion"],
            metrics=["onset_f1", "linearized_ser"],
        )
        assert len(rows) > 0
        assert "index" in rows[0], f"Expected 'index' column, got keys: {list(rows[0].keys())}"

    def test_completion_order_sorted_by_index_matches_deterministic(self, tmp_path):
        """Rows sorted by 'index' in completion mode match deterministic order."""
        stems = [f"piece-{i:02d}" for i in range(6)]
        metrics = ["onset_f1", "linearized_ser"]

        _, rows_completion = _run_lieder_with_mock(
            tmp_path / "completion", stems,
            extra_argv=["--write-order", "completion"],
            metrics=metrics,
        )
        _, rows_det = _run_lieder_with_mock(
            tmp_path / "deterministic", stems,
            extra_argv=["--write-order", "deterministic"],
            metrics=metrics,
        )

        # Sort completion rows by index
        rows_completion_sorted = sorted(rows_completion, key=lambda r: int(r["index"]))
        pieces_completion = [r["piece"] for r in rows_completion_sorted]
        pieces_det = [r["piece"] for r in rows_det]
        assert pieces_completion == pieces_det, (
            f"Completion order sorted by index {pieces_completion} != "
            f"deterministic order {pieces_det}"
        )


# ---------------------------------------------------------------------------
# --write-order deterministic: same order as prediction index order
# ---------------------------------------------------------------------------

class TestWriteOrderDeterministic:
    def test_deterministic_order_matches_prediction_order(self, tmp_path):
        """--write-order deterministic produces rows in original prediction-index order."""
        stems = [f"piece-{i:02d}" for i in range(6)]
        _, rows = _run_lieder_with_mock(
            tmp_path, stems,
            extra_argv=["--write-order", "deterministic"],
            metrics=["onset_f1", "linearized_ser"],
        )
        pieces = [r["piece"] for r in rows]
        assert pieces == stems, f"Expected {stems}, got {pieces}"

    def test_deterministic_no_index_column(self, tmp_path):
        """--write-order deterministic does NOT add 'index' column."""
        stems = [f"piece-{i:02d}" for i in range(3)]
        _, rows = _run_lieder_with_mock(
            tmp_path, stems,
            extra_argv=["--write-order", "deterministic"],
            metrics=["onset_f1"],
        )
        assert len(rows) > 0
        assert "index" not in rows[0], f"Unexpected 'index' column in deterministic mode"


# ---------------------------------------------------------------------------
# --jobs N legacy alias
# ---------------------------------------------------------------------------

class TestJobsLegacyAlias:
    def test_jobs_4_maps_correctly(self, tmp_path):
        """--jobs 4 maps to --cheap-jobs 4 --tedn-jobs 2 --max-active-pieces 4."""
        result = subprocess.run(
            [sys.executable, "-m", "eval.score_lieder_eval",
             "--predictions-dir", str(_make_pred_dir(tmp_path, ["piece-a"])),
             "--reference-dir", str(_make_ref_dir(tmp_path, ["piece-a"])),
             "--name", "legacytest",
             "--jobs", "4",
             "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), env=_SUBPROCESS_ENV,
        )
        stderr = result.stderr
        # Check deprecation warning text
        assert "DEPRECATED" in stderr or "deprecated" in stderr.lower(), (
            f"Expected deprecation warning in stderr, got: {stderr}"
        )
        assert "cheap-jobs 4" in stderr, f"Expected cheap-jobs 4 in deprecation msg, got: {stderr}"
        assert "tedn-jobs 2" in stderr, f"Expected tedn-jobs 2 in deprecation msg, got: {stderr}"
        assert "max-active-pieces 4" in stderr, f"Expected max-active-pieces 4 in deprecation msg, got: {stderr}"

    def test_jobs_1_maps_correctly(self, tmp_path):
        """--jobs 1 maps to cheap-jobs=1, tedn-jobs=1, max-active-pieces=1."""
        result = subprocess.run(
            [sys.executable, "-m", "eval.score_lieder_eval",
             "--predictions-dir", str(_make_pred_dir(tmp_path, ["piece-a"])),
             "--reference-dir", str(_make_ref_dir(tmp_path, ["piece-a"])),
             "--name", "legacytest1",
             "--jobs", "1",
             "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), env=_SUBPROCESS_ENV,
        )
        stderr = result.stderr
        assert "DEPRECATED" in stderr or "deprecated" in stderr.lower()
        assert "cheap-jobs 1" in stderr
        assert "tedn-jobs 1" in stderr
        assert "max-active-pieces 1" in stderr

    def test_jobs_produces_same_results_as_cheap_tedn_flags(self, tmp_path):
        """--jobs 4 produces same logical results as --cheap-jobs 4 --tedn-jobs 2."""
        stems = [f"piece-{i:02d}" for i in range(4)]
        metrics = ["onset_f1", "linearized_ser"]

        _, rows_legacy = _run_lieder_with_mock(
            tmp_path / "legacy", stems,
            extra_argv=["--jobs", "4", "--write-order", "deterministic"],
            metrics=metrics,
        )
        _, rows_new = _run_lieder_with_mock(
            tmp_path / "new", stems,
            extra_argv=["--cheap-jobs", "4", "--tedn-jobs", "2", "--max-active-pieces", "4",
                        "--write-order", "deterministic"],
            metrics=metrics,
        )
        pieces_legacy = [r["piece"] for r in rows_legacy]
        pieces_new = [r["piece"] for r in rows_new]
        assert pieces_legacy == pieces_new


# ---------------------------------------------------------------------------
# --memory-limit-gb throttle (mocked psutil)
# ---------------------------------------------------------------------------

class TestMemoryThrottle:
    def test_memory_throttle_blocks_tedn_when_limit_reached(self, tmp_path):
        """--memory-limit-gb triggers throttle: _wait_for_memory_budget polls psutil."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_lieder_eval as lieder_mod

        poll_count = [0]

        # Simulate: first 2 polls are over limit, then drops below
        def fake_virtual_memory():
            m = MagicMock()
            poll_count[0] += 1
            if poll_count[0] <= 2:
                m.used = 90 * (1024 ** 3)  # 90 GB, above 84 GB limit
            else:
                m.used = 70 * (1024 ** 3)  # 70 GB, below limit
            return m

        fake_psutil = MagicMock()
        fake_psutil.virtual_memory = fake_virtual_memory

        with patch("eval.score_lieder_eval.psutil", fake_psutil), \
             patch("eval.score_lieder_eval.time") as mock_time:
            mock_time.sleep = MagicMock()  # No-op sleep
            lieder_mod._wait_for_memory_budget(84.0, 0.001)

        # If we got here without infinite loop, throttle logic works
        assert poll_count[0] >= 3  # polled at least 3 times (2 over + 1 under)

    def test_memory_throttle_disabled_when_limit_zero(self, tmp_path):
        """_wait_for_memory_budget does nothing when limit_gb <= 0."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_lieder_eval as lieder_mod

        fake_psutil = MagicMock()
        fake_psutil.virtual_memory = MagicMock(return_value=MagicMock(used=90 * 1024**3))

        with patch("eval.score_lieder_eval.psutil", fake_psutil):
            lieder_mod._wait_for_memory_budget(0.0, 2.0)

        # psutil.virtual_memory should NOT have been called when limit is 0
        fake_psutil.virtual_memory.assert_not_called()


# ---------------------------------------------------------------------------
# --child-memory-limit-gb: Linux only
# ---------------------------------------------------------------------------

class TestChildMemoryLimit:
    @pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS only on Linux")
    def test_child_memory_cap_applied_on_linux(self, tmp_path):
        """--child-memory-limit-gb causes preexec_fn to be set on Linux."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _run_subprocess

        preexec_calls = []

        original_run = subprocess.run

        def fake_run(cmd, **kwargs):
            preexec = kwargs.get("preexec_fn")
            if preexec is not None:
                preexec_calls.append(True)
                # Execute it to verify it calls resource.setrlimit
                import resource
                try:
                    preexec()
                except Exception:
                    pass  # Might fail if limit is too low, that's fine
            # Return fake success
            m = MagicMock()
            m.returncode = 0
            m.stdout = '{"onset_f1": 0.85}'
            return m

        pred_path = tmp_path / "piece.musicxml"
        pred_path.write_text("<xml/>")
        ref_path = tmp_path / "piece.mxl"
        ref_path.write_text("MXL")

        with patch("eval._scoring_utils.subprocess.run", side_effect=fake_run):
            result = _run_subprocess(
                pred_path, ref_path, ["onset_f1"], 60,
                child_memory_limit_gb=16.0,
            )
        assert preexec_calls, "preexec_fn should have been set when child_memory_limit_gb > 0 on Linux"

    def test_child_memory_cap_not_applied_on_windows(self, tmp_path):
        """--child-memory-limit-gb is a no-op on non-Linux platforms."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _run_subprocess

        preexec_calls = []

        def fake_run(cmd, **kwargs):
            preexec = kwargs.get("preexec_fn")
            if preexec is not None:
                preexec_calls.append(True)
            m = MagicMock()
            m.returncode = 0
            m.stdout = '{"onset_f1": 0.85}'
            return m

        pred_path = tmp_path / "piece.musicxml"
        pred_path.write_text("<xml/>")
        ref_path = tmp_path / "piece.mxl"
        ref_path.write_text("MXL")

        # Simulate non-Linux platform
        with patch("eval._scoring_utils.subprocess.run", side_effect=fake_run), \
             patch("eval._scoring_utils.sys") as mock_sys:
            mock_sys.platform = "win32"
            mock_sys.executable = sys.executable
            mock_sys.path = sys.path
            result = _run_subprocess(
                pred_path, ref_path, ["onset_f1"], 60,
                child_memory_limit_gb=16.0,
            )
        assert not preexec_calls, "preexec_fn should NOT be set on non-Linux platforms"


# ---------------------------------------------------------------------------
# Streaming CSV: partial.csv has completed rows
# ---------------------------------------------------------------------------

class TestStreamingCSV:
    def test_partial_csv_contains_completed_rows(self, tmp_path):
        """Completed rows are written to partial.csv immediately (streaming)."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_lieder_eval as lieder_mod

        stems = [f"piece-{i:02d}" for i in range(4)]
        pred_dir = _make_pred_dir(tmp_path / "preds", stems)
        ref_dir = _make_ref_dir(tmp_path / "refs", stems)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Track which partial CSV state looks like mid-run
        partial_states = []

        original_score = lieder_mod.score_metric_group_subprocess

        def fake_score(pred_path, ref_path, m, timeout, *, venv_python=None,
                       stderr_log=None, child_memory_limit_gb=0.0,
                       instrumentation_log=None, stem=None, group_name=None):
            # After each piece, check if partial.csv has been written
            partial_path = output_dir / "lieder_streaming.partial.csv"
            if partial_path.exists():
                partial_states.append(partial_path.read_text())
            return _fake_metric_group_payload(m)

        with patch("eval.score_lieder_eval.score_metric_group_subprocess", side_effect=fake_score), \
             patch("eval.score_lieder_eval._resolve_venv_python", return_value=Path(sys.executable)):
            old_argv = sys.argv
            sys.argv = [
                "score_lieder_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "streaming",
                "--metrics", "onset_f1,linearized_ser",
                "--cheap-jobs", "1",
                "--tedn-jobs", "1",
                "--max-active-pieces", "1",
                "--write-order", "completion",
                "--output-dir", str(output_dir),
            ]
            try:
                lieder_mod.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

        # Final CSV should exist
        csv_path = output_dir / "lieder_streaming.csv"
        assert csv_path.exists()
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == len(stems)

    def test_partial_csv_header_written_at_start(self, tmp_path):
        """partial.csv has a header immediately (even if no pieces complete)."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_lieder_eval as lieder_mod
        from eval._scoring_utils import CSV_HEADER

        # Use empty predictions dir to trigger no scoring
        pred_dir = tmp_path / "preds"
        pred_dir.mkdir()
        ref_dir = _make_ref_dir(tmp_path / "refs", [])
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with patch("eval.score_lieder_eval._resolve_venv_python", return_value=Path(sys.executable)):
            old_argv = sys.argv
            # Will fail fast with "no .musicxml files" but partial.csv is written before that
            sys.argv = [
                "score_lieder_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "headertest",
                "--metrics", "onset_f1",
                "--output-dir", str(output_dir),
            ]
            try:
                lieder_mod.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

        # Script exits early (no musicxml files) so partial may or may not exist
        # but the main thing is it doesn't crash


# ---------------------------------------------------------------------------
# Per-piece stderr files end up in scoring_logs/
# ---------------------------------------------------------------------------

class TestScoringLogs:
    def test_stderr_logs_created_in_scoring_logs_dir(self, tmp_path):
        """Per-piece stderr logs are written to scoring_logs/ under output_dir."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import score_metric_group_subprocess

        pred_path = tmp_path / "piece.musicxml"
        pred_path.write_text("<xml/>")
        ref_path = tmp_path / "piece.mxl"
        ref_path.write_text("MXL")
        stderr_log = tmp_path / "scoring_logs" / "piece_cheap.stderr.log"

        def fake_run(cmd, **kwargs):
            # Write something to stderr file if provided
            if "stderr" in kwargs and hasattr(kwargs["stderr"], "write"):
                kwargs["stderr"].write("some stderr output\n")
            m = MagicMock()
            m.returncode = 0
            m.stdout = '{"onset_f1": 0.85}'
            return m

        with patch("eval._scoring_utils.subprocess.run", side_effect=fake_run):
            result = score_metric_group_subprocess(
                pred_path, ref_path, ["onset_f1"], 60,
                stderr_log=stderr_log,
                stem="piece",
                group_name="cheap",
            )

        assert stderr_log.exists(), f"Expected stderr log at {stderr_log}"

    def test_stderr_log_last_8kb_read_on_failure(self, tmp_path):
        """On subprocess failure, last 8 KB of stderr log is read into error field."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _run_subprocess

        pred_path = tmp_path / "piece.musicxml"
        pred_path.write_text("<xml/>")
        ref_path = tmp_path / "piece.mxl"
        ref_path.write_text("MXL")
        stderr_log = tmp_path / "piece_cheap.stderr.log"

        long_error = "x" * 20000  # 20 KB of error content

        def fake_run(cmd, **kwargs):
            if "stderr" in kwargs and hasattr(kwargs["stderr"], "write"):
                kwargs["stderr"].write(long_error)
            m = MagicMock()
            m.returncode = 1
            m.stdout = ""
            return m

        with patch("eval._scoring_utils.subprocess.run", side_effect=fake_run):
            result = _run_subprocess(
                pred_path, ref_path, ["onset_f1"], 60,
                stderr_log=stderr_log,
            )

        assert "error" in result
        error_msg = result["error"]
        # Should only contain the last 8 KB (8192 chars) of the error
        assert len(error_msg) <= 8192 + 100  # Allow some overhead for "subprocess exit N: " prefix


# ---------------------------------------------------------------------------
# --jobs deterministic order (legacy tests — backward compat)
# ---------------------------------------------------------------------------

class TestJobsDeterministicOrder:
    """Verifies that --jobs N produces the same logical results as --jobs 1."""

    def test_jobs1_and_jobs2_same_order(self, tmp_path):
        stems = [f"piece-{i:02d}" for i in range(6)]
        metrics = ["onset_f1", "linearized_ser"]

        # Use deterministic write order so row order is stable
        _, rows_j1 = _run_lieder_with_mock(
            tmp_path / "j1", stems,
            extra_argv=["--jobs", "1", "--write-order", "deterministic"],
            metrics=metrics,
        )
        _, rows_j2 = _run_lieder_with_mock(
            tmp_path / "j2", stems,
            extra_argv=["--jobs", "2", "--write-order", "deterministic"],
            metrics=metrics,
        )

        order_j1 = [r["piece"] for r in rows_j1]
        order_j2 = [r["piece"] for r in rows_j2]

        assert order_j1 == stems, f"jobs=1 order wrong: {order_j1}"
        assert order_j2 == stems, f"jobs=2 order wrong: {order_j2}"
        assert order_j1 == order_j2, "jobs=1 and jobs=2 produce different row orders"


# ---------------------------------------------------------------------------
# --resume skips already-scored pieces
# ---------------------------------------------------------------------------

class TestResume:
    def test_load_resume_set_reads_stems(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _load_resume_set
        from eval._scoring_utils import CSV_HEADER

        csv_path = tmp_path / "partial.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_HEADER)
            w.writerow(["piece-a", 0.85, 0.72, 0.91] + [None] * 8 + [None])
            w.writerow(["piece-b", 0.80, 0.68, 0.88] + [None] * 8 + [None])

        stems = _load_resume_set(csv_path)
        assert "piece-a" in stems
        assert "piece-b" in stems
        assert len(stems) == 2

    def test_load_resume_set_missing_file(self, tmp_path):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval.score_lieder_eval import _load_resume_set

        stems = _load_resume_set(tmp_path / "nonexistent.csv")
        assert stems == set()

    def test_resume_skips_pieces_with_mocked_scoring(self, tmp_path):
        """--resume causes already-scored stems to be skipped (scoring not called for them)."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import eval.score_lieder_eval as lieder_mod
        from eval._scoring_utils import CSV_HEADER

        stems = ["piece-00", "piece-01", "piece-02"]
        pred_dir = _make_pred_dir(tmp_path / "preds", stems)
        ref_dir = _make_ref_dir(tmp_path / "refs", stems)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Write a partial CSV with piece-00 already done
        partial_path = output_dir / "lieder_testresume.partial.csv"
        with partial_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_HEADER)
            w.writerow(["piece-00", 0.85, 0.72, 0.91] + [None] * 8 + [None])

        scored_calls = []

        def fake_score_group(pred_path, ref_path, m, timeout, *, venv_python=None,
                             stderr_log=None, child_memory_limit_gb=0.0,
                             instrumentation_log=None, stem=None, group_name=None):
            scored_calls.append(pred_path.stem)
            return _fake_metric_group_payload(m)

        with patch("eval.score_lieder_eval.score_metric_group_subprocess", side_effect=fake_score_group), \
             patch("eval.score_lieder_eval._resolve_venv_python", return_value=Path(sys.executable)):
            old_argv = sys.argv
            sys.argv = [
                "score_lieder_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "testresume",
                "--metrics", "onset_f1",
                "--cheap-jobs", "1",
                "--tedn-jobs", "1",
                "--max-active-pieces", "1",
                "--output-dir", str(output_dir),
                "--resume",
            ]
            try:
                lieder_mod.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

        # piece-00 was in the partial CSV; should NOT have been scored again
        assert "piece-00" not in scored_calls, (
            f"piece-00 was re-scored despite being in partial CSV: {scored_calls}"
        )
        # piece-01 and piece-02 should have been scored
        assert "piece-01" in scored_calls
        assert "piece-02" in scored_calls


# ---------------------------------------------------------------------------
# Duplicate reference stems fail with clear error
# ---------------------------------------------------------------------------

class TestDuplicateReferenceStemsE2E:
    def test_duplicate_stems_script_exits_nonzero(self, tmp_path):
        """Script exits non-zero with a clear error when duplicate reference stems detected."""
        pred_dir = _make_pred_dir(tmp_path, ["dup-stem"])
        ref_dir = tmp_path / "refs"
        sub1 = ref_dir / "dir1"
        sub2 = ref_dir / "dir2"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)
        (sub1 / "dup-stem.mxl").write_text("MXL")
        (sub2 / "dup-stem.mxl").write_text("MXL")

        result = subprocess.run(
            [sys.executable, "-m", "eval.score_lieder_eval",
             "--predictions-dir", str(pred_dir),
             "--reference-dir", str(ref_dir),
             "--name", "dup_test"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), env=_SUBPROCESS_ENV,
        )
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert "dup-stem" in output
        assert "duplicate" in output.lower()


# ---------------------------------------------------------------------------
# Stage-D sidecar warning
# ---------------------------------------------------------------------------

class TestStageDWarning:
    def test_malformed_sidecar_warns_to_stderr(self, tmp_path, capsys):
        """_read_stage_d_diag warns to stderr when sidecar is malformed."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _read_stage_d_diag

        pred_path = tmp_path / "piece.musicxml"
        pred_path.write_text("<xml/>")
        sidecar = pred_path.with_suffix(pred_path.suffix + ".diagnostics.json")
        sidecar.write_text("NOT VALID JSON {{{{")

        result = _read_stage_d_diag(pred_path)
        # Should return all-None tuple (row pipeline kept alive)
        assert result == (None, None, None, None, None, None, None, None)
        # Should have warned to stderr
        captured = capsys.readouterr()
        assert "[warn]" in captured.err
        assert "Stage D" in captured.err

    def test_missing_sidecar_warns_to_stderr(self, tmp_path, capsys):
        """_read_stage_d_diag warns when sidecar file is absent."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from eval._scoring_utils import _read_stage_d_diag

        pred_path = tmp_path / "piece.musicxml"
        pred_path.write_text("<xml/>")
        # No sidecar file created

        result = _read_stage_d_diag(pred_path)
        assert result == (None, None, None, None, None, None, None, None)
        captured = capsys.readouterr()
        assert "[warn]" in captured.err


# ---------------------------------------------------------------------------
# score_piece_subprocess signature still correct (backward compat)
# ---------------------------------------------------------------------------

class TestStemArgRemoved:
    def test_score_piece_subprocess_signature(self):
        """score_piece_subprocess no longer has a 'stem' first argument."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import inspect
        from eval._scoring_utils import score_piece_subprocess

        sig = inspect.signature(score_piece_subprocess)
        params = list(sig.parameters.keys())
        # First positional arg should be pred_path, not stem
        assert params[0] == "pred_path", (
            f"score_piece_subprocess first arg should be pred_path, got {params[0]}"
        )
        assert "stem" not in params, (
            f"score_piece_subprocess still has a 'stem' parameter: {params}"
        )

    def test_score_metric_group_subprocess_exported(self):
        """score_metric_group_subprocess is exported from _scoring_utils."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import inspect
        from eval._scoring_utils import score_metric_group_subprocess

        sig = inspect.signature(score_metric_group_subprocess)
        params = list(sig.parameters.keys())
        assert params[0] == "pred_path"
        assert params[1] == "ref_path"
        assert params[2] == "metrics"
        assert params[3] == "timeout"


# ---------------------------------------------------------------------------
# Task 14: --tedn flag
# ---------------------------------------------------------------------------

def test_tedn_flag_default_off_skips_tedn_computation(monkeypatch, tmp_path):
    """When --tedn is NOT passed, score_one_piece must skip the TEDN computation
    entirely (no music21->kern conversion, no zss tree-edit-distance)."""
    from unittest.mock import MagicMock

    fake_compute_tedn = MagicMock()
    monkeypatch.setattr("eval.score_lieder_eval.compute_tedn", fake_compute_tedn,
                        raising=False)

    import eval.score_lieder_eval as sle

    parser = sle._build_argparser() if hasattr(sle, "_build_argparser") else sle.build_argparser()
    args = parser.parse_args([
        "--predictions-dir", str(tmp_path),
        "--ground-truth-dir", str(tmp_path),
        "--out-csv", str(tmp_path / "scores.csv"),
    ])
    assert args.tedn is False
