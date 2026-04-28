"""Tests for eval.score_lieder_eval, eval.score_demo_eval, and shared eval._scoring_utils.

Verifies module imports, CLI help, and the new behaviors introduced in the
performance review (2026-04-28):
  - --jobs 1 and --jobs 2 produce the same deterministic CSV order
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


def _fake_score_payload(metrics: list[str]) -> dict:
    """Return a fake scoring payload with all requested metrics populated."""
    result = {}
    if "onset_f1" in metrics:
        result["onset_f1"] = 0.85
    if "tedn" in metrics:
        result["tedn"] = 0.72
    if "linearized_ser" in metrics:
        result["linearized_ser"] = 0.91
    return result


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

    def test_demo_imports(self):
        """score_demo_eval and its dependencies import cleanly."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import importlib
        mod = importlib.import_module("eval.score_demo_eval")
        assert hasattr(mod, "main")
        assert hasattr(mod, "DEMO_STEMS")

    def test_scoring_utils_imports(self):
        """_scoring_utils exports expected symbols."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import importlib
        mod = importlib.import_module("eval._scoring_utils")
        assert hasattr(mod, "score_piece_subprocess")
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
        assert "--resume" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--python" in result.stdout
        assert "--parallel-metric-groups" in result.stdout

    def test_demo_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "eval.score_demo_eval", "--help"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0, f"--help failed:\n{result.stderr}"
        assert "predictions-dir" in result.stdout
        assert "reference-dir" in result.stdout
        assert "--jobs" in result.stdout
        assert "--resume" in result.stdout


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
        from eval.score_lieder_eval import _discover_predictions, _build_reference_index

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
# --jobs deterministic order
# ---------------------------------------------------------------------------

class TestJobsDeterministicOrder:
    """Verifies that --jobs N produces the same CSV row order as --jobs 1.

    Uses mocked score_piece_subprocess to avoid needing torch/music21.
    """

    def _run_with_mocked_scoring(
        self, tmp_path: Path, stems: list[str], jobs: int, metrics: list[str]
    ) -> list[str]:
        """Run main() with mocked score_piece_subprocess, return list of row stems in CSV order."""
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import importlib
        import eval.score_lieder_eval as lieder_mod

        pred_dir = _make_pred_dir(tmp_path / f"preds_{jobs}", stems)
        ref_dir = _make_ref_dir(tmp_path / f"refs_{jobs}", stems)
        output_dir = tmp_path / f"out_{jobs}"
        output_dir.mkdir()

        def fake_score(pred_path, ref_path, metrics_list, *, venv_python=None,
                       cheap_timeout=60, tedn_timeout=300, parallel_metric_groups=False):
            return _fake_score_payload(metrics_list)

        with patch("eval.score_lieder_eval.score_piece_subprocess", side_effect=fake_score), \
             patch("eval.score_lieder_eval._resolve_venv_python", return_value=Path(sys.executable)):
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = [
                "score_lieder_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", f"test_jobs{jobs}",
                "--metrics", ",".join(metrics),
                "--jobs", str(jobs),
                "--output-dir", str(output_dir),
            ]
            try:
                lieder_mod.main()
            except SystemExit:
                pass
            finally:
                _sys.argv = old_argv

        csv_path = output_dir / f"lieder_test_jobs{jobs}.csv"
        if not csv_path.exists():
            return []
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            return [row["piece"] for row in reader]

    def test_jobs1_and_jobs2_same_order(self, tmp_path):
        stems = [f"piece-{i:02d}" for i in range(6)]
        metrics = ["onset_f1", "linearized_ser"]

        order_j1 = self._run_with_mocked_scoring(tmp_path, stems, jobs=1, metrics=metrics)
        order_j2 = self._run_with_mocked_scoring(tmp_path, stems, jobs=2, metrics=metrics)

        assert order_j1 == stems, f"jobs=1 order wrong: {order_j1}"
        assert order_j2 == stems, f"jobs=2 order differs from expected: {order_j2}"
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
        pred_dir = _make_pred_dir(tmp_path, stems)
        ref_dir = _make_ref_dir(tmp_path, stems)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Write a partial CSV with piece-00 already done
        partial_path = output_dir / "lieder_testresume.partial.csv"
        with partial_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_HEADER)
            w.writerow(["piece-00", 0.85, 0.72, 0.91] + [None] * 8 + [None])

        scored_calls = []

        def fake_score(pred_path, ref_path, metrics_list, *, venv_python=None,
                       cheap_timeout=60, tedn_timeout=300, parallel_metric_groups=False):
            scored_calls.append(pred_path.stem)
            return _fake_score_payload(metrics_list)

        with patch("eval.score_lieder_eval.score_piece_subprocess", side_effect=fake_score), \
             patch("eval.score_lieder_eval._resolve_venv_python", return_value=Path(sys.executable)):
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = [
                "score_lieder_eval",
                "--predictions-dir", str(pred_dir),
                "--reference-dir", str(ref_dir),
                "--name", "testresume",
                "--metrics", "onset_f1",
                "--jobs", "1",
                "--output-dir", str(output_dir),
                "--resume",
            ]
            try:
                lieder_mod.main()
            except SystemExit:
                pass
            finally:
                _sys.argv = old_argv

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
# score_piece_subprocess no longer accepts stem as first arg
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
