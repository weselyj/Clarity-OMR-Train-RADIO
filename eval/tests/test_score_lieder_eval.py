"""Smoke tests for eval.score_lieder_eval.

Verifies module imports and CLI help renders without error.
Does NOT run actual metric scoring (requires torch + music21 not available on
the Linux CI box).
"""
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Subprocess env with repo root on PYTHONPATH so `eval.*` modules resolve
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}


def test_imports():
    """score_lieder_eval and its _scoring_utils dependency import cleanly."""
    import importlib
    import sys

    # Ensure repo root is on path
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    mod = importlib.import_module("eval.score_lieder_eval")
    assert hasattr(mod, "main")
    assert hasattr(mod, "_discover_predictions")
    assert hasattr(mod, "_find_reference")


def test_help_renders(tmp_path):
    """--help exits 0 and produces non-empty output."""
    result = subprocess.run(
        [sys.executable, "-m", "eval.score_lieder_eval", "--help"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=_SUBPROCESS_ENV,
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"
    assert "predictions-dir" in result.stdout
    assert "reference-dir" in result.stdout
    assert "max-pieces" in result.stdout


def test_missing_predictions_dir_exits_nonzero():
    """Passing a nonexistent --predictions-dir exits with a non-zero code."""
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.score_lieder_eval",
            "--predictions-dir", "/nonexistent/path/to/preds",
            "--reference-dir", "/nonexistent/path/to/refs",
            "--name", "smoke",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=_SUBPROCESS_ENV,
    )
    assert result.returncode != 0
