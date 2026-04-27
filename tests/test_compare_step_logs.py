"""Smoke test for the step-log comparison utility (cu132 plan, Phase 5.1)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_step_logs import compare_loss_trajectories


def test_identical_logs_compare_within_tolerance(tmp_path: Path):
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    rows = [
        {"global_step": i, "loss": 1.0 - i * 0.001, "non_finite_loss": False}
        for i in range(1, 101)
    ]
    log_a.write_text("\n".join(json.dumps(r) for r in rows))
    log_b.write_text("\n".join(json.dumps(r) for r in rows))

    result = compare_loss_trajectories(log_a, log_b, rel_tol=0.05)
    assert result["max_relative_diff"] == pytest.approx(0.0, abs=1e-9)
    assert result["regressed"] is False
    assert result["compared_steps"] == 100


def test_diverged_logs_flag_regression(tmp_path: Path):
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    rows_a = [{"global_step": i, "loss": 1.0, "non_finite_loss": False} for i in range(1, 101)]
    rows_b = [{"global_step": i, "loss": 1.5, "non_finite_loss": False} for i in range(1, 101)]
    log_a.write_text("\n".join(json.dumps(r) for r in rows_a))
    log_b.write_text("\n".join(json.dumps(r) for r in rows_b))

    result = compare_loss_trajectories(log_a, log_b, rel_tol=0.05)
    assert result["regressed"] is True
    assert result["max_relative_diff"] > 0.4


def test_handles_truncated_baseline(tmp_path: Path):
    """Compare uses min(len(base), len(cand)) so a shorter baseline is OK."""
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    rows_a = [{"global_step": i, "loss": 1.0} for i in range(1, 51)]
    rows_b = [{"global_step": i, "loss": 1.01} for i in range(1, 101)]
    log_a.write_text("\n".join(json.dumps(r) for r in rows_a))
    log_b.write_text("\n".join(json.dumps(r) for r in rows_b))

    result = compare_loss_trajectories(log_a, log_b, rel_tol=0.05)
    assert result["compared_steps"] == 50
    assert result["max_relative_diff"] == pytest.approx(0.01, abs=1e-9)
    assert result["regressed"] is False


def test_skips_rows_without_numeric_loss(tmp_path: Path):
    """Rows with `loss=null` (e.g. cadence-skipped diag rows) are skipped."""
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    log_a.write_text(json.dumps({"global_step": 1, "loss": 1.0}) + "\n" +
                     json.dumps({"global_step": 2, "loss": None}) + "\n" +
                     json.dumps({"global_step": 3, "loss": 0.9}) + "\n")
    log_b.write_text(json.dumps({"global_step": 1, "loss": 1.0}) + "\n" +
                     json.dumps({"global_step": 2, "loss": None}) + "\n" +
                     json.dumps({"global_step": 3, "loss": 0.9}) + "\n")

    result = compare_loss_trajectories(log_a, log_b, rel_tol=0.05)
    assert result["compared_steps"] == 2  # null-loss row dropped
    assert result["regressed"] is False


def test_handles_empty_baseline(tmp_path: Path):
    log_a = tmp_path / "empty.jsonl"
    log_b = tmp_path / "b.jsonl"
    log_a.write_text("")
    log_b.write_text(json.dumps({"global_step": 1, "loss": 1.0}) + "\n")

    result = compare_loss_trajectories(log_a, log_b, rel_tol=0.05)
    assert result["compared_steps"] == 0
    assert result["regressed"] is False
    assert "reason" in result and "empty" in result["reason"].lower()


def test_handles_zero_baseline_loss_safely(tmp_path: Path):
    """Division-by-zero guard: a baseline loss of 0.0 is skipped from the rel-diff calc."""
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    log_a.write_text(json.dumps({"global_step": 1, "loss": 0.0}) + "\n" +
                     json.dumps({"global_step": 2, "loss": 1.0}) + "\n")
    log_b.write_text(json.dumps({"global_step": 1, "loss": 0.5}) + "\n" +
                     json.dumps({"global_step": 2, "loss": 1.05}) + "\n")

    result = compare_loss_trajectories(log_a, log_b, rel_tol=0.10)
    # Step 1's baseline 0.0 is skipped; step 2 gives rel=0.05 which is within tol
    assert result["max_relative_diff"] == pytest.approx(0.05, abs=1e-9)
    assert result["regressed"] is False
    # compared_steps reflects pairs actually evaluated (excludes the b==0 skip)
    assert result["compared_steps"] == 1
    assert result["skipped_zero_baseline"] == 1


def test_reason_key_always_present(tmp_path: Path):
    """The 'reason' key is in the result dict on both success and skipped paths."""
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    rows = [{"global_step": i, "loss": 1.0} for i in range(1, 6)]
    log_a.write_text("\n".join(json.dumps(r) for r in rows))
    log_b.write_text("\n".join(json.dumps(r) for r in rows))

    result = compare_loss_trajectories(log_a, log_b, rel_tol=0.05)
    assert "reason" in result
    assert result["reason"] is None  # success-path sentinel


def test_cli_returns_exit_one_on_regression(tmp_path: Path):
    """Invoking the CLI with --baseline/--candidate that diverge exits non-zero."""
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    log_a.write_text(json.dumps({"global_step": 1, "loss": 1.0}) + "\n")
    log_b.write_text(json.dumps({"global_step": 1, "loss": 2.0}) + "\n")

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "compare_step_logs.py"),
         "--baseline", str(log_a), "--candidate", str(log_b), "--rel-tol", "0.05"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1, f"expected exit 1, got {result.returncode}; stdout={result.stdout}"
    payload = json.loads(result.stdout)
    assert payload["regressed"] is True


def test_cli_returns_exit_zero_on_no_regression(tmp_path: Path):
    """Invoking the CLI on identical step-logs exits zero."""
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    rows = [{"global_step": i, "loss": 1.0} for i in range(1, 11)]
    log_a.write_text("\n".join(json.dumps(r) for r in rows))
    log_b.write_text("\n".join(json.dumps(r) for r in rows))

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "compare_step_logs.py"),
         "--baseline", str(log_a), "--candidate", str(log_b), "--rel-tol", "0.05"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"expected exit 0, got {result.returncode}; stdout={result.stdout}"
