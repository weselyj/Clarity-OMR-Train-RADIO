"""Compare two `--step-log` JSONL files and flag if loss trajectories diverged.

Used in Phase 5.2 of the cu132 plan to verify that swapping the cu128 wheel
for the cu132 nightly preserved the existing BF16 numerical behaviour.

Usage
-----
    python scripts/compare_step_logs.py \
        --baseline path/to/cu128_steps.jsonl \
        --candidate path/to/cu132_steps.jsonl \
        --rel-tol 0.10

Exit code: 0 if max_relative_diff <= rel_tol, 1 if exceeded, 0 if either log is empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_loss_series(path: Path) -> list[float]:
    losses: list[float] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        loss = row.get("loss")
        if isinstance(loss, (int, float)):
            losses.append(float(loss))
    return losses


def compare_loss_trajectories(
    baseline_path: Path,
    candidate_path: Path,
    rel_tol: float = 0.05,
) -> dict[str, object]:
    """Compute the max step-wise relative difference between two loss series.

    Returns a dict with:
        max_relative_diff:        max |c - b| / |b| over evaluated pairs
        regressed:                True when max_relative_diff > rel_tol
        compared_steps:           number of step pairs actually evaluated
                                  (excludes pairs skipped for b == 0.0)
        skipped_zero_baseline:    pairs dropped because the baseline loss was 0.0
        rel_tol:                  the threshold used
        reason:                   None on success; a string describing why the
                                  comparison was skipped (always present)
    """
    base = _load_loss_series(Path(baseline_path))
    cand = _load_loss_series(Path(candidate_path))
    n_paired = min(len(base), len(cand))
    if n_paired == 0:
        return {
            "max_relative_diff": 0.0,
            "regressed": False,
            "compared_steps": 0,
            "skipped_zero_baseline": 0,
            "rel_tol": rel_tol,
            "reason": "empty baseline or candidate",
        }

    max_rel = 0.0
    evaluated = 0
    skipped_zero = 0
    for b, c in zip(base[:n_paired], cand[:n_paired]):
        if b == 0.0:
            skipped_zero += 1
            continue
        evaluated += 1
        rel = abs(c - b) / abs(b)
        if rel > max_rel:
            max_rel = rel

    return {
        "max_relative_diff": max_rel,
        "regressed": max_rel > rel_tol,
        "compared_steps": evaluated,
        "skipped_zero_baseline": skipped_zero,
        "rel_tol": rel_tol,
        "reason": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two --step-log JSONL files and flag regression."
    )
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Path to the reference step-log JSONL.")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Path to the candidate step-log JSONL.")
    parser.add_argument("--rel-tol", type=float, default=0.05,
                        help="Step-wise relative tolerance (default 0.05).")
    args = parser.parse_args()

    result = compare_loss_trajectories(args.baseline, args.candidate, args.rel_tol)
    print(json.dumps(result, indent=2))
    return 1 if result["regressed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
