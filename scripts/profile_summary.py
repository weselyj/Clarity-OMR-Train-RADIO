#!/usr/bin/env python3
"""Summarize profile_step_timing.jsonl produced by train.py --profile-step-timing.

Reads one JSONL row per optimizer step and prints:
  - mean / p50 / p95 per phase (ms)
  - amortized fraction of wall_total_ms per phase
  - how often each phase fires (validation/checkpoint don't fire every step)
  - total phase-coverage % of wall (residual = overhead / uninstrumented)

The first --warm steps are dropped (CUDA caches, allocator, hub download).

Usage:
    python scripts/profile_summary.py logs/profile_step_timing.jsonl
    python scripts/profile_summary.py logs/profile_stage2.jsonl --warm 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("jsonl", type=Path, help="profile JSONL path")
    parser.add_argument(
        "--warm",
        type=int,
        default=50,
        help="Skip first N rows (warmup). Default 50.",
    )
    args = parser.parse_args()

    rows: List[dict] = []
    with args.jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: skipping invalid JSON line: {exc}", file=sys.stderr)
    if not rows:
        print(f"error: no rows in {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    analyzed = rows[args.warm:]
    if not analyzed:
        print(f"error: --warm {args.warm} skips all {total} rows", file=sys.stderr)
        sys.exit(1)

    phase_keys = set()
    for r in analyzed:
        for k in r:
            if k.endswith("_ms") and k != "wall_total_ms":
                phase_keys.add(k)

    def sort_key(k: str) -> tuple:
        if k.startswith("cpu_"):
            return (0, k)
        if k.startswith("gpu_"):
            return (1, k)
        return (2, k)

    ordered_keys = sorted(phase_keys, key=sort_key)

    wall_values = [r.get("wall_total_ms", 0.0) for r in analyzed]
    wall_mean = sum(wall_values) / len(wall_values)
    wall_p50 = percentile(wall_values, 50)
    wall_p95 = percentile(wall_values, 95)

    micro_batch_values = [r.get("micro_batches", 0) for r in analyzed]
    mb_mean = sum(micro_batch_values) / len(micro_batch_values) if micro_batch_values else 0.0

    print()
    print(f"Profile summary: {args.jsonl}")
    print(f"  Rows total: {total}, analyzed: {len(analyzed)} (skipped first {args.warm} as warmup)")
    print(f"  wall_total_ms     mean {wall_mean:8.1f}   p50 {wall_p50:8.1f}   p95 {wall_p95:8.1f}")
    print(f"  micro_batches     mean {mb_mean:8.2f}")
    print()
    print(f"  {'phase':<26}  {'mean ms':>10}  {'p50 ms':>10}  {'p95 ms':>10}  {'% of wall':>10}  {'fires/step':>10}")
    print(f"  {'-' * 26}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    sum_amortized = 0.0
    for key in ordered_keys:
        vals = [r[key] for r in analyzed if r.get(key) is not None]
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        p50 = percentile(vals, 50)
        p95 = percentile(vals, 95)
        amortized_ms = sum(vals) / len(analyzed)
        pct = amortized_ms / wall_mean * 100.0 if wall_mean > 0 else 0.0
        fires_per_step = n / len(analyzed)
        sum_amortized += amortized_ms
        print(
            f"  {key:<26}  {mean:>10.2f}  {p50:>10.2f}  {p95:>10.2f}  {pct:>9.1f}%  {fires_per_step:>10.2f}"
        )

    coverage = sum_amortized / wall_mean * 100.0 if wall_mean > 0 else 0.0
    residual_ms = max(0.0, wall_mean - sum_amortized)
    print()
    print(
        f"  Phase coverage of wall_total_ms: {coverage:5.1f}%   "
        f"(residual ~{residual_ms:.2f} ms/step is overhead / uninstrumented)"
    )
    print()


if __name__ == "__main__":
    main()
