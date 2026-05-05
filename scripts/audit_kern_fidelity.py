#!/usr/bin/env python3
"""Audit kern-conversion fidelity by running round-trip validation across all GrandStaff .krn files.

Read-only. Produces audit/kern_fidelity_report.json with per-divergence-category statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.kern_validation import compare_via_music21, summarize_divergences, CompareResult


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grandstaff-root", type=Path, default=Path("data/grandstaff"))
    parser.add_argument(
        "--output", type=Path, default=Path("audit/kern_fidelity_report.json")
    )
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="Optional cap on number of files audited (0 = all). Useful for smoke-testing the script.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=1000,
        help="Log progress every N files.",
    )
    args = parser.parse_args()

    krn_paths = sorted(args.grandstaff_root.rglob("*.krn"))
    if args.max_files > 0:
        krn_paths = krn_paths[: args.max_files]

    print(f"[audit] auditing {len(krn_paths):,} .krn files", file=sys.stderr)

    results: List[CompareResult] = []
    files_passed = 0
    files_failed_to_compare = 0
    for i, kp in enumerate(krn_paths):
        try:
            r = compare_via_music21(kp)
            results.append(r)
            if r.passed:
                files_passed += 1
        except Exception as e:
            files_failed_to_compare += 1
            # Don't stop the audit; just record an error result.
            results.append(
                CompareResult(
                    kern_path=kp,
                    ref_canonical=[],
                    our_canonical=[],
                    divergences=[],
                )
            )
        if (i + 1) % args.progress_every == 0:
            print(f"[audit] {i+1}/{len(krn_paths)}  passed={files_passed}  errors={files_failed_to_compare}", file=sys.stderr)

    summary = summarize_divergences(results)

    report = {
        "audit_run_at": datetime.now(timezone.utc).isoformat(),
        "files_audited": len(results),
        "files_passed": files_passed,
        "files_with_divergences": sum(1 for r in results if not r.passed),
        "files_failed_to_compare": files_failed_to_compare,
        "divergence_categories": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[audit] report: {args.output}", file=sys.stderr)
    print(f"[audit] {files_passed}/{len(results)} files passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
