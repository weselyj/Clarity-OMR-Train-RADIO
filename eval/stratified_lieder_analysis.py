# eval/stratified_lieder_analysis.py
"""Stratified onset_f1 analysis on a lieder eval CSV.

Reads the per-piece CSV produced by eval.score_lieder_eval and emits:
- Per-bucket mean onset_f1 grouped by staves_in_system
- The lc6548281 sanity check (architectural sanity from spec §"Phase 2 §1"
  line 241: should improve from DaViT's 0.05 to >= 0.10)
- Overall mean onset_f1 across all valid rows

Used by eval.decision_gate to attach the stratified breakdown to the
verdict report. The threshold for lc6548281 is the spec's, locked here
as a constant; a future spec revision is a single-line change.
"""
from __future__ import annotations
import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# Architectural sanity check (spec §"Phase 2 §1" line 241): lc6548281 should
# improve from DaViT's 0.05 baseline to >= 0.10 on Stage 3 v2.
LC6548281_SANITY_THRESHOLD = 0.10
SANITY_PIECE_ID = "lc6548281"


@dataclass(frozen=True)
class StratifiedResult:
    per_bucket: Dict[str, float]
    bucket_counts: Dict[str, int]
    overall_mean: float
    lc6548281_onset_f1: Optional[float]
    lc6548281_passes_sanity: Optional[bool]


def analyze(csv_path: Path) -> StratifiedResult:
    """Read a lieder eval CSV and return the stratified breakdown.

    Rows with missing/empty onset_f1 are excluded (scoring errors). The
    staves_in_system column is read as a string key to preserve any 1/2/3+
    or 'multi' semantics the lieder split chooses to encode.
    """
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            f1 = r.get("onset_f1")
            if f1 in (None, ""):
                continue
            try:
                f1_val = float(f1)
            except (TypeError, ValueError):
                continue
            rows.append({"piece": r.get("piece", ""), "staves_in_system": str(r.get("staves_in_system", "")), "onset_f1": f1_val})

    by_bucket: Dict[str, list] = {}
    all_vals: list = []
    lc_val: Optional[float] = None
    for r in rows:
        by_bucket.setdefault(r["staves_in_system"], []).append(r["onset_f1"])
        all_vals.append(r["onset_f1"])
        if r["piece"] == SANITY_PIECE_ID:
            lc_val = r["onset_f1"]

    per_bucket = {k: statistics.mean(v) for k, v in by_bucket.items()}
    counts = {k: len(v) for k, v in by_bucket.items()}
    overall = statistics.mean(all_vals) if all_vals else 0.0
    sanity = (lc_val >= LC6548281_SANITY_THRESHOLD) if lc_val is not None else None

    return StratifiedResult(
        per_bucket=per_bucket,
        bucket_counts=counts,
        overall_mean=overall,
        lc6548281_onset_f1=lc_val,
        lc6548281_passes_sanity=sanity,
    )


def _format_report(result: StratifiedResult) -> str:
    lines = ["# Stratified lieder onset_f1 analysis", ""]
    lines.append(f"Overall mean onset_f1 (n={sum(result.bucket_counts.values())}): **{result.overall_mean:.4f}**")
    lines.append("")
    lines.append("## By staves_in_system")
    lines.append("")
    lines.append("| Bucket | n | mean onset_f1 |")
    lines.append("|---|---|---|")
    for k in sorted(result.per_bucket.keys()):
        lines.append(f"| {k} | {result.bucket_counts[k]} | {result.per_bucket[k]:.4f} |")
    lines.append("")
    lines.append("## lc6548281 sanity check (spec §Phase 2 §1)")
    lines.append("")
    if result.lc6548281_onset_f1 is None:
        lines.append("- **NOT EVALUATED** (lc6548281 not present in eval set)")
    else:
        verdict = "PASS" if result.lc6548281_passes_sanity else "FAIL"
        lines.append(f"- onset_f1 = {result.lc6548281_onset_f1:.4f}")
        lines.append(f"- threshold = {LC6548281_SANITY_THRESHOLD:.2f} (DaViT baseline: 0.05)")
        lines.append(f"- **{verdict}**")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Stratified onset_f1 analysis on a lieder eval CSV.")
    p.add_argument("--csv", type=Path, required=True, help="Path to eval/results/lieder_<name>.csv")
    p.add_argument("--output", type=Path, default=None, help="Optional path to write the markdown report")
    args = p.parse_args()

    result = analyze(args.csv)
    report = _format_report(result)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
