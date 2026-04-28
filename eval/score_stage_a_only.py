"""Score Stage-A-only manifests against expected staff counts.

Per piece: count page-0 detections, compare to a music21-derived oracle.
Missing count = max(0, expected - detected).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import music21


def expected_p1_staves(mxl_path: Path) -> int:
    """Crude staff-count oracle: number of music21 parts in the score.

    For lieder this is approximate (a piano "part" renders as 2 staves), but
    consistent across pre/post comparisons. The metric we care about is whether
    missing_count goes DOWN after retrain, not the absolute number.
    """
    s = music21.converter.parse(str(mxl_path))
    return len(s.parts)


def score_run(manifest_dir: Path, scores_dir: Path, out_csv: Path) -> None:
    rows: list[tuple[str, int, int, int]] = []
    for manifest in sorted(manifest_dir.glob("*_stage_a.jsonl")):
        piece = manifest.stem.removesuffix("_stage_a")
        mxl = scores_dir / f"{piece}.mxl"
        if not mxl.exists():
            continue
        try:
            expected = expected_p1_staves(mxl)
        except Exception:
            continue
        detected = 0
        with manifest.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("page") == 0:
                    detected += 1
        rows.append((piece, expected, detected, max(0, expected - detected)))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["piece", "expected_p1_staves", "detected_p1_staves", "missing_count"])
        w.writerows(rows)

    if rows:
        total_expected = sum(r[1] for r in rows)
        total_detected = sum(r[2] for r in rows)
        total_missing = sum(r[3] for r in rows)
        print(f"Pieces scored: {len(rows)}")
        print(f"Total expected p1 staves: {total_expected}")
        print(f"Total detected p1 staves: {total_detected}")
        print(f"Total missing: {total_missing}")
        if total_expected:
            print(f"Recall (lower-bound): {total_detected/total_expected:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=Path("data/openscore_lieder/scores"),
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    score_run(args.manifest_dir, args.scores_dir, args.out_csv)


if __name__ == "__main__":
    main()
