"""Subprocess worker for eval.score_demo_eval — scores one piece, prints JSON.

This module is invoked as:
    python -m eval._score_one_piece \\
        --pred <predicted.musicxml> \\
        --ref <reference.mxl> \\
        --metrics tedn,linearized_ser,onset_f1

It loads the metric libraries (music21, zss, mir_eval), computes the requested
metrics, prints a single JSON line to stdout, then exits. The parent process
(score_demo_eval.py) collects that JSON line.

Keeping this in a separate module (rather than a -c snippet) means:
- It can be imported/tested independently
- The command line is readable in process listings
- Stack traces on failure include file+line context

Output contract:
    On success: one JSON line {"onset_f1": <float|null>, "tedn": <float|null>,
                               "linearized_ser": <float|null>}
    On any exception: exits with code 1 (stderr has the traceback)
"""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--metrics", required=True, help="comma-separated metric names")
    args = p.parse_args()

    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}

    result: dict = {}

    if "onset_f1" in metrics:
        from eval.playback import playback_f
        result["onset_f1"] = playback_f(pred=args.pred, gt=args.ref)["f"]

    if "tedn" in metrics:
        from eval.tedn import compute_tedn
        result["tedn"] = compute_tedn(args.ref, args.pred)

    if "linearized_ser" in metrics:
        from eval.linearized_musicxml import compute_linearized_ser
        result["linearized_ser"] = compute_linearized_ser(args.ref, args.pred)

    # Print JSON as the last line of stdout — parent reads it
    print(json.dumps(result))


if __name__ == "__main__":
    main()
