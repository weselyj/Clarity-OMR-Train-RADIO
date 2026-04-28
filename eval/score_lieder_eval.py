"""Score per-piece metric results for a Lieder eval inference run.

Two-pass design:
- Pass 1 (run_lieder_eval.py): inference only — writes predicted MusicXML
  and Stage-D diagnostics sidecars to --predictions-dir.
- Pass 2 (this script): scoring only — subprocess-isolates per-piece metric
  computation so each piece's music21/zss memory is fully reclaimed after the
  subprocess exits. The parent process stays small throughout.

This design was adopted after a 20-piece lieder run hit 43 GB committed memory
at piece 6/20 and had to be killed before pagefile exhaustion. The subprocess
isolation reclaims all music21/zss state on subprocess exit; PR #26's demo eval
ran 4/4 pieces with the parent staying ~5 GB and per-piece subprocess peaks
~2.5 GB.

Each piece is scored in up to TWO fresh child processes (eval._score_one_piece):

  1. Cheap pair (onset_f1 + linearized_ser): 60 s timeout. These metrics finish
     in <2 s even for large scores and are always attempted first.
  2. Tedn-only: 300 s timeout. May time out for very large scores; when it does
     the cheap metrics are still recovered.

Shared scoring infrastructure lives in eval._scoring_utils (also used by
eval.score_demo_eval).

Piece discovery: auto-discovers all .musicxml files in --predictions-dir, so
there is no hard-coded stem list. Reference MXLs are looked up in --reference-dir
by stem (e.g. <stem>.mxl) — point this at the openscore_lieder scores directory
(data/openscore_lieder/scores or the eval_mxl mirror).

Usage:
    venv\\Scripts\\python -m eval.score_lieder_eval \\
        --predictions-dir eval/results/lieder_mvp \\
        --reference-dir data/openscore_lieder/scores \\
        --name mvp

Output: eval/results/lieder_<name>.csv
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval._scoring_utils import (
    CSV_HEADER,
    CHEAP_METRICS,
    CHEAP_TIMEOUT_SEC,
    TEDN_TIMEOUT_SEC,
    _read_stage_d_diag,
    score_piece_subprocess,
)

# Path to our venv's Python — same one that ran lieder inference
VENV_PYTHON = Path(__file__).resolve().parents[1] / "venv" / "Scripts" / "python.exe"


def _discover_predictions(predictions_dir: Path) -> list[Path]:
    """Return sorted list of .musicxml prediction files in *predictions_dir*."""
    return sorted(predictions_dir.glob("*.musicxml"))


def _find_reference(stem: str, reference_dir: Path) -> Path | None:
    """Locate the ground-truth MXL for *stem* under *reference_dir*.

    Searches recursively — the openscore_lieder corpus is nested under
    <Composer>/<Opus>/<Song>/<id>.mxl, so a flat directory is not assumed.
    Returns the first match, or None if not found.
    """
    # Try flat first (e.g. a pre-flattened eval_mxl mirror)
    flat = reference_dir / f"{stem}.mxl"
    if flat.exists():
        return flat
    # Recursive search for the first match
    matches = list(reference_dir.rglob(f"{stem}.mxl"))
    if matches:
        return matches[0]
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Score predicted MusicXMLs from a Lieder eval inference run against "
                    "reference MXLs, subprocess-isolating per-piece metric computation."
    )
    p.add_argument(
        "--predictions-dir", type=Path, required=True,
        help="Directory containing predicted .musicxml files (output of run_lieder_eval.py). "
             "All .musicxml files in this directory are scored.",
    )
    p.add_argument(
        "--reference-dir", type=Path, required=True,
        help="Directory containing reference .mxl files. Can be the full openscore_lieder/scores "
             "tree (recursive search used) or a flat mirror. "
             "Example: data/openscore_lieder/scores",
    )
    p.add_argument(
        "--name", required=True,
        help="Run name (used for output CSV filename: eval/results/lieder_<name>.csv). "
             "Should match the name used with run_lieder_eval.py.",
    )
    p.add_argument(
        "--metrics",
        default="tedn,linearized_ser,onset_f1",
        help="Comma-separated list of metrics to compute (default: tedn,linearized_ser,onset_f1)",
    )
    p.add_argument(
        "--max-pieces", type=int, default=None,
        help="Score only the first N predictions (for validating a partial inference run). "
             "Default None scores all discovered predictions.",
    )
    args = p.parse_args()

    if not args.predictions_dir.exists():
        raise SystemExit(f"FATAL: predictions-dir not found: {args.predictions_dir}")
    if not args.reference_dir.exists():
        raise SystemExit(f"FATAL: reference-dir not found: {args.reference_dir}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    valid_metrics = {"tedn", "linearized_ser", "onset_f1"}
    unknown = set(metrics) - valid_metrics
    if unknown:
        raise SystemExit(f"FATAL: unknown metrics: {unknown}. Valid: {valid_metrics}")

    preds = _discover_predictions(args.predictions_dir)
    if not preds:
        raise SystemExit(
            f"FATAL: no .musicxml files found in {args.predictions_dir}. "
            "Run eval/run_lieder_eval.py first."
        )
    if args.max_pieces is not None:
        full_count = len(preds)
        preds = preds[: args.max_pieces]
        print(f"Truncating to first {args.max_pieces} predictions of {full_count}")

    cheap_requested = [m for m in metrics if m in CHEAP_METRICS]
    tedn_requested = [m for m in metrics if m == "tedn"]
    splitting = bool(cheap_requested) and bool(tedn_requested)

    print(f"Run name:        {args.name}")
    print(f"Predictions dir: {args.predictions_dir}")
    print(f"Reference dir:   {args.reference_dir}")
    print(f"Metrics:         {metrics}")
    print(f"Pieces:          {len(preds)}")
    if splitting:
        print(f"Scoring mode:    split (cheap={cheap_requested} @{CHEAP_TIMEOUT_SEC}s,"
              f" tedn @{TEDN_TIMEOUT_SEC}s)")
    else:
        timeout = CHEAP_TIMEOUT_SEC if not tedn_requested else TEDN_TIMEOUT_SEC
        print(f"Scoring mode:    single subprocess @{timeout}s")
    print()

    repo_root = Path(__file__).resolve().parents[1]
    n_total = len(preds)
    rows: list[tuple] = []

    for i, pred in enumerate(preds, 1):
        stem = pred.stem  # e.g. "bach-bwv123-liebster-gott"
        ref = _find_reference(stem, args.reference_dir)

        if not ref:
            print(f"[{i}/{n_total}] SKIP {stem}: reference MXL not found under {args.reference_dir}")
            rows.append((stem, None, None, None) + (None,) * 8 + ("reference_mxl_missing",))
            continue

        print(f"[{i}/{n_total}] scoring {stem} ...")

        # Stage D diagnostics are read in-process (fast JSON read, no music21)
        stage_d_cols = _read_stage_d_diag(pred)

        # Metric scoring runs in subprocess(es) -- OS reclaims all music21/zss memory
        # when the child exits, preventing the 43 GB OOM seen in the old in-process design.
        # cheap metrics and tedn are split into separate calls so a tedn timeout
        # does not discard already-computed onset_f1 / linearized_ser values.
        payload = score_piece_subprocess(
            stem, pred, ref, metrics, venv_python=VENV_PYTHON
        )

        failure_reason = payload.get("error", None)
        f1 = payload.get("onset_f1") if "onset_f1" in metrics else None
        tedn = payload.get("tedn") if "tedn" in metrics else None
        lin_ser = payload.get("linearized_ser") if "linearized_ser" in metrics else None

        rows.append((stem, f1, tedn, lin_ser) + stage_d_cols + (failure_reason,))

        tedn_str = f"{tedn:.4f}" if tedn is not None else "N/A"
        lin_str = f"{lin_ser:.4f}" if lin_ser is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        if failure_reason:
            print(f"[{i}/{n_total}] PARTIAL/FAIL {stem}: {failure_reason}")
            print(f"             onset_f1={f1_str}  tedn={tedn_str}  lin_ser={lin_str}")
        else:
            print(f"[{i}/{n_total}] {stem}: onset_f1={f1_str}  tedn={tedn_str}  lin_ser={lin_str}")

    csv_path = (repo_root / "eval/results" / f"lieder_{args.name}.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    print(f"\nResults written to {csv_path}")

    valid = [row[1] for row in rows if row[1] is not None]
    failed_count = sum(1 for row in rows if row[1] is None)
    if not valid:
        print(f"\nNo pieces scored successfully ({failed_count}/{n_total} failed/skipped).")
        return

    mean_f1 = statistics.mean(valid)
    med_f1 = statistics.median(valid)
    min_f1 = min(valid)
    max_f1 = max(valid)
    print(f"\n=== Lieder Scoring Results ({args.name}) ===")
    print(f"Pieces evaluated: {len(valid)} / {n_total} (failed/skipped: {failed_count})")
    print(f"Mean onset-F1:   {mean_f1:.4f}")
    print(f"Median onset-F1: {med_f1:.4f}")
    print(f"Min onset-F1:    {min_f1:.4f}")
    print(f"Max onset-F1:    {max_f1:.4f}")


if __name__ == "__main__":
    main()
