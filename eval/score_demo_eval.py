"""Score per-piece metric results for the 4 canonical demo pieces.

Two-pass design:
- Pass 1 (run_clarity_demo_eval.py): inference only — writes predicted MusicXML
  and optional diagnostics sidecars to --predictions-dir.
- Pass 2 (this script): scoring only — subprocess-isolates per-piece metric
  computation so each piece's music21/zss memory is fully reclaimed after the
  subprocess exits. The parent process stays small throughout.

Each piece is scored in up to TWO fresh child processes (eval._score_one_piece):

  1. Cheap pair (onset_f1 + linearized_ser): 60 s timeout. These metrics finish
     in <2 s even for large scores (<=43 MB data) and are always attempted first.
  2. Tedn-only: 300 s timeout.  May time out for very large scores (Clair de
     Lune peaks >37 GB); when it does the cheap metrics are still recovered.

If only cheap metrics were requested (or only tedn), a single subprocess is
used.  The split only activates when the requested set includes both tedn and
at least one cheap metric.

Shared scoring infrastructure (subprocess dispatch, sidecar reading, CSV schema)
lives in eval._scoring_utils and is also used by eval.score_lieder_eval.

Parallelism: bounded piece-level parallelism via --jobs N (default 1, serial).
Use --jobs 2 or --jobs 3 first; increase only after watching peak RAM.

Checkpointing: writes eval/results/clarity_demo_<name>.partial.csv after each
completed piece so that a long run is resumable on interruption.

Resume: --resume reads an existing partial or full CSV and skips already-scored
pieces.

Usage:
    venv-cu132\\Scripts\\python -m eval.score_demo_eval \\
        --predictions-dir eval/results/clarity_demo_stage2_best \\
        --reference-dir data/clarity_demo/mxl \\
        --name stage2_best

Output: eval/results/clarity_demo_<name>.csv  (or --output-dir override)
"""
import argparse
import csv
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval._scoring_utils import (
    CSV_HEADER,
    CHEAP_METRICS,
    CHEAP_TIMEOUT_SEC,
    TEDN_TIMEOUT_SEC,
    _read_stage_d_diag,
    _resolve_venv_python,
    score_piece_subprocess,
)

# Backward-compat alias
SCORE_TIMEOUT_SEC = TEDN_TIMEOUT_SEC

# Canonical demo stems -- must match run_clarity_demo_eval.py
DEMO_STEMS = [
    "clair-de-lune-debussy",
    "fugue-no-2-bwv-847-in-c-minor",
    "gnossienne-no-1",
    "prelude-in-d-flat-major-op31-no1-scriabin",
]


def _load_resume_set(csv_path: Path) -> set[str]:
    """Return the set of stems that already have a complete row in *csv_path*."""
    if not csv_path.exists():
        return set()
    stems: set[str] = set()
    try:
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                piece = row.get("piece", "").strip()
                if piece:
                    stems.add(piece)
    except Exception:
        pass
    return stems


def _write_csv(csv_path: Path, rows_by_index: "dict[int, tuple]", n_total: int) -> None:
    """Write rows in original DEMO_STEMS order to *csv_path*."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for i in range(1, n_total + 1):
            if i in rows_by_index:
                w.writerow(rows_by_index[i])


def _format_progress(
    i: int,
    n_total: int,
    stem: str,
    f1,
    tedn,
    lin_ser,
    failure_reason: Optional[str],
) -> str:
    """Return a formatted progress line for piece *i*."""
    tedn_str = f"{tedn:.4f}" if tedn is not None else "N/A"
    lin_str = f"{lin_ser:.4f}" if lin_ser is not None else "N/A"
    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
    if failure_reason:
        return (
            f"[{i}/{n_total}] PARTIAL/FAIL {stem}: {failure_reason}\n"
            f"             onset_f1={f1_str}  tedn={tedn_str}  lin_ser={lin_str}"
        )
    return f"[{i}/{n_total}] {stem}: onset_f1={f1_str}  tedn={tedn_str}  lin_ser={lin_str}"


def _score_task(
    i: int,
    n_total: int,
    stem: str,
    pred: Path,
    ref: Path,
    metrics: list[str],
    venv_python: Path,
    parallel_metric_groups: bool,
) -> "tuple[int, tuple, str]":
    """Worker function: scores one piece and returns (index, row, progress_msg).

    Designed to run in a thread (ThreadPoolExecutor). Returns without printing
    so the main thread owns all stdout output (avoids interleaved lines).
    """
    stage_d_cols = _read_stage_d_diag(pred)
    payload = score_piece_subprocess(
        pred,
        ref,
        metrics,
        venv_python=venv_python,
        parallel_metric_groups=parallel_metric_groups,
    )

    failure_reason = payload.get("error", None)
    f1 = payload.get("onset_f1") if "onset_f1" in metrics else None
    tedn = payload.get("tedn") if "tedn" in metrics else None
    lin_ser = payload.get("linearized_ser") if "linearized_ser" in metrics else None

    row = (stem, f1, tedn, lin_ser) + stage_d_cols + (failure_reason,)
    msg = _format_progress(i, n_total, stem, f1, tedn, lin_ser, failure_reason)
    return i, row, msg


def _print_summary(
    rows_by_index: "dict[int, tuple]",
    metrics: list[str],
    n_total: int,
    missing_ref_count: int,
    missing_pred_count: int,
    name: str,
) -> None:
    """Print per-metric summary statistics."""
    metric_col = {
        "onset_f1": 1,
        "tedn": 2,
        "linearized_ser": 3,
    }
    all_rows = [rows_by_index[i] for i in sorted(rows_by_index)]

    _skip_reasons = {"reference_mxl_missing", "predicted_xml_missing", None, "", "None"}
    scoring_failures = sum(
        1 for row in all_rows
        if row[-1] not in _skip_reasons
    )
    pieces_evaluated = len(all_rows) - missing_ref_count - missing_pred_count

    print(f"\n=== Clarity Demo Scoring Results ({name}) ===")
    print(f"Pieces evaluated: {pieces_evaluated} / {n_total}")
    print(f"Missing predictions: {missing_pred_count}")
    print(f"Missing references: {missing_ref_count}")
    print(f"Scoring failures: {scoring_failures}")

    def _to_float(v):
        """Coerce v to float, returning None on failure (handles CSV string values)."""
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Per-metric counts
    for m in metrics:
        col = metric_col.get(m)
        if col is None:
            continue
        scored_vals = [_to_float(row[col]) for row in all_rows if _to_float(row[col]) is not None]
        print(f"  {m:<20s} {len(scored_vals)} / {n_total} scored")

    # Headline stats for the first requested metric that has data
    headline_metric = None
    headline_vals = []
    for m in metrics:
        col = metric_col.get(m)
        if col is None:
            continue
        vals = [_to_float(row[col]) for row in all_rows if _to_float(row[col]) is not None]
        if vals:
            headline_metric = m
            headline_vals = vals
            break

    if not headline_vals:
        print("\nNo pieces scored successfully.")
        return

    mean_v = statistics.mean(headline_vals)
    med_v = statistics.median(headline_vals)
    min_v = min(headline_vals)
    max_v = max(headline_vals)
    print(f"\nHeadline metric: {headline_metric}")
    print(f"  Mean:   {mean_v:.4f}")
    print(f"  Median: {med_v:.4f}")
    print(f"  Min:    {min_v:.4f}")
    print(f"  Max:    {max_v:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Score predicted MusicXMLs against reference MXLs, "
                    "subprocess-isolating per-piece metric computation."
    )
    p.add_argument(
        "--predictions-dir", type=Path, required=True,
        help="Directory containing predicted .musicxml files (output of run_clarity_demo_eval.py)",
    )
    p.add_argument(
        "--reference-dir", type=Path, required=True,
        help="Directory containing reference .mxl files",
    )
    p.add_argument(
        "--name", required=True,
        help="Run name (used for output CSV filename: eval/results/clarity_demo_<name>.csv)",
    )
    p.add_argument(
        "--metrics",
        default="tedn,linearized_ser,onset_f1",
        help="Comma-separated list of metrics to compute (default: tedn,linearized_ser,onset_f1)",
    )
    p.add_argument(
        "--jobs", type=int, default=1,
        help="Number of pieces to score concurrently via ThreadPoolExecutor (default: 1 = serial). "
             "Use --jobs 2 or --jobs 3 first; increase only after watching peak RAM.",
    )
    p.add_argument(
        "--python", type=Path, default=None, dest="python_path",
        help="Path to the Python interpreter to use for scoring subprocesses. "
             "Defaults to auto-detection via _resolve_venv_python().",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the output directory for the CSV file. "
             "Default: eval/results/ under the repo root.",
    )
    p.add_argument(
        "--resume", action="store_true", default=False,
        help="Read an existing output or partial CSV and skip already-scored pieces.",
    )
    p.add_argument(
        "--parallel-metric-groups", action="store_true", default=False,
        help="(Advanced) Run cheap and tedn metric subprocesses concurrently within each piece. "
             "Can reduce per-piece wall time but roughly doubles per-piece peak RAM. Default OFF.",
    )
    args = p.parse_args()

    # Validate --jobs
    if args.jobs < 1:
        raise SystemExit(f"FATAL: --jobs must be >= 1, got {args.jobs}")

    if not args.predictions_dir.exists():
        raise SystemExit(f"FATAL: predictions-dir not found: {args.predictions_dir}")
    if not args.reference_dir.exists():
        raise SystemExit(f"FATAL: reference-dir not found: {args.reference_dir}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    valid_metrics = {"tedn", "linearized_ser", "onset_f1"}
    unknown = set(metrics) - valid_metrics
    if unknown:
        raise SystemExit(f"FATAL: unknown metrics: {unknown}. Valid: {valid_metrics}")

    # Resolve venv Python once at startup
    venv_python = _resolve_venv_python(args.python_path)

    cheap_requested = [m for m in metrics if m in CHEAP_METRICS]
    tedn_requested = [m for m in metrics if m == "tedn"]
    splitting = bool(cheap_requested) and bool(tedn_requested)

    # Determine output paths
    output_dir = args.output_dir if args.output_dir else (_REPO_ROOT / "eval" / "results")
    csv_path = (output_dir / f"clarity_demo_{args.name}.csv").resolve()
    partial_csv_path = (output_dir / f"clarity_demo_{args.name}.partial.csv").resolve()

    n_total = len(DEMO_STEMS)

    # Resume: load already-scored stems
    rows_by_index: "dict[int, tuple]" = {}
    resume_stems: set[str] = set()
    if args.resume:
        resume_source = partial_csv_path if partial_csv_path.exists() else csv_path
        resume_stems = _load_resume_set(resume_source)
        if resume_stems:
            print(f"Resumed {len(resume_stems)} pieces from previous run ({resume_source})")
            resumed_rows: dict[str, tuple] = {}
            try:
                with resume_source.open(newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader)  # skip header
                    for row in reader:
                        if row:
                            resumed_rows[row[0]] = tuple(row)
            except Exception:
                pass
            for idx, stem in enumerate(DEMO_STEMS, 1):
                if stem in resumed_rows:
                    rows_by_index[idx] = resumed_rows[stem]

    print(f"Run name:        {args.name}")
    print(f"Predictions dir: {args.predictions_dir}")
    print(f"Reference dir:   {args.reference_dir}")
    print(f"Metrics:         {metrics}")
    print(f"Pieces:          {n_total}")
    print(f"Jobs:            {args.jobs}")
    print(f"Python:          {venv_python}")
    if splitting:
        print(f"Scoring mode:    split (cheap={cheap_requested} @{CHEAP_TIMEOUT_SEC}s,"
              f" tedn @{TEDN_TIMEOUT_SEC}s)")
    else:
        timeout = CHEAP_TIMEOUT_SEC if not tedn_requested else TEDN_TIMEOUT_SEC
        print(f"Scoring mode:    single subprocess @{timeout}s")
    if args.parallel_metric_groups:
        print("Metric groups:   concurrent (--parallel-metric-groups ON)")
    print()

    missing_ref_count = 0
    missing_pred_count = 0

    # Pre-pass: assign skip rows for missing files
    tasks = []
    for i, stem in enumerate(DEMO_STEMS, 1):
        if stem in resume_stems:
            continue

        pred = args.predictions_dir / f"{stem}.musicxml"
        ref = args.reference_dir / f"{stem}.mxl"

        if not pred.exists():
            print(f"[{i}/{n_total}] SKIP {stem}: predicted XML not found at {pred}")
            rows_by_index[i] = (stem, None, None, None) + (None,) * 8 + ("predicted_xml_missing",)
            missing_pred_count += 1
            continue
        if not ref.exists():
            print(f"[{i}/{n_total}] SKIP {stem}: reference MXL not found at {ref}")
            rows_by_index[i] = (stem, None, None, None) + (None,) * 8 + ("reference_mxl_missing",)
            missing_ref_count += 1
            continue

        tasks.append((i, stem, pred, ref))

    # Score tasks with bounded ThreadPoolExecutor
    if tasks:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    _score_task,
                    i, n_total, stem, pred, ref, metrics, venv_python, args.parallel_metric_groups
                ): i
                for i, stem, pred, ref in tasks
            }
            for fut in as_completed(futures):
                i, row, msg = fut.result()
                rows_by_index[i] = row
                print(msg)

                # Checkpoint: write partial CSV after each completion
                _write_csv(partial_csv_path, rows_by_index, n_total)

    # Write final CSV
    _write_csv(csv_path, rows_by_index, n_total)
    print(f"\nResults written to {csv_path}")

    # Clean up partial file
    try:
        if partial_csv_path.exists():
            partial_csv_path.unlink()
    except Exception:
        pass

    _print_summary(rows_by_index, metrics, n_total, missing_ref_count, missing_pred_count, args.name)


if __name__ == "__main__":
    main()
