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
via a single index built at startup (replaces per-piece rglob calls) — point this
at the openscore_lieder scores directory (data/openscore_lieder/scores or the
eval_mxl mirror). Duplicate stems in the reference directory raise a hard error.

Parallelism: bounded piece-level parallelism via --jobs N (default 1, serial).
Use --jobs 2 or --jobs 3 first; increase only after watching peak RAM.
Memory budget: peak_ram ~= parent_ram + jobs * per_piece_peak_ram.
At ~2.5 GB per piece: --jobs 2 ~= 10 GB, --jobs 4 ~= 15 GB.

Checkpointing: writes eval/results/<name>.partial.csv after each completed
piece so that a long run is resumable on interruption.

Resume: --resume reads an existing partial or full CSV and skips already-scored
pieces (those with a non-empty row already recorded).

Usage:
    venv\\Scripts\\python -m eval.score_lieder_eval \\
        --predictions-dir eval/results/lieder_mvp \\
        --reference-dir data/openscore_lieder/scores \\
        --name mvp

Output: eval/results/lieder_<name>.csv  (or --output-dir override)
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
    _build_reference_index,
    _read_stage_d_diag,
    _resolve_venv_python,
    score_piece_subprocess,
)


def _discover_predictions(predictions_dir: Path) -> list[Path]:
    """Return sorted list of .musicxml prediction files in *predictions_dir*."""
    return sorted(predictions_dir.glob("*.musicxml"))


def _load_resume_set(csv_path: Path) -> set[str]:
    """Return the set of stems that already have a complete row in *csv_path*.

    A row is considered complete if the piece column is non-empty. Used by
    --resume to skip already-scored pieces.
    """
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
    """Write rows in original prediction order to *csv_path*."""
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
    stem = pred.stem
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
    name: str,
) -> None:
    """Print per-metric summary statistics.

    Uses the first requested metric in *metrics* as the headline for mean/median.
    Prints per-metric scored counts so metric-subset runs are clearly reported.
    """
    # Build per-metric value lists using CSV_HEADER column positions
    metric_col = {
        "onset_f1": 1,
        "tedn": 2,
        "linearized_ser": 3,
    }
    # Rows that are in rows_by_index (excludes never-started resumed pieces)
    all_rows = [rows_by_index[i] for i in sorted(rows_by_index)]

    # Distinguish missing references from scoring failures.
    # row[-1] may be None (fresh row) or "" / "None" (string from resumed CSV).
    _skip_reasons = {"reference_mxl_missing", None, "", "None"}
    scoring_failures = sum(
        1 for row in all_rows
        if row[-1] not in _skip_reasons
    )
    pieces_evaluated = len(all_rows) - missing_ref_count

    print(f"\n=== Lieder Scoring Results ({name}) ===")
    print(f"Pieces evaluated: {pieces_evaluated} / {n_total}")
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
    p.add_argument(
        "--jobs", type=int, default=1,
        help="Number of pieces to score concurrently via ThreadPoolExecutor (default: 1 = serial). "
             "Use --jobs 2 or --jobs 3 first; increase only after watching peak RAM. "
             "Memory budget: peak_ram ~= parent_ram + jobs * per_piece_peak_ram (~2.5 GB each).",
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
        help="Read an existing output or partial CSV and skip already-scored pieces. "
             "Prints how many pieces are resumed.",
    )
    p.add_argument(
        "--parallel-metric-groups", action="store_true", default=False,
        help="(Advanced) Run cheap and tedn metric subprocesses concurrently within each piece. "
             "Can reduce per-piece wall time but roughly doubles per-piece peak RAM. "
             "Default OFF. See score_piece_subprocess() docstring for details.",
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

    # Build reference index ONCE — avoids N rglob calls; fails fast on duplicate stems.
    print(f"Building reference index from {args.reference_dir} ...")
    ref_index = _build_reference_index(args.reference_dir)
    print(f"  {len(ref_index)} reference stems indexed.")

    cheap_requested = [m for m in metrics if m in CHEAP_METRICS]
    tedn_requested = [m for m in metrics if m == "tedn"]
    splitting = bool(cheap_requested) and bool(tedn_requested)

    # Determine output paths
    output_dir = args.output_dir if args.output_dir else (_REPO_ROOT / "eval" / "results")
    csv_path = (output_dir / f"lieder_{args.name}.csv").resolve()
    partial_csv_path = (output_dir / f"lieder_{args.name}.partial.csv").resolve()

    # Resume: load already-scored stems and pre-populate rows_by_index
    rows_by_index: "dict[int, tuple]" = {}
    resume_stems: set[str] = set()
    if args.resume:
        # Prefer the partial CSV (more up-to-date during an interrupted run)
        resume_source = partial_csv_path if partial_csv_path.exists() else csv_path
        resume_stems = _load_resume_set(resume_source)
        if resume_stems:
            print(f"Resumed {len(resume_stems)} pieces from previous run ({resume_source})")
            # Pre-populate rows_by_index with resumed rows (in original pred order)
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
            for idx, pred in enumerate(preds, 1):
                stem = pred.stem
                if stem in resumed_rows:
                    rows_by_index[idx] = resumed_rows[stem]

    print(f"Run name:        {args.name}")
    print(f"Predictions dir: {args.predictions_dir}")
    print(f"Reference dir:   {args.reference_dir}")
    print(f"Metrics:         {metrics}")
    print(f"Pieces:          {len(preds)}")
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

    n_total = len(preds)
    missing_ref_count = 0

    # Pre-pass: assign skip rows for missing references (immediate, not submitted to pool)
    tasks = []
    for i, pred in enumerate(preds, 1):
        stem = pred.stem
        if stem in resume_stems:
            continue  # Already in rows_by_index
        ref = ref_index.get(stem)
        if ref is None:
            # Check flat path as well (pre-flattened eval_mxl mirror)
            flat = args.reference_dir / f"{stem}.mxl"
            if flat.exists():
                ref = flat
        if ref is None:
            print(f"[{i}/{n_total}] SKIP {stem}: reference MXL not found under {args.reference_dir}")
            rows_by_index[i] = (stem, None, None, None) + (None,) * 8 + ("reference_mxl_missing",)
            missing_ref_count += 1
        else:
            tasks.append((i, pred, ref))

    # Score tasks with bounded ThreadPoolExecutor
    if tasks:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    _score_task,
                    i, n_total, pred, ref, metrics, venv_python, args.parallel_metric_groups
                ): i
                for i, pred, ref in tasks
            }
            for fut in as_completed(futures):
                i, row, msg = fut.result()
                rows_by_index[i] = row
                print(msg)

                # Checkpoint: write partial CSV after each completion
                _write_csv(partial_csv_path, rows_by_index, n_total)

    # Write final CSV (same content as last partial)
    _write_csv(csv_path, rows_by_index, n_total)
    print(f"\nResults written to {csv_path}")

    # Clean up partial file if the final write succeeded
    try:
        if partial_csv_path.exists():
            partial_csv_path.unlink()
    except Exception:
        pass

    _print_summary(rows_by_index, metrics, n_total, missing_ref_count, args.name)


if __name__ == "__main__":
    main()
