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

Two-lane scheduler (96 GB review, 2026-04-28):
  - Cheap metrics lane (onset_f1, linearized_ser): high concurrency via
    --cheap-jobs (default 8).
  - TEDN lane: lower, separately controlled concurrency via --tedn-jobs (default 4).
  - --max-active-pieces N: semaphore-bounded total active pieces (default 8).
  - --memory-limit-gb: adaptive throttle; blocks new TEDN submissions while
    system used RAM exceeds the limit (default 84 GB).

CSV output: streaming, one row written per completed piece. Two modes:
  - --write-order completion (default): rows written in completion order; an
    'index' column is prepended for downstream sorting.
  - --write-order deterministic: rows buffered and written in original
    prediction-index order.

Parallelism:
  --cheap-jobs N    ThreadPoolExecutor max workers for cheap metrics (default 8)
  --tedn-jobs N     ThreadPoolExecutor max workers for TEDN (default 4)
  --max-active-pieces N  semaphore-bounded concurrent pieces total (default 8)
  --jobs N          Backward-compat alias: sets cheap_jobs=N, tedn_jobs=max(1,N//2),
                    max_active_pieces=N. Prints deprecation warning.

Checkpointing: .partial.csv is the live write target; renamed to the canonical
name on clean exit.

Resume: --resume reads an existing partial or full CSV and skips already-scored
pieces (those with a non-empty row already recorded).

Usage:
    venv\\Scripts\\python -m eval.score_lieder_eval \\
        --predictions-dir eval/results/lieder_mvp \\
        --reference-dir data/openscore_lieder/scores \\
        --name mvp \\
        --cheap-jobs 8 --tedn-jobs 4

Output: eval/results/lieder_<name>.csv  (or --output-dir override)
"""
import argparse
import csv
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Optional psutil import — used for adaptive memory throttling.
# Import at module level so tests can patch it; fall back gracefully if absent.
try:
    import psutil as psutil  # noqa: PLC0414
except ImportError:
    psutil = None  # type: ignore[assignment]

from eval._scoring_utils import (
    CSV_HEADER,
    CHEAP_METRICS,
    CHEAP_TIMEOUT_SEC,
    TEDN_TIMEOUT_SEC,
    _build_reference_index,
    _read_stage_d_diag,
    _resolve_venv_python,
    score_metric_group_subprocess,
    score_piece_subprocess,
)

# CSV header extended with 'index' column for completion-order output
CSV_HEADER_WITH_INDEX = ["index"] + CSV_HEADER


@dataclass
class PieceResult:
    """Parent-side state machine for one piece across both metric lanes."""
    index: int
    stem: str
    pred: Path
    ref: Optional[Path]
    stage_d_cols: tuple
    onset_f1: Optional[float] = None
    tedn: Optional[float] = None
    linearized_ser: Optional[float] = None
    failure_parts: list = field(default_factory=list)
    cheap_done: bool = False
    tedn_done: bool = False
    written: bool = False

    def is_complete(self, need_cheap: bool, need_tedn: bool) -> bool:
        """Return True when all requested groups have completed."""
        cheap_ok = (not need_cheap) or self.cheap_done
        tedn_ok = (not need_tedn) or self.tedn_done
        return cheap_ok and tedn_ok

    def failure_reason(self) -> Optional[str]:
        if self.failure_parts:
            return " | ".join(self.failure_parts)
        return None

    def to_row(self) -> tuple:
        """Build the CSV row tuple (without 'index' prefix)."""
        return (
            self.stem,
            self.onset_f1,
            self.tedn,
            self.linearized_ser,
        ) + self.stage_d_cols + (self.failure_reason(),)

    def to_indexed_row(self) -> tuple:
        """Build the CSV row tuple with 'index' as first column."""
        return (self.index,) + self.to_row()


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
                # Support both normal and index-prefixed CSVs
                piece = row.get("piece", "").strip()
                if piece:
                    stems.add(piece)
    except Exception:
        pass
    return stems


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


def _print_summary(
    rows_by_index: "dict[int, tuple]",
    metrics: list[str],
    n_total: int,
    missing_ref_count: int,
    name: str,
    has_index_col: bool = False,
) -> None:
    """Print per-metric summary statistics.

    Uses the first requested metric in *metrics* as the headline for mean/median.
    Prints per-metric scored counts so metric-subset runs are clearly reported.

    Args:
        has_index_col: When True, rows have an extra 'index' column prepended,
            shifting all metric columns by 1.
    """
    # Build per-metric value lists using CSV_HEADER column positions
    # (with optional index-col shift)
    _shift = 1 if has_index_col else 0
    metric_col = {
        "onset_f1": 1 + _shift,
        "tedn": 2 + _shift,
        "linearized_ser": 3 + _shift,
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


def _wait_for_memory_budget(limit_gb: float, poll_sec: float) -> None:
    """Block until system used RAM drops below *limit_gb*.

    Uses psutil.virtual_memory().used. Polls every *poll_sec* seconds.
    Does nothing when limit_gb <= 0 or psutil is unavailable.
    """
    if limit_gb <= 0:
        return
    if psutil is None:
        return  # psutil not available — skip throttle

    while psutil.virtual_memory().used / (1024 ** 3) >= limit_gb:
        time.sleep(poll_sec)


def _build_argparser() -> argparse.ArgumentParser:
    """Build and return the argument parser for score_lieder_eval.

    Extracted so tests can call _build_argparser() directly without invoking main().
    """
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
        "--reference-dir", "--ground-truth-dir", type=Path, required=True,
        dest="reference_dir",
        help="Directory containing reference .mxl files. Can be the full openscore_lieder/scores "
             "tree (recursive search used) or a flat mirror. "
             "Example: data/openscore_lieder/scores. "
             "(--ground-truth-dir is an alias used by invoke_scoring_phase.)",
    )
    p.add_argument(
        "--name", default=None,
        help="Run name (used for output CSV filename: eval/results/lieder_<name>.csv). "
             "Should match the name used with run_lieder_eval.py. "
             "Required unless --out-csv is given.",
    )
    p.add_argument(
        "--out-csv", type=Path, default=None, dest="out_csv",
        help="Full path to the output CSV file. Overrides --name / --output-dir. "
             "Alias used by invoke_scoring_phase in run_lieder_eval.py.",
    )
    p.add_argument(
        "--metrics",
        default="linearized_ser,onset_f1",
        help="Comma-separated list of metrics to compute (default: linearized_ser,onset_f1). "
             "Use --tedn to also enable TEDN (slow, ~300s/piece worst case).",
    )
    p.add_argument(
        "--tedn",
        action="store_true",
        default=False,
        help="Compute TEDN (Tree-Edit Distance on kern) — slow, ~300s/piece worst case. "
             "Default off. When set, 'tedn' is appended to the metrics list.",
    )
    p.add_argument(
        "--max-pieces", type=int, default=None,
        help="Score only the first N predictions (for validating a partial inference run). "
             "Default None scores all discovered predictions.",
    )
    # --- New two-lane flags ---
    p.add_argument(
        "--cheap-jobs", type=int, default=None,
        help="ThreadPoolExecutor max workers for cheap metrics (onset_f1, linearized_ser). "
             "Default 8. Mutually overridden by --jobs (legacy).",
    )
    p.add_argument(
        "--tedn-jobs", type=int, default=None,
        help="ThreadPoolExecutor max workers for TEDN. Default 4. "
             "Mutually overridden by --jobs (legacy).",
    )
    p.add_argument(
        "--max-active-pieces", type=int, default=None,
        help="Semaphore-bounded maximum concurrent pieces total. Default 8. "
             "Prevents one slow TEDN from leaving cheap workers idle while "
             "also limiting total active pieces.",
    )
    # --- Legacy --jobs (backward compat) ---
    p.add_argument(
        "--jobs", type=int, default=None,
        help="[DEPRECATED] Legacy flag. Sets --cheap-jobs N --tedn-jobs max(1,N//2) "
             "--max-active-pieces N. Use the new flags instead.",
    )
    # --- Write-order ---
    p.add_argument(
        "--write-order", choices=["completion", "deterministic"], default="completion",
        help="CSV write order. 'completion' (default): write each row as it completes; "
             "adds 'index' column for downstream sorting. "
             "'deterministic': buffer completed rows and write in original prediction order.",
    )
    # --- Phase 2: memory / observability ---
    p.add_argument(
        "--memory-limit-gb", type=float, default=84.0,
        help="Adaptive TEDN throttle: block new TEDN submissions while system used RAM "
             "exceeds this value in GB. Default 84 (leaves ~12 GB headroom on 96 GB host). "
             "Set to 0 to disable.",
    )
    p.add_argument(
        "--memory-poll-sec", type=float, default=2.0,
        help="Polling cadence in seconds for --memory-limit-gb. Default 2.",
    )
    p.add_argument(
        "--child-memory-limit-gb", type=float, default=0.0,
        help="Linux-only: set RLIMIT_AS on child subprocesses. 0 (default) = disabled. "
             "CAVEAT: RLIMIT_AS caps virtual address space, not RSS. Some Python/scientific "
             "libraries reserve more virtual memory than they actively use; raise to 24 GB "
             "or disable if this causes false failures. No-op on Windows.",
    )
    # --- Standard flags ---
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
    return p


def main() -> None:
    p = _build_argparser()
    args = p.parse_args()

    # --tedn appends 'tedn' to the metrics list
    if args.tedn and "tedn" not in args.metrics:
        args.metrics = args.metrics + ",tedn"

    # --- Resolve two-lane concurrency settings ---
    if args.jobs is not None:
        # Legacy --jobs: map to new flags with deprecation warning
        n = args.jobs
        if n < 1:
            raise SystemExit(f"FATAL: --jobs must be >= 1, got {n}")
        cheap_jobs = n
        tedn_jobs = max(1, n // 2)
        max_active_pieces = n
        print(
            f"[DEPRECATED] --jobs {n} is deprecated. Equivalent to: "
            f"--cheap-jobs {cheap_jobs} --tedn-jobs {tedn_jobs} "
            f"--max-active-pieces {max_active_pieces}. "
            "Use the new flags for independent control.",
            file=sys.stderr,
        )
    else:
        cheap_jobs = args.cheap_jobs if args.cheap_jobs is not None else 8
        tedn_jobs = args.tedn_jobs if args.tedn_jobs is not None else 4
        max_active_pieces = args.max_active_pieces if args.max_active_pieces is not None else 8

    if cheap_jobs < 1:
        raise SystemExit(f"FATAL: --cheap-jobs must be >= 1, got {cheap_jobs}")
    if tedn_jobs < 1:
        raise SystemExit(f"FATAL: --tedn-jobs must be >= 1, got {tedn_jobs}")
    if max_active_pieces < 1:
        raise SystemExit(f"FATAL: --max-active-pieces must be >= 1, got {max_active_pieces}")

    # Validate: need either --name or --out-csv (not both required, but at least one)
    if args.out_csv is None and args.name is None:
        raise SystemExit("FATAL: one of --name or --out-csv is required.")

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
    need_cheap = bool(cheap_requested)
    need_tedn = bool(tedn_requested)

    # Determine output paths
    # --out-csv takes precedence; derive name/output_dir from it if provided.
    if args.out_csv is not None:
        csv_path = args.out_csv.resolve()
        output_dir = csv_path.parent
        # Derive run name from the CSV filename (strip leading "lieder_" if present, drop .csv)
        _stem = csv_path.stem
        run_name = _stem[len("lieder_"):] if _stem.startswith("lieder_") else _stem
    else:
        output_dir = args.output_dir if args.output_dir else (_REPO_ROOT / "eval" / "results")
        run_name = args.name
        csv_path = (output_dir / f"lieder_{run_name}.csv").resolve()

    partial_csv_path = csv_path.with_suffix(".partial.csv")

    # Scoring logs directory (stderr + instrumentation)
    scoring_logs_dir = output_dir / "scoring_logs"
    instrumentation_log = scoring_logs_dir / f"lieder_{run_name}_instrumentation.jsonl"

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

    print(f"Run name:          {run_name}")
    print(f"Predictions dir:   {args.predictions_dir}")
    print(f"Reference dir:     {args.reference_dir}")
    print(f"Metrics:           {metrics}")
    print(f"Pieces:            {len(preds)}")
    print(f"Cheap jobs:        {cheap_jobs}")
    print(f"TEDN jobs:         {tedn_jobs}")
    print(f"Max active pieces: {max_active_pieces}")
    print(f"Write order:       {args.write_order}")
    print(f"Memory limit:      {args.memory_limit_gb} GB (poll {args.memory_poll_sec}s)")
    print(f"Child mem cap:     {args.child_memory_limit_gb} GB (0=disabled)")
    print(f"Python:            {venv_python}")
    print(f"Scoring logs:      {scoring_logs_dir}")
    print()

    n_total = len(preds)
    missing_ref_count = 0

    # --- Streaming CSV setup ---
    output_dir.mkdir(parents=True, exist_ok=True)
    use_index_col = (args.write_order == "completion")
    header = CSV_HEADER_WITH_INDEX if use_index_col else CSV_HEADER

    partial_fh = partial_csv_path.open("w", newline="", encoding="utf-8")
    partial_writer = csv.writer(partial_fh)
    partial_writer.writerow(header)
    partial_fh.flush()

    # For deterministic write-order buffering
    next_to_write = [1]  # list so closure can mutate
    pending_rows: "dict[int, tuple]" = {}
    write_lock = threading.Lock()

    def _write_row_streaming(piece_result: PieceResult) -> None:
        """Write a completed row to the partial CSV, flushing after every row."""
        nonlocal pending_rows
        if use_index_col:
            row = piece_result.to_indexed_row()
            with write_lock:
                rows_by_index[piece_result.index] = row
                partial_writer.writerow(row)
                partial_fh.flush()
        else:
            # Deterministic order: buffer and flush when the sequence is contiguous
            row = piece_result.to_row()
            with write_lock:
                rows_by_index[piece_result.index] = row
                pending_rows[piece_result.index] = row
                while next_to_write[0] in pending_rows:
                    partial_writer.writerow(pending_rows.pop(next_to_write[0]))
                    partial_fh.flush()
                    next_to_write[0] += 1

    def _write_missing_row(i: int, stem: str, reason: str) -> None:
        """Write a missing-reference row immediately."""
        stage_d = (None,) * 9
        if use_index_col:
            row = (i, stem, None, None, None) + stage_d + (reason,)
        else:
            row = (stem, None, None, None) + stage_d + (reason,)
        with write_lock:
            rows_by_index[i] = row
            if use_index_col:
                partial_writer.writerow(row)
            else:
                # Still need to handle ordering for deterministic mode
                pending_rows[i] = row
                while next_to_write[0] in pending_rows:
                    partial_writer.writerow(pending_rows.pop(next_to_write[0]))
                    partial_fh.flush()
                    next_to_write[0] += 1
            partial_fh.flush()

    # --- Pre-pass: handle missing references immediately ---
    tasks = []
    for i, pred in enumerate(preds, 1):
        stem = pred.stem
        if stem in resume_stems:
            # Restore from resume into rows_by_index (already done above)
            # Also advance next_to_write for deterministic mode
            if not use_index_col:
                pending_rows[i] = rows_by_index.get(i, ())
                while next_to_write[0] in pending_rows:
                    r = pending_rows.pop(next_to_write[0])
                    if r:  # already written at resume time; just advance counter
                        pass
                    next_to_write[0] += 1
            continue
        ref = ref_index.get(stem)
        if ref is None:
            # Check flat path as well (pre-flattened eval_mxl mirror)
            flat = args.reference_dir / f"{stem}.mxl"
            if flat.exists():
                ref = flat
        if ref is None:
            print(f"[{i}/{n_total}] SKIP {stem}: reference MXL not found under {args.reference_dir}")
            _write_missing_row(i, stem, "reference_mxl_missing")
            missing_ref_count += 1
        else:
            tasks.append((i, pred, ref))

    # --- Two-lane scheduler ---
    active_semaphore = threading.Semaphore(max_active_pieces)
    print_lock = threading.Lock()

    # All PieceResult objects, keyed by index
    piece_results: "dict[int, PieceResult]" = {}
    piece_results_lock = threading.Lock()

    # Futures for TEDN tasks keyed by piece index
    tedn_futures: "dict[int, Future]" = {}
    tedn_futures_lock = threading.Lock()

    # Per-piece completion lock: prevents double-completion in the race where
    # cheap finishes after TEDN and both observe is_complete() == True.
    _completion_lock = threading.Lock()
    _completed_indices: set = set()

    def _on_piece_complete(pr: PieceResult) -> None:
        """Called when all requested groups for a piece are done. Thread-safe.

        Guards against double-completion: only the first caller for a given
        piece index proceeds.
        """
        with _completion_lock:
            if pr.index in _completed_indices:
                return
            _completed_indices.add(pr.index)
        _write_row_streaming(pr)
        with print_lock:
            print(_format_progress(
                pr.index, n_total, pr.stem,
                pr.onset_f1, pr.tedn, pr.linearized_ser,
                pr.failure_reason(),
            ))
        active_semaphore.release()

    def _cheap_done_callback(
        pr: PieceResult,
        cheap_result: dict,
        tedn_pool: ThreadPoolExecutor,
    ) -> None:
        """Update PieceResult with cheap results; submit TEDN if needed."""
        if "error" in cheap_result:
            pr.failure_parts.append(f"cheap-pair: {cheap_result['error']}")
        else:
            if "onset_f1" in cheap_result:
                pr.onset_f1 = cheap_result["onset_f1"]
            if "linearized_ser" in cheap_result:
                pr.linearized_ser = cheap_result["linearized_ser"]

        pr.cheap_done = True

        if pr.is_complete(need_cheap, need_tedn):
            # Either no TEDN was requested, or TEDN already finished before cheap.
            _on_piece_complete(pr)

    def _submit_tedn(
        pr: PieceResult,
        tedn_pool: ThreadPoolExecutor,
        stderr_dir: Path,
    ) -> None:
        """Wait for memory budget, then submit TEDN to the pool."""
        _wait_for_memory_budget(args.memory_limit_gb, args.memory_poll_sec)
        sl = stderr_dir / f"{pr.stem}_tedn.stderr.log"
        fut = tedn_pool.submit(
            _run_tedn_task,
            pr,
            tedn_pool,
            sl,
            stderr_dir,
        )
        with tedn_futures_lock:
            tedn_futures[pr.index] = fut

    def _run_tedn_task(
        pr: PieceResult,
        tedn_pool: ThreadPoolExecutor,
        stderr_log: Path,
        stderr_dir: Path,
    ) -> None:
        """Execute TEDN for one piece and update state."""
        tedn_result = score_metric_group_subprocess(
            pr.pred,
            pr.ref,
            ["tedn"],
            TEDN_TIMEOUT_SEC,
            venv_python=venv_python,
            stderr_log=stderr_log,
            child_memory_limit_gb=args.child_memory_limit_gb,
            instrumentation_log=instrumentation_log,
            stem=pr.stem,
            group_name="tedn",
        )
        if "error" in tedn_result:
            pr.failure_parts.append(f"tedn: {tedn_result['error']}")
        else:
            pr.tedn = tedn_result.get("tedn")
        pr.tedn_done = True

        if pr.is_complete(need_cheap, need_tedn):
            _on_piece_complete(pr)

    def _run_cheap_task(
        pr: PieceResult,
        tedn_pool: ThreadPoolExecutor,
        stderr_dir: Path,
    ) -> None:
        """Execute cheap metrics for one piece; submit TEDN if needed."""
        sl = stderr_dir / f"{pr.stem}_cheap.stderr.log" if cheap_requested else None
        cheap_result = score_metric_group_subprocess(
            pr.pred,
            pr.ref,
            cheap_requested,
            CHEAP_TIMEOUT_SEC,
            venv_python=venv_python,
            stderr_log=sl,
            child_memory_limit_gb=args.child_memory_limit_gb,
            instrumentation_log=instrumentation_log,
            stem=pr.stem,
            group_name="cheap",
        )
        _cheap_done_callback(pr, cheap_result, tedn_pool)

        # If TEDN was not yet submitted (it was submitted concurrently with cheap),
        # nothing to do here — TEDN callback handles completion.
        # If no TEDN is needed, _cheap_done_callback already called _on_piece_complete.

    if tasks:
        with (
            ThreadPoolExecutor(max_workers=cheap_jobs) as cheap_pool,
            ThreadPoolExecutor(max_workers=tedn_jobs) as tedn_pool,
        ):
            all_futures = []

            for i, pred, ref in tasks:
                # Acquire semaphore before submitting any work for this piece
                active_semaphore.acquire()

                # Build PieceResult
                stage_d_cols = _read_stage_d_diag(pred)
                pr = PieceResult(
                    index=i,
                    stem=pred.stem,
                    pred=pred,
                    ref=ref,
                    stage_d_cols=stage_d_cols,
                )
                with piece_results_lock:
                    piece_results[i] = pr

                if need_tedn:
                    # Submit TEDN immediately (with memory throttle applied inside)
                    # The memory wait happens in the TEDN thread, not here.
                    # We submit to the TEDN pool; the pool bounds concurrent TEDN subprocesses.
                    # The per-piece memory throttle gates submission of the actual subprocess.
                    # Strategy: submit cheap + tedn concurrently (both pools independent).
                    tedn_sl = scoring_logs_dir / f"{pred.stem}_tedn.stderr.log"

                    def _tedn_with_throttle(pr=pr, sl=tedn_sl):
                        _wait_for_memory_budget(args.memory_limit_gb, args.memory_poll_sec)
                        _run_tedn_task(pr, tedn_pool, sl, scoring_logs_dir)

                    tedn_fut = tedn_pool.submit(_tedn_with_throttle)
                    all_futures.append(tedn_fut)

                if need_cheap:
                    cheap_fut = cheap_pool.submit(
                        _run_cheap_task, pr, tedn_pool, scoring_logs_dir
                    )
                    all_futures.append(cheap_fut)
                elif not need_cheap and need_tedn:
                    # TEDN-only: cheap_done is vacuously true, just need tedn
                    pr.cheap_done = True
                    # TEDN already submitted above

                if not need_cheap and not need_tedn:
                    # No metrics at all — write immediately
                    pr.cheap_done = True
                    pr.tedn_done = True
                    _on_piece_complete(pr)

            # Wait for all submitted futures
            for fut in all_futures:
                try:
                    fut.result()
                except Exception as exc:
                    # Unexpected error in worker thread — log but don't crash the run
                    print(f"[warn] Worker thread raised: {exc}", file=sys.stderr)

    partial_fh.flush()
    partial_fh.close()

    # Write final CSV (rename partial to canonical)
    try:
        partial_csv_path.replace(csv_path)
    except Exception:
        # Fallback: copy then remove
        import shutil
        shutil.copy2(str(partial_csv_path), str(csv_path))
        try:
            partial_csv_path.unlink()
        except Exception:
            pass

    print(f"\nResults written to {csv_path}")

    _print_summary(
        rows_by_index, metrics, n_total, missing_ref_count, run_name,
        has_index_col=use_index_col,
    )


if __name__ == "__main__":
    main()
