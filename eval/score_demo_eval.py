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

Two-lane scheduler:
  --cheap-jobs N    ThreadPoolExecutor max workers for cheap metrics (default 8)
  --tedn-jobs N     ThreadPoolExecutor max workers for TEDN (default 4)
  --max-active-pieces N  semaphore-bounded concurrent pieces total (default 8)
  --jobs N          Backward-compat alias: sets cheap_jobs=N, tedn_jobs=max(1,N//2),
                    max_active_pieces=N. Prints deprecation warning.

Checkpointing: .partial.csv is the live write target; renamed to the canonical
name on clean exit.

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
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
    score_metric_group_subprocess,
    score_piece_subprocess,
)

# Backward-compat alias
SCORE_TIMEOUT_SEC = TEDN_TIMEOUT_SEC

# CSV header extended with 'index' column for completion-order output
CSV_HEADER_WITH_INDEX = ["index"] + CSV_HEADER

# Canonical demo stems -- must match run_clarity_demo_eval.py
DEMO_STEMS = [
    "clair-de-lune-debussy",
    "fugue-no-2-bwv-847-in-c-minor",
    "gnossienne-no-1",
    "prelude-in-d-flat-major-op31-no1-scriabin",
]


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
        cheap_ok = (not need_cheap) or self.cheap_done
        tedn_ok = (not need_tedn) or self.tedn_done
        return cheap_ok and tedn_ok

    def failure_reason(self) -> Optional[str]:
        if self.failure_parts:
            return " | ".join(self.failure_parts)
        return None

    def to_row(self) -> tuple:
        return (
            self.stem,
            self.onset_f1,
            self.tedn,
            self.linearized_ser,
        ) + self.stage_d_cols + (self.failure_reason(),)

    def to_indexed_row(self) -> tuple:
        return (self.index,) + self.to_row()


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
    missing_pred_count: int,
    name: str,
    has_index_col: bool = False,
) -> None:
    """Print per-metric summary statistics."""
    _shift = 1 if has_index_col else 0
    metric_col = {
        "onset_f1": 1 + _shift,
        "tedn": 2 + _shift,
        "linearized_ser": 3 + _shift,
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


def _wait_for_memory_budget(limit_gb: float, poll_sec: float) -> None:
    """Block until system used RAM drops below *limit_gb*."""
    if limit_gb <= 0:
        return
    try:
        import psutil
    except ImportError:
        return
    while psutil.virtual_memory().used / (1024 ** 3) >= limit_gb:
        time.sleep(poll_sec)


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
    # --- New two-lane flags ---
    p.add_argument(
        "--cheap-jobs", type=int, default=None,
        help="ThreadPoolExecutor max workers for cheap metrics. Default 8.",
    )
    p.add_argument(
        "--tedn-jobs", type=int, default=None,
        help="ThreadPoolExecutor max workers for TEDN. Default 4.",
    )
    p.add_argument(
        "--max-active-pieces", type=int, default=None,
        help="Semaphore-bounded maximum concurrent pieces total. Default 8.",
    )
    # --- Legacy --jobs (backward compat) ---
    p.add_argument(
        "--jobs", type=int, default=None,
        help="[DEPRECATED] Legacy flag. Sets --cheap-jobs N --tedn-jobs max(1,N//2) "
             "--max-active-pieces N. Use the new flags instead.",
    )
    p.add_argument(
        "--write-order", choices=["completion", "deterministic"], default="completion",
        help="CSV write order. 'completion' (default): write each row as it completes; "
             "adds 'index' column. 'deterministic': write in original stem order.",
    )
    # --- Phase 2: memory / observability ---
    p.add_argument(
        "--memory-limit-gb", type=float, default=84.0,
        help="Adaptive TEDN throttle: block new TEDN submissions while system used RAM "
             "exceeds this value in GB. Default 84. Set to 0 to disable.",
    )
    p.add_argument(
        "--memory-poll-sec", type=float, default=2.0,
        help="Polling cadence in seconds for --memory-limit-gb. Default 2.",
    )
    p.add_argument(
        "--child-memory-limit-gb", type=float, default=0.0,
        help="Linux-only: set RLIMIT_AS on child subprocesses. 0 (default) = disabled. "
             "CAVEAT: RLIMIT_AS caps virtual address space, not RSS.",
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

    # --- Resolve two-lane concurrency settings ---
    if args.jobs is not None:
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
    need_cheap = bool(cheap_requested)
    need_tedn = bool(tedn_requested)

    # Determine output paths
    output_dir = args.output_dir if args.output_dir else (_REPO_ROOT / "eval" / "results")
    csv_path = (output_dir / f"clarity_demo_{args.name}.csv").resolve()
    partial_csv_path = (output_dir / f"clarity_demo_{args.name}.partial.csv").resolve()

    # Scoring logs directory
    scoring_logs_dir = output_dir / "scoring_logs"
    instrumentation_log = scoring_logs_dir / f"clarity_demo_{args.name}_instrumentation.jsonl"

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

    print(f"Run name:          {args.name}")
    print(f"Predictions dir:   {args.predictions_dir}")
    print(f"Reference dir:     {args.reference_dir}")
    print(f"Metrics:           {metrics}")
    print(f"Pieces:            {n_total}")
    print(f"Cheap jobs:        {cheap_jobs}")
    print(f"TEDN jobs:         {tedn_jobs}")
    print(f"Max active pieces: {max_active_pieces}")
    print(f"Write order:       {args.write_order}")
    print(f"Memory limit:      {args.memory_limit_gb} GB (poll {args.memory_poll_sec}s)")
    print(f"Child mem cap:     {args.child_memory_limit_gb} GB (0=disabled)")
    print(f"Python:            {venv_python}")
    print()

    missing_ref_count = 0
    missing_pred_count = 0

    # --- Streaming CSV setup ---
    output_dir.mkdir(parents=True, exist_ok=True)
    use_index_col = (args.write_order == "completion")
    header = CSV_HEADER_WITH_INDEX if use_index_col else CSV_HEADER

    partial_fh = partial_csv_path.open("w", newline="", encoding="utf-8")
    partial_writer = csv.writer(partial_fh)
    partial_writer.writerow(header)
    partial_fh.flush()

    next_to_write = [1]
    pending_rows: "dict[int, tuple]" = {}
    write_lock = threading.Lock()

    def _write_row_streaming(piece_result: PieceResult) -> None:
        if use_index_col:
            row = piece_result.to_indexed_row()
            with write_lock:
                rows_by_index[piece_result.index] = row
                partial_writer.writerow(row)
                partial_fh.flush()
        else:
            row = piece_result.to_row()
            with write_lock:
                rows_by_index[piece_result.index] = row
                pending_rows[piece_result.index] = row
                while next_to_write[0] in pending_rows:
                    partial_writer.writerow(pending_rows.pop(next_to_write[0]))
                    partial_fh.flush()
                    next_to_write[0] += 1

    def _write_skip_row(i: int, stem: str, reason: str) -> None:
        stage_d = (None,) * 8
        if use_index_col:
            row = (i, stem, None, None, None) + stage_d + (reason,)
        else:
            row = (stem, None, None, None) + stage_d + (reason,)
        with write_lock:
            rows_by_index[i] = row
            if use_index_col:
                partial_writer.writerow(row)
            else:
                pending_rows[i] = row
                while next_to_write[0] in pending_rows:
                    partial_writer.writerow(pending_rows.pop(next_to_write[0]))
                    partial_fh.flush()
                    next_to_write[0] += 1
            partial_fh.flush()

    # --- Pre-pass: handle missing files ---
    tasks = []
    for i, stem in enumerate(DEMO_STEMS, 1):
        if stem in resume_stems:
            if not use_index_col:
                pending_rows[i] = rows_by_index.get(i, ())
                while next_to_write[0] in pending_rows:
                    next_to_write[0] += 1
            continue

        pred = args.predictions_dir / f"{stem}.musicxml"
        ref = args.reference_dir / f"{stem}.mxl"

        if not pred.exists():
            print(f"[{i}/{n_total}] SKIP {stem}: predicted XML not found at {pred}")
            _write_skip_row(i, stem, "predicted_xml_missing")
            missing_pred_count += 1
            continue
        if not ref.exists():
            print(f"[{i}/{n_total}] SKIP {stem}: reference MXL not found at {ref}")
            _write_skip_row(i, stem, "reference_mxl_missing")
            missing_ref_count += 1
            continue

        tasks.append((i, stem, pred, ref))

    # --- Two-lane scheduler ---
    active_semaphore = threading.Semaphore(max_active_pieces)
    print_lock = threading.Lock()
    _completion_lock = threading.Lock()
    _completed_indices: set = set()

    def _on_piece_complete(pr: PieceResult) -> None:
        """Called when all requested groups for a piece are done. Thread-safe."""
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

    def _run_tedn_task(pr: PieceResult, sl: Path) -> None:
        _wait_for_memory_budget(args.memory_limit_gb, args.memory_poll_sec)
        tedn_result = score_metric_group_subprocess(
            pr.pred, pr.ref, ["tedn"], TEDN_TIMEOUT_SEC,
            venv_python=venv_python,
            stderr_log=sl,
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

    def _run_cheap_task(pr: PieceResult) -> None:
        sl = scoring_logs_dir / f"{pr.stem}_cheap.stderr.log" if cheap_requested else None
        cheap_result = score_metric_group_subprocess(
            pr.pred, pr.ref, cheap_requested, CHEAP_TIMEOUT_SEC,
            venv_python=venv_python,
            stderr_log=sl,
            child_memory_limit_gb=args.child_memory_limit_gb,
            instrumentation_log=instrumentation_log,
            stem=pr.stem,
            group_name="cheap",
        )
        if "error" in cheap_result:
            pr.failure_parts.append(f"cheap-pair: {cheap_result['error']}")
        else:
            if "onset_f1" in cheap_result:
                pr.onset_f1 = cheap_result["onset_f1"]
            if "linearized_ser" in cheap_result:
                pr.linearized_ser = cheap_result["linearized_ser"]
        pr.cheap_done = True
        if pr.is_complete(need_cheap, need_tedn):
            _on_piece_complete(pr)

    if tasks:
        with (
            ThreadPoolExecutor(max_workers=cheap_jobs) as cheap_pool,
            ThreadPoolExecutor(max_workers=tedn_jobs) as tedn_pool,
        ):
            all_futures = []

            for i, stem, pred, ref in tasks:
                active_semaphore.acquire()

                stage_d_cols = _read_stage_d_diag(pred)
                pr = PieceResult(
                    index=i, stem=stem, pred=pred, ref=ref,
                    stage_d_cols=stage_d_cols,
                )

                if need_tedn:
                    tedn_sl = scoring_logs_dir / f"{stem}_tedn.stderr.log"
                    tedn_fut = tedn_pool.submit(_run_tedn_task, pr, tedn_sl)
                    all_futures.append(tedn_fut)

                if need_cheap:
                    cheap_fut = cheap_pool.submit(_run_cheap_task, pr)
                    all_futures.append(cheap_fut)
                elif not need_cheap and need_tedn:
                    pr.cheap_done = True

                if not need_cheap and not need_tedn:
                    pr.cheap_done = True
                    pr.tedn_done = True
                    _on_piece_complete(pr)

            for fut in all_futures:
                try:
                    fut.result()
                except Exception as exc:
                    print(f"[warn] Worker thread raised: {exc}", file=sys.stderr)

    partial_fh.flush()
    partial_fh.close()

    # Rename partial to canonical
    try:
        partial_csv_path.replace(csv_path)
    except Exception:
        import shutil
        shutil.copy2(str(partial_csv_path), str(csv_path))
        try:
            partial_csv_path.unlink()
        except Exception:
            pass

    print(f"\nResults written to {csv_path}")

    _print_summary(
        rows_by_index, metrics, n_total, missing_ref_count, missing_pred_count, args.name,
        has_index_col=use_index_col,
    )


if __name__ == "__main__":
    main()
