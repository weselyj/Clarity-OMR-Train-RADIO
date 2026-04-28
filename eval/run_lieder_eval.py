"""Run a trained Clarity-OMR-Train-RADIO checkpoint through the Lieder eval split.

**Inference-only pass.** This script does NOT compute metrics. Its only job is
to run the RADIO/YOLO inference pipeline for each piece and write the predicted
MusicXML (plus Stage-D diagnostics sidecar) to disk. Metric scoring is handled
by the separate eval/score_lieder_eval.py, which subprocess-isolates per-piece
metric computation so music21/zss memory is fully reclaimed after each piece.

This two-pass design was adopted after a 20-piece lieder run hit 43 GB committed
memory at piece 6/20 and had to be killed before pagefile exhaustion. Same root
cause as the demo eval OOM (PR #26): in-process metric scoring accumulates
music21/zss state across pieces. See eval/score_lieder_eval.py and PR #26 for
the motivation and memory profile.

Wraps src.pdf_to_musicxml (full Stage A YOLO + Stage B RADIO/DaViT + MusicXML
export pipeline already vendored in this repo). Uses sibling Clarity-OMR's
shipped YOLO weights for Stage A; loads our trained Stage B checkpoint via
--stage-b-checkpoint. Our src.eval.evaluate_stage_b_checkpoint detects the
DoRA-wrapped (PEFT base_model.model.* with lora_*) checkpoint format
automatically and applies _prepare_model_for_dora before load_state_dict.

Defaults to greedy decode (beam_width=1, max_decode_steps=256) — fast enough
for the MVP gate where the goal is just "any non-NaN F1 means inference works."
For real eval against a Stage 3 checkpoint, override with --beam-width 5
--max-decode-steps 512.

NOTE: --config is metadata-only. src.pdf_to_musicxml does NOT accept a --config
argument (it uses its own internal config baked into the checkpoint). The config
path is validated at startup (to catch typos early) and written to the per-piece
status JSONL as run metadata, but it is NOT forwarded to the inference subprocess.

Usage:
    python -m eval.run_lieder_eval \\
        --checkpoint checkpoints/mvp_radio_stage2/stage2-radio-mvp_step_0000150.pt \\
        --config configs/train_stage2_radio_mvp.yaml \\
        --name mvp \\
        --max-pieces 20   # optional smoke cap

Multi-GPU example:
    python -m eval.run_lieder_eval \\
        --checkpoint checkpoints/stage3.pt \\
        --config configs/stage3.yaml \\
        --name stage3 \\
        --devices cuda:0,cuda:1,cuda:2,cuda:3 \\
        --jobs 4

Then score:
    python -m eval.score_lieder_eval \\
        --predictions-dir eval/results/lieder_mvp \\
        --reference-dir data/openscore_lieder/scores \\
        --name mvp
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Deque, List, Optional

# Insert repo root so imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval._scoring_utils import _resolve_venv_python
from eval.lieder_split import get_eval_pieces, split_hash

# Default Stage A YOLO checkpoint (shipped by sibling Clarity-OMR).
# Override via --stage-a-weights.
_DEFAULT_STAGE_A_YOLO = Path.home() / "Clarity-OMR" / "info" / "yolo.pt"

# Default inference timeout per piece (seconds). Override via --timeout-sec.
_DEFAULT_TIMEOUT_SEC = 1800

# Cache validation: minimum acceptable output file size (bytes).
# Override via --min-output-bytes.
_DEFAULT_MIN_OUTPUT_BYTES = 1024

# Minimum detectable XML smell — we look for this near the start of the file.
_XML_SMELL = b"<score-partwise"

# Number of recent piece durations used for ETA rolling average.
_ETA_WINDOW = 10


def _is_cache_valid(pred: Path, min_bytes: int) -> tuple[bool, str]:
    """Return (valid, reason) for an existing cached prediction.

    Checks:
      1. File size >= min_bytes.
      2. File contains the '<score-partwise' XML marker (read first 4 KB).
    """
    try:
        size = pred.stat().st_size
    except OSError as exc:
        return False, f"stat failed: {exc}"

    if size < min_bytes:
        return False, f"too small ({size} < {min_bytes} bytes)"

    try:
        header = pred.read_bytes()[:4096]
    except OSError as exc:
        return False, f"read failed: {exc}"

    if _XML_SMELL not in header:
        return False, "missing <score-partwise marker (possibly truncated or corrupt)"

    return True, "ok"


def _write_status_record(jsonl_path: Path, record: dict) -> None:
    """Append one JSON record to the status JSONL file, flushing immediately."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _format_eta(remaining_pieces: int, durations: Deque[float]) -> str:
    """Return a HH:MM:SS ETA string based on rolling average of recent durations."""
    if not durations or remaining_pieces <= 0:
        return "unknown"
    avg = statistics.mean(durations)
    total_sec = int(avg * remaining_pieces)
    hours, rem = divmod(total_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_inference(
    python: Path,
    checkpoint: Path,
    config: Path,  # metadata-only; not forwarded to subprocess
    pdf: Path,
    out: Path,
    work_dir: Path,
    stage_a_yolo: Path,
    *,
    beam_width: int = 1,
    max_decode_steps: int = 256,
    stage_b_device: str = "cuda",
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    stdout_log: Optional[Path] = None,
    stderr_log: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run our trained Stage B + sibling YOLO Stage A on `pdf`, write MusicXML to `out`.

    Spawns src.pdf_to_musicxml in a *child process* so that the RADIO encoder,
    checkpoint weights, and all torch tensors are released when the child exits.
    This is the key isolation that prevents OOM across pieces.

    NOTE: `config` is accepted for symmetry with the parent CLI but is NOT
    forwarded to the subprocess — src.pdf_to_musicxml has no --config flag.
    It is stored in the status manifest as run metadata only.

    Returns the CompletedProcess for the caller to inspect returncode / logs.
    """
    if not stage_a_yolo.exists():
        raise SystemExit(
            f"FATAL: Stage A YOLO weights not found at {stage_a_yolo}. "
            "Run sibling Clarity-OMR's omr.py once on any PDF to trigger the HF download, "
            "or supply --stage-a-weights pointing to an existing yolo.pt."
        )
    repo_root = Path(__file__).resolve().parents[1]
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python),
        "-m", "src.pdf_to_musicxml",
        "--pdf", str(pdf),
        "--output-musicxml", str(out),
        "--weights", str(stage_a_yolo),
        "--stage-b-checkpoint", str(checkpoint),
        "--work-dir", str(work_dir),
        "--project-root", str(repo_root),
        "--stage-b-device", stage_b_device,
        "--beam-width", str(beam_width),
        "--max-decode-steps", str(max_decode_steps),
        "--image-height", "250",
        "--image-max-width", "2500",
        "--length-penalty-alpha", "0.4",
        "--pdf-dpi", "300",
        "--quiet",
    ]

    kwargs: dict = dict(cwd=str(repo_root), timeout=timeout_sec)
    if stdout_log is not None or stderr_log is not None:
        # Capture to files for per-piece log artifacts.
        stdout_log = stdout_log or Path(os.devnull)  # type: ignore[assignment]
        stderr_log = stderr_log or Path(os.devnull)  # type: ignore[assignment]
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        with stdout_log.open("wb") as sout, stderr_log.open("wb") as serr:
            result = subprocess.run(cmd, stdout=sout, stderr=serr, check=False, **kwargs)
    else:
        result = subprocess.run(cmd, check=False, **kwargs)

    return result


def _tail_bytes(path: Optional[Path], max_bytes: int = 2048) -> str:
    """Return the last *max_bytes* of a file as a string, or '' if unavailable."""
    if path is None or not path.exists():
        return ""
    try:
        raw = path.read_bytes()
        return raw[-max_bytes:].decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _run_piece(
    *,
    i: int,
    n_total: int,
    piece,
    pdf: Path,
    pred: Path,
    work_dir: Path,
    args: argparse.Namespace,
    python: Path,
    status_jsonl: Path,
    logs_dir: Path,
    device: str,
    rolling_durations: Deque[float],
    durations_lock,
) -> dict:
    """Process one piece end-to-end. Returns a status record dict.

    This function is the unit of work dispatched by both the serial loop and the
    ThreadPoolExecutor.  It is thread-safe: all shared mutable state is guarded by
    *durations_lock* or is per-piece (pred, work_dir, logs).
    """
    stem = piece.stem

    stdout_log = logs_dir / f"{stem}.stdout.log"
    stderr_log = logs_dir / f"{stem}.stderr.log"

    # --- Cache check -------------------------------------------------------
    if not args.force and pred.exists():
        valid, reason = _is_cache_valid(pred, args.min_output_bytes)
        if valid:
            size = pred.stat().st_size
            diag_path = pred.with_suffix(pred.suffix + ".diagnostics.json")
            diag_ok = diag_path.exists()
            record = dict(
                index=i,
                stem=stem,
                pdf=str(pdf),
                prediction_path=str(pred),
                work_dir=str(work_dir),
                status="cached",
                duration_sec=None,
                output_size_bytes=size,
                checkpoint=str(args.checkpoint),
                config=str(args.config),
                beam_width=args.beam_width,
                max_decode_steps=args.max_decode_steps,
                device=device,
                diagnostics_sidecar_present=diag_ok,
                error=None,
            )
            _write_status_record(status_jsonl, record)
            print(f"[{i}/{n_total}] cached   {stem}  ({size // 1024} KB)"
                  + (" [no sidecar!]" if not diag_ok else ""))
            return record
        else:
            # Cache corrupted — fall through to rerun.
            print(f"[{i}/{n_total}] RERUN    {stem}: corrupted cache ({reason}), rerunning")
            status_prefix = "corrupted_cache"
    else:
        status_prefix = None

    # --- Run inference -----------------------------------------------------
    start = time.monotonic()
    error_msg: Optional[str] = None
    run_status = "unknown"
    result = None

    print(f"[{i}/{n_total}] inference {stem} (device={device}) ...")

    try:
        result = run_inference(
            python=python,
            checkpoint=args.checkpoint,
            config=args.config,
            pdf=pdf,
            out=pred,
            work_dir=work_dir,
            stage_a_yolo=args.stage_a_weights,
            beam_width=args.beam_width,
            max_decode_steps=args.max_decode_steps,
            stage_b_device=device,
            timeout_sec=args.timeout_sec,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        if result.returncode != 0:
            run_status = "failed"
            error_msg = (
                f"exit code {result.returncode}; "
                + _tail_bytes(stderr_log)
            )
        elif pred.exists():
            size = pred.stat().st_size
            if size < args.min_output_bytes:
                run_status = "missing_output"
                error_msg = f"output too small ({size} < {args.min_output_bytes} bytes)"
            else:
                run_status = "done"
        else:
            run_status = "missing_output"
            error_msg = "inference completed but output file not found"

    except subprocess.TimeoutExpired:
        run_status = "timeout"
        error_msg = f"subprocess timeout after {args.timeout_sec}s"
    except Exception as exc:
        run_status = "failed"
        error_msg = f"{type(exc).__name__}: {exc}"

    duration = time.monotonic() - start

    # Diagnostics sidecar check
    diag_path = pred.with_suffix(pred.suffix + ".diagnostics.json")
    diag_ok = diag_path.exists() if run_status == "done" else None

    if run_status == "done" and not diag_ok:
        print(f"[{i}/{n_total}] WARN     {stem}: no diagnostics sidecar found")

    # Build status record
    size_bytes: Optional[int] = None
    if pred.exists():
        try:
            size_bytes = pred.stat().st_size
        except OSError:
            pass

    record = dict(
        index=i,
        stem=stem,
        pdf=str(pdf),
        prediction_path=str(pred),
        work_dir=str(work_dir),
        status=run_status,
        duration_sec=round(duration, 3),
        output_size_bytes=size_bytes,
        checkpoint=str(args.checkpoint),
        config=str(args.config),
        beam_width=args.beam_width,
        max_decode_steps=args.max_decode_steps,
        device=device,
        diagnostics_sidecar_present=diag_ok,
        error=error_msg,
    )
    _write_status_record(status_jsonl, record)

    # Update rolling duration window (thread-safe)
    with durations_lock:
        rolling_durations.append(duration)

    # Print result line
    if run_status == "done":
        print(f"[{i}/{n_total}] done     {stem}  ({size_bytes // 1024} KB)  {duration:.1f}s")
    elif run_status == "missing_output":
        print(f"[{i}/{n_total}] MISSING  {stem}: {error_msg}")
    elif run_status == "timeout":
        print(f"[{i}/{n_total}] TIMEOUT  {stem}: {error_msg}")
    else:
        short_err = (error_msg or "")[:200]
        print(f"[{i}/{n_total}] FAIL     {stem}: {short_err}")

    return record


def main() -> None:
    p = argparse.ArgumentParser(
        description="Inference-only pass: run Stage B checkpoint on the Lieder eval split "
                    "and write predicted MusicXML + Stage-D diagnostics sidecars to disk. "
                    "Run eval/score_lieder_eval.py afterwards to compute metrics."
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="path to trained Stage B .pt checkpoint",
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help=(
            "path to .yaml config matching checkpoint. "
            "NOTE: metadata-only — src.pdf_to_musicxml has no --config flag. "
            "Validated at startup (catches typos) and stored in status JSONL."
        ),
    )
    p.add_argument(
        "--name", required=True,
        help="run name (used for output dir: eval/results/lieder_<name>/)",
    )
    p.add_argument(
        "--beam-width", type=int, default=1,
        help="Stage-B beam width (default 1 = greedy; ~5x slower per beam at 5)",
    )
    p.add_argument(
        "--max-decode-steps", type=int, default=256,
        help="Stage-B max decode steps per staff (default 256; full quality is 512)",
    )
    p.add_argument(
        "--max-pieces", type=int, default=None,
        help="Truncate the eval split to the first N pieces (smoke runs). Default: all.",
    )
    p.add_argument(
        "--python", type=Path, default=None,
        help=(
            "Python interpreter used to spawn src.pdf_to_musicxml. "
            "Falls back to: sys.executable (if it can import eval), "
            "venv-cu132/Scripts/python.exe, venv/Scripts/python.exe, "
            "venv-cu132/bin/python, venv/bin/python."
        ),
    )
    p.add_argument(
        "--stage-a-weights", type=Path, default=_DEFAULT_STAGE_A_YOLO,
        help=(
            "Path to Stage-A YOLO weights (default: ~/Clarity-OMR/info/yolo.pt). "
            "Download by running sibling Clarity-OMR's omr.py once on any PDF."
        ),
    )
    p.add_argument(
        "--stage-b-device", type=str, default="cuda",
        help="Stage-B device passed to src.pdf_to_musicxml (default: cuda). "
             "Overridden per-subprocess when --devices is used.",
    )
    p.add_argument(
        "--timeout-sec", type=int, default=_DEFAULT_TIMEOUT_SEC,
        help=f"Per-piece subprocess timeout in seconds (default: {_DEFAULT_TIMEOUT_SEC}). "
             "Increase for full-quality beam search (--beam-width 5 --max-decode-steps 512).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run inference even when a valid cached prediction already exists.",
    )
    p.add_argument(
        "--min-output-bytes", type=int, default=_DEFAULT_MIN_OUTPUT_BYTES,
        help=f"Minimum output file size (bytes) for a cache hit to be valid "
             f"(default: {_DEFAULT_MIN_OUTPUT_BYTES}). Files below this threshold or "
             "missing the <score-partwise XML marker are treated as corrupted_cache.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override default output directory (eval/results/lieder_<name>/).",
    )
    p.add_argument(
        "--work-dir", type=Path, default=None,
        help="Override default working directory (eval/results/lieder_<name>_workdirs/).",
    )
    p.add_argument(
        "--jobs", type=int, default=1,
        help="Number of parallel inference subprocesses (default: 1). "
             "For multi-GPU runs use --devices to pin each job to a GPU.",
    )
    p.add_argument(
        "--devices", type=str, default=None,
        help="Comma-separated device list (e.g. cuda:0,cuda:1). "
             "When set, each subprocess is assigned a device round-robin from this pool. "
             "Defaults to --stage-b-device for all jobs when not set.",
    )
    args = p.parse_args()

    # --- Validate inputs ---------------------------------------------------
    if not args.checkpoint.exists():
        raise SystemExit(f"FATAL: checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"FATAL: config not found: {args.config}")
    if args.jobs < 1:
        raise SystemExit("FATAL: --jobs must be >= 1")

    # Resolve Python interpreter via shared helper from _scoring_utils.
    python = _resolve_venv_python(args.python)

    # --- Device pool setup -----------------------------------------------
    if args.devices:
        device_list = [d.strip() for d in args.devices.split(",") if d.strip()]
        if not device_list:
            raise SystemExit("FATAL: --devices list is empty")
        if args.jobs > 1 and len(device_list) < args.jobs:
            print(
                f"[warn] --jobs {args.jobs} but only {len(device_list)} device(s) in --devices. "
                "Multiple jobs will share the same device(s) — watch VRAM.",
                file=sys.stderr,
            )
    else:
        device_list = [args.stage_b_device]

    device_queue: Queue[str] = Queue()
    # Populate the queue with enough slots for all jobs (cycling through devices).
    for slot_idx in range(args.jobs):
        device_queue.put(device_list[slot_idx % len(device_list)])

    # --- Split hash guard -------------------------------------------------
    print(f"Lieder split hash: {split_hash()}")
    assert split_hash() == "8e7d206f53ae3976", (
        f"FATAL: split hash mismatch! Got {split_hash()}, expected 8e7d206f53ae3976. "
        "The eval split has changed — do not proceed."
    )
    print("Split hash verified: 8e7d206f53ae3976")
    print(f"Run name:        {args.name}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Config:          {args.config}  [metadata-only]")
    print(f"Python:          {python}")
    print(f"Stage-A weights: {args.stage_a_weights}")
    print(f"Stage-B device:  {args.stage_b_device}" + (f"  (devices pool: {device_list})" if args.devices else ""))
    print(f"Jobs:            {args.jobs}")
    print(f"Timeout:         {args.timeout_sec}s per piece")
    print(f"Force:           {args.force}")
    print(f"Mode:            INFERENCE ONLY (no metric scoring)")
    print()

    # --- Paths ------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = (repo_root / "data/openscore_lieder/eval_pdfs").resolve()
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (repo_root / "eval/results" / f"lieder_{args.name}").resolve()
    )
    work_base = (
        args.work_dir.resolve()
        if args.work_dir
        else (repo_root / "eval/results" / f"lieder_{args.name}_workdirs").resolve()
    )
    logs_dir = (repo_root / "eval/results" / f"lieder_{args.name}_logs").resolve()
    status_jsonl = (repo_root / "eval/results" / f"lieder_{args.name}_inference_status.jsonl").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # --- Eval pieces -------------------------------------------------------
    eval_pieces = get_eval_pieces()
    if args.max_pieces is not None:
        full_count = len(eval_pieces)
        eval_pieces = eval_pieces[: args.max_pieces]
        print(f"Truncating to first {args.max_pieces} pieces of {full_count}")
    n_total = len(eval_pieces)

    # Counters
    n_ok = 0
    n_skip = 0
    n_fail = 0
    n_missing_output = 0
    n_cached = 0
    n_corrupted_cache = 0
    n_timeout = 0
    all_records: list[dict] = []

    # Rolling duration window for ETA
    rolling_durations: Deque[float] = deque(maxlen=_ETA_WINDOW)
    durations_lock = threading.Lock()

    # Filter pieces with missing PDFs upfront so n_total reflects runnable pieces.
    runnable: list[tuple[int, object]] = []
    for i, piece in enumerate(eval_pieces, 1):
        pdf = pdf_dir / f"{piece.stem}.pdf"
        if not pdf.exists():
            print(f"[{i}/{n_total}] SKIP {piece.stem}: no rendered PDF at {pdf}")
            n_skip += 1
        else:
            runnable.append((i, piece))

    # --- Main loop / parallel dispatch ------------------------------------
    def _dispatch_piece(i_piece_tuple: tuple[int, object]) -> dict:
        i, piece = i_piece_tuple
        pdf = pdf_dir / f"{piece.stem}.pdf"
        pred = out_dir / f"{piece.stem}.musicxml"
        work_dir = work_base / piece.stem

        # Acquire a device from the pool.
        device = device_queue.get()
        try:
            rec = _run_piece(
                i=i,
                n_total=n_total,
                piece=piece,
                pdf=pdf,
                pred=pred,
                work_dir=work_dir,
                args=args,
                python=python,
                status_jsonl=status_jsonl,
                logs_dir=logs_dir,
                device=device,
                rolling_durations=rolling_durations,
                durations_lock=durations_lock,
            )
            # Print ETA after each non-cached piece (cached pieces are near-instant).
            if rec["status"] != "cached":
                with durations_lock:
                    completed_inference = len(rolling_durations)
                    remaining = len(runnable) - sum(1 for r in all_records if r["status"] != "cached") - 1
                eta = _format_eta(max(0, remaining), rolling_durations)
                print(f"  ETA: {eta} (rolling avg of last {min(_ETA_WINDOW, completed_inference)} pieces)")
            return rec
        finally:
            device_queue.put(device)

    wall_start = time.monotonic()

    if args.jobs == 1:
        # Serial path — simpler, deterministic output order.
        for i_piece in runnable:
            rec = _dispatch_piece(i_piece)
            all_records.append(rec)
    else:
        # Parallel path via ThreadPoolExecutor.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_dispatch_piece, ip): ip for ip in runnable}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as exc:
                    # Unexpected worker-level failure (not an inference error).
                    _, piece = futures[fut]
                    print(f"[worker] FATAL for {piece.stem}: {exc}", file=sys.stderr)
                    rec = dict(stem=piece.stem, status="failed", error=str(exc))
                all_records.append(rec)

    wall_time = time.monotonic() - wall_start

    # Tally counters from records.
    for rec in all_records:
        st = rec.get("status", "unknown")
        if st in ("done",):
            n_ok += 1
        elif st == "cached":
            n_cached += 1
        elif st == "missing_output":
            n_missing_output += 1
        elif st == "corrupted_cache":
            n_corrupted_cache += 1
        elif st == "timeout":
            n_timeout += 1
        else:
            n_fail += 1

    # --- Per-piece duration summary ---------------------------------------
    inference_durations = [
        r["duration_sec"]
        for r in all_records
        if r.get("duration_sec") is not None and r.get("status") not in ("cached",)
    ]
    if inference_durations:
        print()
        print("=== Per-piece duration summary ===")
        print(f"  Pieces timed:  {len(inference_durations)}")
        print(f"  Mean:          {statistics.mean(inference_durations):.1f}s")
        if len(inference_durations) >= 2:
            print(f"  Median:        {statistics.median(inference_durations):.1f}s")
            sorted_d = sorted(inference_durations)
            p95_idx = int(len(sorted_d) * 0.95)
            print(f"  P95:           {sorted_d[p95_idx]:.1f}s")
        slowest = max(all_records, key=lambda r: r.get("duration_sec") or 0.0)
        fastest = min(
            [r for r in all_records if r.get("duration_sec") is not None and r.get("status") not in ("cached",)],
            key=lambda r: r.get("duration_sec") or float("inf"),
            default=None,
        )
        print(f"  Slowest:       {slowest.get('stem', '?')} ({slowest.get('duration_sec', '?'):.1f}s)")
        if fastest:
            print(f"  Fastest:       {fastest.get('stem', '?')} ({fastest.get('duration_sec', '?'):.1f}s)")

    # --- Post-run summary -------------------------------------------------
    output_sizes = [
        r["output_size_bytes"]
        for r in all_records
        if r.get("output_size_bytes") is not None
    ]
    print()
    print("=== Inference complete ===")
    print(f"  OK (done):         {n_ok}/{n_total}")
    print(f"  Cached:            {n_cached}/{n_total}")
    print(f"  Skipped (no PDF):  {n_skip}/{n_total}")
    print(f"  Missing output:    {n_missing_output}/{n_total}")
    print(f"  Corrupted cache:   {n_corrupted_cache}/{n_total}")
    print(f"  Timeout:           {n_timeout}/{n_total}")
    print(f"  Failed:            {n_fail}/{n_total}")
    if output_sizes:
        mean_kb = statistics.mean(output_sizes) / 1024
        print(f"  Mean output size:  {mean_kb:.1f} KB")
    print(f"  Total wall time:   {wall_time:.0f}s ({wall_time/60:.1f} min)")
    print()
    print(f"Predicted XMLs written to:  {out_dir}")
    print(f"Per-piece logs:             {logs_dir}")
    print(f"Inference status JSONL:     {status_jsonl}")
    print()
    print("Next step — run metric scoring:")
    print(f"  python -m eval.score_lieder_eval \\")
    print(f"      --predictions-dir {out_dir} \\")
    print(f"      --reference-dir {(repo_root / 'data/openscore_lieder/scores').resolve()} \\")
    print(f"      --name {args.name}")


if __name__ == "__main__":
    main()
