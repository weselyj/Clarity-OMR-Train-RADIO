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

Uses src.inference.system_pipeline.SystemInferencePipeline — a single instance
is constructed once before the piece loop (YOLO + Stage B weights loaded once)
and reused across all pieces. This avoids per-piece model reload overhead and is
safe because music21/zss state stays out of inference (scoring subprocess-
isolated in score_lieder_eval.py).

Uses the in-repo system-level YOLO weights (default
runs/detect/runs/yolo26m_systems/weights/best.pt; override via
--stage-a-weights). Loads the trained Stage B checkpoint via
--checkpoint. src.eval.evaluate_stage_b_checkpoint detects the
DoRA-wrapped (PEFT base_model.model.* with lora_*) checkpoint format
automatically and applies _prepare_model_for_dora before load_state_dict.

Defaults to greedy decode (beam_width=1, max_decode_steps=2048) — aligned
with SystemInferencePipeline's per-system-crop default. Override --beam-width
to 5 for higher-quality (slower) decoding.

NOTE: --config is metadata-only. The config path is validated at startup (to
catch typos early) and written to the per-piece status JSONL as run metadata,
but it is NOT forwarded to the pipeline — SystemInferencePipeline uses config
baked into the checkpoint.

Usage:
    python -m eval.run_lieder_eval \\
        --checkpoint checkpoints/mvp_radio_stage2/stage2-radio-mvp_step_0000150.pt \\
        --config configs/train_stage2_radio_mvp.yaml \\
        --name mvp \\
        --max-pieces 20   # optional smoke cap

Then score:
    python -m eval.score_lieder_eval \\
        --predictions-dir eval/results/lieder_mvp \\
        --reference-dir data/openscore_lieder/scores \\
        --name mvp
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

# Insert repo root so imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval.lieder_split import get_eval_pieces, split_hash

# Default Stage A YOLO checkpoint (in-repo, produced by Stage A training).
# Override via --stage-a-weights. Path is repo-relative; resolved against
# the current working directory (typically the repo root).
_DEFAULT_STAGE_A_YOLO = Path("runs/detect/runs/yolo26m_systems/weights/best.pt")

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
    *,
    pipeline,
    pdf: Path,
    out: Path,
    work_dir: Path,
) -> None:
    """Run inference on a single piece via the in-process SystemInferencePipeline.

    Writes the predicted .musicxml and the .musicxml.diagnostics.json sidecar
    next to `out`. Does NOT score; metric scoring happens in Phase 2 via
    eval.score_lieder_eval (subprocess-isolated to prevent music21/zss memory
    accumulation — see PR #26 motivation in this file's header).
    """
    from src.pipeline.export_musicxml import StageDExportDiagnostics

    work_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    diagnostics = StageDExportDiagnostics()
    score = pipeline.run_pdf(pdf, diagnostics=diagnostics)
    pipeline.export_musicxml(score, out, diagnostics=diagnostics)


def invoke_scoring_phase(
    *,
    predictions_dir: Path,
    ground_truth_dir: Path,
    out_csv: Path,
    with_tedn: bool = False,
) -> int:
    """Spawn eval.score_lieder_eval as a subprocess for Phase 2 scoring.

    NEVER imports the scorer inline — keeps music21/zss isolated from the
    long-running inference process.
    """
    cmd = [
        sys.executable, "-m", "eval.score_lieder_eval",
        "--predictions-dir", str(predictions_dir),
        "--ground-truth-dir", str(ground_truth_dir),
        "--out-csv", str(out_csv),
    ]
    if with_tedn:
        cmd.append("--tedn")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _run_piece(
    *,
    i: int,
    n_total: int,
    piece,
    pdf: Path,
    pred: Path,
    work_dir: Path,
    args: argparse.Namespace,
    pipeline,
    status_jsonl: Path,
) -> dict:
    """Process one piece end-to-end. Returns a status record dict."""
    stem = piece.stem

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
                device=args.stage_b_device,
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

    # --- Run inference -----------------------------------------------------
    start = time.monotonic()
    error_msg: Optional[str] = None
    run_status = "unknown"

    print(f"[{i}/{n_total}] inference {stem} (device={args.stage_b_device}) ...")

    try:
        run_inference(pipeline=pipeline, pdf=pdf, out=pred, work_dir=work_dir)
        if pred.exists():
            size = pred.stat().st_size
            if size < args.min_output_bytes:
                run_status = "missing_output"
                error_msg = f"output too small ({size} < {args.min_output_bytes} bytes)"
            else:
                run_status = "done"
        else:
            run_status = "missing_output"
            error_msg = "inference completed but output file not found"

    except Exception as exc:
        run_status = f"failed:{type(exc).__name__}"
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
        device=args.stage_b_device,
        diagnostics_sidecar_present=diag_ok,
        error=error_msg,
    )
    _write_status_record(status_jsonl, record)

    # Print result line
    if run_status == "done":
        print(f"[{i}/{n_total}] done     {stem}  ({size_bytes // 1024} KB)  {duration:.1f}s")
    elif run_status == "missing_output":
        print(f"[{i}/{n_total}] MISSING  {stem}: {error_msg}")
    else:
        short_err = (error_msg or "")[:200]
        print(f"[{i}/{n_total}] FAIL     {stem}: {short_err}")

    return record


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the run_lieder_eval CLI parser. Module-level for testability."""
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
            "NOTE: metadata-only — config is baked into the checkpoint. "
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
        "--max-decode-steps", type=int, default=2048,
        help="Stage-B max decode steps per system crop (default 2048; aligns with "
             "SystemInferencePipeline default).",
    )
    p.add_argument(
        "--max-pieces", type=int, default=None,
        help="Truncate the eval split to the first N pieces (smoke runs). Default: all.",
    )
    p.add_argument(
        "--stage-a-weights", type=Path, default=_DEFAULT_STAGE_A_YOLO,
        help=(
            f"Path to Stage A YOLO weights (default: {_DEFAULT_STAGE_A_YOLO}). "
            "Train Stage A first via scripts/train_yolo.py — see docs/TRAINING.md."
        ),
    )
    p.add_argument(
        "--stage-b-device", type=str, default="cuda",
        help="Device for SystemInferencePipeline (default: cuda).",
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
        "--run-scoring", action="store_true",
        help=(
            "After the inference loop completes, automatically spawn "
            "eval.score_lieder_eval as a subprocess to compute metrics. "
            "Subprocess-isolated to prevent music21/zss memory accumulation (PR #26)."
        ),
    )
    p.add_argument(
        "--tedn", action="store_true",
        help=(
            "Pass --tedn to eval.score_lieder_eval when --run-scoring is set. "
            "Enables tree-edit-distance normalization in the scorer."
        ),
    )
    return p


def main() -> None:
    p = build_argument_parser()
    args = p.parse_args()

    # --- Validate inputs ---------------------------------------------------
    # Stage A weights check first: an actionable error pointing at the training
    # docs is more useful than the cryptic failure the YOLO loader produces
    # later, and this is the most common misconfiguration.
    if not args.stage_a_weights.is_file():
        p.error(
            f"Stage A weights not found at {args.stage_a_weights}. "
            f"Train Stage A first (see docs/TRAINING.md) or pass --stage-a-weights."
        )
    if not args.checkpoint.exists():
        raise SystemExit(f"FATAL: checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"FATAL: config not found: {args.config}")

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
    print(f"Stage-A weights: {args.stage_a_weights}")
    print(f"Stage-B device:  {args.stage_b_device}")
    print(f"Beam width:      {args.beam_width}")
    print(f"Max decode steps:{args.max_decode_steps}")
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
    status_jsonl = (repo_root / "eval/results" / f"lieder_{args.name}_inference_status.jsonl").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Instantiate pipeline (once for all pieces) -----------------------
    from src.inference.system_pipeline import SystemInferencePipeline
    pipeline = SystemInferencePipeline(
        yolo_weights=args.stage_a_weights,
        stage_b_ckpt=args.checkpoint,
        device=args.stage_b_device,
        beam_width=args.beam_width,
        max_decode_steps=args.max_decode_steps,
    )

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
    all_records: list[dict] = []

    # Rolling duration window for ETA
    rolling_durations: Deque[float] = deque(maxlen=_ETA_WINDOW)

    # Filter pieces with missing PDFs upfront so n_total reflects runnable pieces.
    runnable: list[tuple[int, object]] = []
    for i, piece in enumerate(eval_pieces, 1):
        pdf = pdf_dir / f"{piece.stem}.pdf"
        if not pdf.exists():
            print(f"[{i}/{n_total}] SKIP {piece.stem}: no rendered PDF at {pdf}")
            n_skip += 1
        else:
            runnable.append((i, piece))

    # --- Serial piece loop -----------------------------------------------
    wall_start = time.monotonic()

    for i_piece in runnable:
        i, piece = i_piece
        pdf = pdf_dir / f"{piece.stem}.pdf"
        pred = out_dir / f"{piece.stem}.musicxml"
        work_dir = work_base / piece.stem

        rec = _run_piece(
            i=i,
            n_total=n_total,
            piece=piece,
            pdf=pdf,
            pred=pred,
            work_dir=work_dir,
            args=args,
            pipeline=pipeline,
            status_jsonl=status_jsonl,
        )
        all_records.append(rec)

        # Print ETA after each non-cached piece (cached pieces are near-instant).
        if rec["status"] != "cached" and rec.get("duration_sec") is not None:
            rolling_durations.append(rec["duration_sec"])
            completed_inference = len(rolling_durations)
            remaining = len(runnable) - len(all_records)
            eta = _format_eta(max(0, remaining), rolling_durations)
            print(f"  ETA: {eta} (rolling avg of last {min(_ETA_WINDOW, completed_inference)} pieces)")

    wall_time = time.monotonic() - wall_start

    # Tally counters from records.
    for rec in all_records:
        st = rec.get("status", "unknown")
        if st == "done":
            n_ok += 1
        elif st == "cached":
            n_cached += 1
        elif st == "missing_output":
            n_missing_output += 1
        elif st == "corrupted_cache":
            n_corrupted_cache += 1
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
    print(f"  Failed:            {n_fail}/{n_total}")
    if output_sizes:
        mean_kb = statistics.mean(output_sizes) / 1024
        print(f"  Mean output size:  {mean_kb:.1f} KB")
    print(f"  Total wall time:   {wall_time:.0f}s ({wall_time/60:.1f} min)")
    print()
    print(f"Predicted XMLs written to:  {out_dir}")
    print(f"Inference status JSONL:     {status_jsonl}")
    print()
    print("Next step — run metric scoring:")
    print(f"  python -m eval.score_lieder_eval \\")
    print(f"      --predictions-dir {out_dir} \\")
    print(f"      --reference-dir {(repo_root / 'data/openscore_lieder/scores').resolve()} \\")
    print(f"      --name {args.name}")

    if args.run_scoring:
        ground_truth_dir = (repo_root / "data/openscore_lieder/scores").resolve()
        out_csv = (repo_root / "eval/results" / f"lieder_{args.name}_scores.csv").resolve()
        print()
        print(f"--run-scoring: spawning eval.score_lieder_eval ...")
        rc = invoke_scoring_phase(
            predictions_dir=out_dir,
            ground_truth_dir=ground_truth_dir,
            out_csv=out_csv,
            with_tedn=args.tedn,
        )
        if rc != 0:
            raise SystemExit(f"eval.score_lieder_eval exited with code {rc}")
        print(f"Scoring complete. Results: {out_csv}")


if __name__ == "__main__":
    main()
