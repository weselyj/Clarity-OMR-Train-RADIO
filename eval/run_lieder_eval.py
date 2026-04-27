"""Run a trained Clarity-OMR-Train-RADIO checkpoint through the Lieder eval split.

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

Usage:
    venv\\Scripts\\python -m eval.run_lieder_eval \\
        --checkpoint checkpoints/mvp_radio_stage2/stage2-radio-mvp_step_0000150.pt \\
        --config configs/train_stage2_radio_mvp.yaml \\
        --name mvp \\
        --limit 3   # optional smoke cap
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# Insert repo root so imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval.lieder_split import get_eval_pieces, split_hash
from eval.playback import playback_f
from eval.tedn import compute_tedn
from eval.linearized_musicxml import compute_linearized_ser

# Path to our venv's Python — used to invoke src.pdf_to_musicxml as a subprocess
VENV_PYTHON = Path(__file__).resolve().parents[1] / "venv" / "Scripts" / "python.exe"

# Stage A YOLO checkpoint is shipped by sibling Clarity-OMR (HuggingFace download
# triggered on first omr.py run, lands at info/yolo.pt). We don't train Stage A
# in this repo, so reuse it.
STAGE_A_YOLO = Path.home() / "Clarity-OMR" / "info" / "yolo.pt"

# Inference uses our own src.pdf_to_musicxml — our fork already has the full
# pipeline (Stage A wrapper + cli helpers + decoding/* + pipeline/*), and our
# src.eval.evaluate_stage_b_checkpoint detects DoRA-wrapped checkpoints and
# applies _prepare_model_for_dora before load_state_dict, so our trained
# checkpoint loads via --stage-b-checkpoint with no further conversion.
PDF_TO_MUSICXML_TIMEOUT_SEC = 1800  # 30-min cap per piece


def run_inference(
    checkpoint: Path,
    config: Path,
    pdf: Path,
    out: Path,
    work_dir: Path,
    *,
    beam_width: int = 1,
    max_decode_steps: int = 256,
) -> None:
    """Run our trained Stage B + sibling YOLO Stage A on `pdf`, write MusicXML to `out`.

    Defaults are tuned for MVP-quality models: greedy decode (beam=1) with a 256-step
    cap is ~10x faster than the standard beam=5 / max=512 used at full-Stage-3 quality.
    For real eval against a Stage 3 checkpoint, override via --beam-width / --max-decode-steps.
    """
    import subprocess
    if not STAGE_A_YOLO.exists():
        raise SystemExit(
            f"FATAL: Stage A YOLO weights not found at {STAGE_A_YOLO}. "
            "Run sibling Clarity-OMR's omr.py once on any PDF to trigger the HF download."
        )
    repo_root = Path(__file__).resolve().parents[1]
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PYTHON),
        "-m", "src.pdf_to_musicxml",
        "--pdf", str(pdf),
        "--output-musicxml", str(out),
        "--weights", str(STAGE_A_YOLO),
        "--stage-b-checkpoint", str(checkpoint),
        "--work-dir", str(work_dir),
        "--project-root", str(repo_root),
        "--stage-b-device", "cuda",
        "--beam-width", str(beam_width),
        "--max-decode-steps", str(max_decode_steps),
        "--image-height", "250",
        "--image-max-width", "2500",
        "--length-penalty-alpha", "0.4",
        "--pdf-dpi", "300",
        "--quiet",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root), timeout=PDF_TO_MUSICXML_TIMEOUT_SEC)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="path to trained Stage B .pt checkpoint",
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help="path to .yaml config matching checkpoint",
    )
    p.add_argument(
        "--name", required=True,
        help="run name (used for output dir + CSV filename)",
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
        "--limit", type=int, default=None,
        help="Optional cap on number of pieces (for smoke runs)",
    )
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"FATAL: checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"FATAL: config not found: {args.config}")

    print(f"Lieder split hash: {split_hash()}")
    assert split_hash() == "8e7d206f53ae3976", (
        f"FATAL: split hash mismatch! Got {split_hash()}, expected 8e7d206f53ae3976. "
        "The eval split has changed - do not proceed."
    )
    print("Split hash verified: 8e7d206f53ae3976")
    print(f"Run name: {args.name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print()

    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = (repo_root / "data/openscore_lieder/eval_pdfs").resolve()
    out_dir = (repo_root / "eval/results" / f"lieder_{args.name}").resolve()
    work_base = (repo_root / "eval/results" / f"lieder_{args.name}_workdirs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_pieces = get_eval_pieces()
    if args.limit is not None:
        eval_pieces = eval_pieces[: args.limit]
        print(f"--limit {args.limit}: running on first {len(eval_pieces)} pieces only")
    n_total = len(eval_pieces)
    # Each row: (piece, onset_f1, tedn, lin_ser,
    #            stage_d_skipped_notes, stage_d_skipped_chords,
    #            stage_d_missing_durations, stage_d_malformed_spans,
    #            stage_d_unknown_tokens, stage_d_fallback_rests,
    #            stage_d_raised_count, stage_d_first_error)
    rows: list[tuple] = []

    def _read_stage_d_diag(pred_path: Path) -> tuple:
        """Return the 8 Stage-D diagnostic CSV values for *pred_path*.

        Looks for <pred_path>.diagnostics.json alongside the MusicXML output.
        Returns a tuple of 8 values (all None if the sidecar is absent or unreadable).
        """
        diag_path = pred_path.with_suffix(pred_path.suffix + ".diagnostics.json")
        try:
            raw = json.loads(diag_path.read_text(encoding="utf-8"))
            raised = raw.get("raised_during_part_append", [])
            first_error = raised[0].get("error_message", "") if raised else ""
            return (
                raw.get("skipped_notes"),
                raw.get("skipped_chords"),
                raw.get("missing_durations"),
                raw.get("malformed_spans"),
                raw.get("unknown_tokens"),
                raw.get("fallback_rests"),
                len(raised),
                first_error,
            )
        except Exception:
            return (None, None, None, None, None, None, None, None)

    for i, piece in enumerate(eval_pieces, 1):
        # piece from get_eval_pieces() is relative to cwd; resolve to absolute
        piece_abs = (repo_root / piece).resolve()
        pdf = pdf_dir / f"{piece.stem}.pdf"
        if not pdf.exists():
            print(f"[{i}/{n_total}] SKIP {piece.stem}: no rendered PDF")
            rows.append((piece.stem, None, None, None) + (None,) * 8)
            continue
        try:
            pred = out_dir / f"{piece.stem}.musicxml"
            work_dir = work_base / piece.stem
            if not pred.exists():
                print(f"[{i}/{n_total}] inference {piece.stem} ...")
                run_inference(
                    args.checkpoint, args.config, pdf, pred, work_dir,
                    beam_width=args.beam_width,
                    max_decode_steps=args.max_decode_steps,
                )
            else:
                print(f"[{i}/{n_total}] cached  {piece.stem}")
            # Read Stage D diagnostics sidecar (written by src.pdf_to_musicxml).
            stage_d_cols = _read_stage_d_diag(pred)
            f1 = playback_f(pred=pred, gt=piece_abs)["f"]
            try:
                tedn = compute_tedn(piece_abs, pred)
            except Exception as te:
                print(f"[{i}/{n_total}]   tedn WARN {piece.stem}: {te}")
                tedn = None
            try:
                lin_ser = compute_linearized_ser(piece_abs, pred)
            except Exception as le:
                print(f"[{i}/{n_total}]   lin_ser WARN {piece.stem}: {le}")
                lin_ser = None
            rows.append((piece.stem, f1, tedn, lin_ser) + stage_d_cols)
            tedn_str = f"{tedn:.4f}" if tedn is not None else "N/A"
            lin_str = f"{lin_ser:.4f}" if lin_ser is not None else "N/A"
            print(f"[{i}/{n_total}] {piece.stem}: onset_f1={f1:.4f}  tedn={tedn_str}  lin_ser={lin_str}")
        except Exception as e:
            print(f"[{i}/{n_total}] FAIL {piece.stem}: {type(e).__name__}: {e}")
            rows.append((piece.stem, None, None, None) + (None,) * 8)

    csv_path = (repo_root / "eval/results" / f"lieder_{args.name}.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "piece", "onset_f1", "tedn", "linearized_ser",
            "stage_d_skipped_notes", "stage_d_skipped_chords",
            "stage_d_missing_durations", "stage_d_malformed_spans",
            "stage_d_unknown_tokens", "stage_d_fallback_rests",
            "stage_d_raised_count", "stage_d_first_error",
        ])
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
    print(f"\n=== Lieder Eval Results ({args.name}) ===")
    print(f"Pieces evaluated: {len(valid)} / {n_total} (failed/skipped: {failed_count})")
    print(f"Mean onset-F1:   {mean_f1:.4f}")
    print(f"Median onset-F1: {med_f1:.4f}")
    print(f"Min onset-F1:    {min_f1:.4f}")
    print(f"Max onset-F1:    {max_f1:.4f}")


if __name__ == "__main__":
    main()
