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

Usage:
    venv\\Scripts\\python -m eval.run_lieder_eval \\
        --checkpoint checkpoints/mvp_radio_stage2/stage2-radio-mvp_step_0000150.pt \\
        --config configs/train_stage2_radio_mvp.yaml \\
        --name mvp \\
        --max-pieces 20   # optional smoke cap

Then score:
    venv\\Scripts\\python -m eval.score_lieder_eval \\
        --predictions-dir eval/results/lieder_mvp \\
        --reference-dir data/openscore_lieder/scores \\
        --name mvp
"""
import argparse
import sys
from pathlib import Path

# Insert repo root so imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval.lieder_split import get_eval_pieces, split_hash

# Path to our venv's Python — used to invoke src.pdf_to_musicxml as a subprocess
VENV_PYTHON = Path(__file__).resolve().parents[1] / "venv" / "Scripts" / "python.exe"

# Stage A YOLO checkpoint is shipped by sibling Clarity-OMR (HuggingFace download
# triggered on first omr.py run, lands at info/yolo.pt). We don't train Stage A
# in this repo, so reuse it.
STAGE_A_YOLO = Path.home() / "Clarity-OMR" / "info" / "yolo.pt"

# Inference timeout: 30 minutes per piece
PDF_TO_MUSICXML_TIMEOUT_SEC = 1800


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

    Spawns src.pdf_to_musicxml in a *child process* so that the RADIO encoder,
    checkpoint weights, and all torch tensors are released when the child exits.
    This is the key isolation that prevents OOM across pieces.

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
        help="path to .yaml config matching checkpoint",
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
        help="Truncate the eval split to the first N pieces (for smoke runs or staged rollout). "
             "Default None runs all pieces in the split.",
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
    print(f"Run name:   {args.name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(f"Mode:       INFERENCE ONLY (no metric scoring)")
    print()

    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = (repo_root / "data/openscore_lieder/eval_pdfs").resolve()
    out_dir = (repo_root / "eval/results" / f"lieder_{args.name}").resolve()
    work_base = (repo_root / "eval/results" / f"lieder_{args.name}_workdirs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_pieces = get_eval_pieces()
    if args.max_pieces is not None:
        full_count = len(eval_pieces)
        eval_pieces = eval_pieces[: args.max_pieces]
        print(f"Truncating to first {args.max_pieces} pieces of {full_count}")
    n_total = len(eval_pieces)
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, piece in enumerate(eval_pieces, 1):
        pdf = pdf_dir / f"{piece.stem}.pdf"

        if not pdf.exists():
            print(f"[{i}/{n_total}] SKIP {piece.stem}: no rendered PDF at {pdf}")
            n_skip += 1
            continue

        pred = out_dir / f"{piece.stem}.musicxml"
        work_dir = work_base / piece.stem

        try:
            if not pred.exists():
                print(f"[{i}/{n_total}] inference {piece.stem} ...")
                run_inference(
                    args.checkpoint, args.config, pdf, pred, work_dir,
                    beam_width=args.beam_width,
                    max_decode_steps=args.max_decode_steps,
                )
                if pred.exists():
                    print(f"[{i}/{n_total}] done     {piece.stem}  ({pred.stat().st_size // 1024} KB)")
                else:
                    print(f"[{i}/{n_total}] WARN     {piece.stem}: inference completed but output not found")
            else:
                print(f"[{i}/{n_total}] cached   {piece.stem}  ({pred.stat().st_size // 1024} KB)")
            n_ok += 1
        except Exception as e:
            print(f"[{i}/{n_total}] FAIL {piece.stem}: {type(e).__name__}: {e}")
            n_fail += 1

    print()
    print(f"=== Inference complete ===")
    print(f"  OK:      {n_ok}/{n_total}")
    print(f"  Skipped: {n_skip}/{n_total}")
    print(f"  Failed:  {n_fail}/{n_total}")
    print()
    print(f"Predicted XMLs written to: {out_dir}")
    print()
    print("Next step — run metric scoring:")
    print(f"  venv\\Scripts\\python -m eval.score_lieder_eval \\")
    print(f"      --predictions-dir {out_dir} \\")
    print(f"      --reference-dir {(repo_root / 'data/openscore_lieder/scores').resolve()} \\")
    print(f"      --name {args.name}")


if __name__ == "__main__":
    main()
