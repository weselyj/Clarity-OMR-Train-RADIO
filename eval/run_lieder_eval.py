"""Run a trained Clarity-OMR-Train-RADIO checkpoint through the Lieder eval split.

SKELETON STATE: the inference call (run_inference) is a stub. Fill it in once
Task 12 (MVP) produces a checkpoint — at that point we'll know:

  - exact checkpoint structure (Stage A YOLO + Stage B DaViT/RADIO);
  - whether to call sibling repo's `src.pdf_to_musicxml.main()` with our
    --weights / --stage-b-checkpoint pointing at our trained files, or to
    compose Stage A detection + Stage B decode + MusicXML serialization
    inline (no inference module exists in this repo yet);
  - whether DoRA-wrapped Stage B layers need any unwrapping at inference.

Until then this script raises NotImplementedError on the first piece.
The surrounding harness (Lieder split iteration, work-dir per piece,
absolute-path ground-truth resolution, resume guard, CSV writing, gate
logic) mirrors eval/run_baseline_reproduction.py and is ready to use as
soon as the inference stub is filled in.

Usage:
    venv\\Scripts\\python -m eval.run_lieder_eval \\
        --checkpoint path\\to\\stage_b.pt \\
        --config configs\\stage_b\\my_run.yaml \\
        --name my_run
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

# Insert repo root so imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval.lieder_split import get_eval_pieces, split_hash
from eval.playback import playback_f

# Sibling repo holding the author's pdf->musicxml pipeline. Once Task 12 lands
# and we know the integration shape, run_inference() will likely import
# src.pdf_to_musicxml from this directory and call it with our weight paths.
CLARITY_OMR_DIR = Path.home() / "Clarity-OMR"


def run_inference(
    checkpoint: Path,
    config: Path,
    pdf: Path,
    out: Path,
    work_dir: Path,
) -> None:
    """Run our trained model on `pdf`, write MusicXML to `out`.

    SKELETON — fill in after Task 12 produces an MVP checkpoint.

    Most likely shape:
      1. Resolve Stage A YOLO weights and Stage B checkpoint paths from
         our checkpoint + config.
      2. Either
           (a) sys.path.insert(0, str(CLARITY_OMR_DIR)) and call
               src.pdf_to_musicxml.main() with overridden argv (--weights,
               --stage-b-checkpoint, --pdf, --output-musicxml, --work-dir,
               --stage-b-device cuda); or
           (b) build Stage A + Stage B via src.train.model_factory, run the
               PDF through staves -> token decode -> MusicXML serialization
               inline (requires inference utilities we don't have yet).
      3. Make sure DoRA-wrapped layers are loaded correctly (merge or keep
         adapters live — TBD per MVP results).
    """
    raise NotImplementedError(
        f"run_inference is a skeleton; fill in after Task 12. "
        f"checkpoint={checkpoint}, pdf={pdf}, out={out}, work_dir={work_dir}"
    )


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
    n_total = len(eval_pieces)
    rows: list[tuple[str, float | None]] = []

    for i, piece in enumerate(eval_pieces, 1):
        # piece from get_eval_pieces() is relative to cwd; resolve to absolute
        piece_abs = (repo_root / piece).resolve()
        pdf = pdf_dir / f"{piece.stem}.pdf"
        if not pdf.exists():
            print(f"[{i}/{n_total}] SKIP {piece.stem}: no rendered PDF")
            rows.append((piece.stem, None))
            continue
        try:
            pred = out_dir / f"{piece.stem}.musicxml"
            work_dir = work_base / piece.stem
            if not pred.exists():
                print(f"[{i}/{n_total}] inference {piece.stem} ...")
                run_inference(args.checkpoint, args.config, pdf, pred, work_dir)
            else:
                print(f"[{i}/{n_total}] cached  {piece.stem}")
            f1 = playback_f(pred=pred, gt=piece_abs)["f"]
            rows.append((piece.stem, f1))
            print(f"[{i}/{n_total}] {piece.stem}: onset_f1={f1:.4f}")
        except Exception as e:
            print(f"[{i}/{n_total}] FAIL {piece.stem}: {type(e).__name__}: {e}")
            rows.append((piece.stem, None))

    csv_path = (repo_root / "eval/results" / f"lieder_{args.name}.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["piece", "onset_f1"])
        w.writerows(rows)
    print(f"\nResults written to {csv_path}")

    valid = [f for _, f in rows if f is not None]
    failed_count = sum(1 for _, f in rows if f is None)
    if not valid:
        print(f"\nNo pieces scored successfully ({failed_count}/{n_total} failed/skipped).")
        print("(Expected while run_inference is still a skeleton.)")
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
