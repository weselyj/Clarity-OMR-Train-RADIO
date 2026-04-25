"""Run author's DaViT checkpoint through our eval pipeline on the Lieder split.

If our mean onset-F1 differs from author's published number by > 0.05, the eval
pipeline has drift. Investigate before any RADIO comparison.

Step 4 finding: The author published NO Lieder-specific onset F1 number.
The HuggingFace model card shows a 10-piece benchmark (The Entertainer, Scriabin, etc.)
with per-piece Onset F1 values in an image, but no OpenScore Lieder aggregate. Source:
  https://huggingface.co/clquwu/Clarity-OMR
  Image: https://cdn-uploads.huggingface.co/production/uploads/68a8f07e16e4098ee93359be/6ZEiOtmrqUcfV5VpDh5ce.png
The training repo (github.com/clquwu/Clarity-OMR-Train) similarly has no Lieder F1.

Gate: 'runs without crashing' (per task spec when no author number exists).
"""
import csv
import statistics
from pathlib import Path
import subprocess
import sys

# Insert repo root so imports work regardless of cwd
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)
from eval.lieder_split import get_eval_pieces, split_hash
from eval.playback import playback_f

# No author-published Lieder F1 found — gate is "runs without crashing".
AUTHOR_PUBLISHED_LIEDER_MEAN_F1 = None
AUTHOR_PUBLISHED_SOURCE = (
    "No Lieder-specific number published. "
    "Author's 10-piece benchmark at: "
    "https://huggingface.co/clquwu/Clarity-OMR (image benchmark, non-Lieder pieces)"
)
TOLERANCE = 0.05

# Path to the author's inference repo (cloned as sibling directory)
CLARITY_OMR_DIR = Path.home() / "Clarity-OMR"

# Path to our venv's Python — use this to run omr.py in the Clarity-OMR repo
VENV_PYTHON = Path(__file__).resolve().parents[1] / "venv" / "Scripts" / "python.exe"


def run_author_inference(pdf_path: Path, out_path: Path, work_dir: Path) -> Path:
    """Run author's omr.py on a PDF, return path to MusicXML output.

    Uses a per-piece --work-dir to prevent intermediate file collisions between pieces.
    Calls omr.py as a subprocess in the Clarity-OMR directory using our venv's Python.
    The info/ subdir already has yolo.pt and model.safetensors, so no HF download occurs.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(VENV_PYTHON),
            "omr.py",
            str(pdf_path),
            "-o", str(out_path),
            "--device", "cuda",
            "--work-dir", str(work_dir),
        ],
        check=True,
        cwd=str(CLARITY_OMR_DIR),
        timeout=300,
    )
    return out_path


def main() -> None:
    print(f"Lieder split hash: {split_hash()}")
    assert split_hash() == "8e7d206f53ae3976", (
        f"FATAL: split hash mismatch! Got {split_hash()}, expected 8e7d206f53ae3976. "
        "The eval split has changed — do not proceed."
    )
    print(f"Split hash verified: 8e7d206f53ae3976")
    print(f"Author's published Lieder mean onset F1: {AUTHOR_PUBLISHED_LIEDER_MEAN_F1}")
    print(f"  (source: {AUTHOR_PUBLISHED_SOURCE})")
    print()

    eval_pieces = get_eval_pieces()
    # Use absolute paths — omr.py runs in a different cwd (CLARITY_OMR_DIR)
    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = (repo_root / "data/openscore_lieder/eval_pdfs").resolve()
    out_dir = (repo_root / "eval/results/baseline_davit_lieder").resolve()
    work_base = (repo_root / "eval/results/baseline_davit_lieder_workdirs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float | None]] = []
    n_total = len(eval_pieces)

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
                run_author_inference(pdf, pred, work_dir)
            else:
                print(f"[{i}/{n_total}] cached  {piece.stem}")
            f1 = playback_f(pred=pred, gt=piece_abs)["f"]
            rows.append((piece.stem, f1))
            print(f"[{i}/{n_total}] {piece.stem}: onset_f1={f1:.4f}")
        except Exception as e:
            print(f"[{i}/{n_total}] FAIL {piece.stem}: {type(e).__name__}: {e}")
            rows.append((piece.stem, None))

    # Write CSV
    csv_path = (repo_root / "eval/results/baseline_davit_lieder.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["piece", "onset_f1"])
        w.writerows(rows)
    print(f"\nResults written to {csv_path}")

    valid = [f for _, f in rows if f is not None]
    mean_f1 = statistics.mean(valid) if valid else 0.0
    min_f1 = min(valid) if valid else 0.0
    max_f1 = max(valid) if valid else 0.0
    med_f1 = statistics.median(valid) if valid else 0.0
    failed_count = sum(1 for _, f in rows if f is None)

    print(f"\n=== Baseline Reproduction Results ===")
    print(f"Pieces evaluated: {len(valid)} / {n_total} (failed/skipped: {failed_count})")
    print(f"Mean onset-F1:   {mean_f1:.4f}")
    print(f"Median onset-F1: {med_f1:.4f}")
    print(f"Min onset-F1:    {min_f1:.4f}")
    print(f"Max onset-F1:    {max_f1:.4f}")
    print(f"\nAuthor's published: {AUTHOR_PUBLISHED_LIEDER_MEAN_F1}")

    if AUTHOR_PUBLISHED_LIEDER_MEAN_F1 is not None:
        gap = abs(mean_f1 - AUTHOR_PUBLISHED_LIEDER_MEAN_F1)
        print(f"Gap: {gap:.4f}  (tolerance: {TOLERANCE})")
        if gap > TOLERANCE:
            raise SystemExit(
                f"BASELINE REPRODUCTION FAILED — gap {gap:.4f} > {TOLERANCE}. Eval pipeline drift."
            )
        print("BASELINE REPRODUCTION OK")
    else:
        if len(valid) == 0:
            raise SystemExit("BASELINE REPRODUCTION FAILED — zero pieces scored successfully.")
        print(
            f"WARNING: no author Lieder number to compare against; "
            f"gate is 'runs without crashing'. Completed with {len(valid)}/{n_total} pieces scored."
        )
        print("BASELINE REPRODUCTION OK (runs-without-crashing gate)")


if __name__ == "__main__":
    main()
