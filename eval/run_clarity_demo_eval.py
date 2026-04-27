"""Run a trained Clarity-OMR-Train-RADIO checkpoint against the 4 canonical
demo pieces shown on the public HuggingFace model card
(huggingface.co/clquwu/Clarity-OMR).

**Inference-only pass.** This script does NOT compute metrics. Its only job is
to run the RADIO/YOLO inference pipeline for each piece and write the predicted
MusicXML (plus optional Stage-D diagnostics sidecar) to disk. Metric scoring
is handled by the separate eval/score_demo_eval.py, which subprocess-isolates
per-piece metric computation so music21/zss memory is fully reclaimed after
each piece.

This two-pass design was adopted after Phase A profiling confirmed that metric
scoring (music21 parse + ZSS tree edit distance for tedn; Levenshtein on large
token sequences for linearized_ser) retains large object graphs not released
between pieces, causing ~11 GB/min committed-memory growth that OOM-killed the
process at 39 GB during Clair de Lune (908 KB MusicXML).

Follows the exact same inference subprocess-isolation pattern as
eval/run_lieder_eval.py — model loads in a child process, dies when it exits.

Usage:
    venv-cu132\\Scripts\\python -m eval.run_clarity_demo_eval \\
        --checkpoint checkpoints/full_radio_stage2/stage2-radio-polyphonic_best.pt \\
        --config configs/train_stage2_radio.yaml \\
        --name stage2_best

Then score:
    venv-cu132\\Scripts\\python -m eval.score_demo_eval \\
        --predictions-dir eval/results/clarity_demo_stage2_best \\
        --reference-dir data/clarity_demo/mxl \\
        --name stage2_best
"""
import argparse
import sys
from pathlib import Path

# Insert repo root so imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Path to our venv's Python — used to invoke src.pdf_to_musicxml as a subprocess
VENV_PYTHON = Path(__file__).resolve().parents[1] / "venv-cu132" / "Scripts" / "python.exe"

# Stage A YOLO checkpoint is shipped by sibling Clarity-OMR (HuggingFace download
# triggered on first omr.py run, lands at info/yolo.pt). We don't train Stage A
# in this repo, so reuse it.
STAGE_A_YOLO = Path.home() / "Clarity-OMR" / "info" / "yolo.pt"

# Inference timeout: 30 minutes per piece (same cap as lieder eval)
PDF_TO_MUSICXML_TIMEOUT_SEC = 1800

# Canonical demo pieces (stems match filenames under data/clarity_demo/{pdf,mxl}/)
# These are the 4 pieces showcased on huggingface.co/clquwu/Clarity-OMR
DEMO_STEMS = [
    "clair-de-lune-debussy",
    "fugue-no-2-bwv-847-in-c-minor",
    "gnossienne-no-1",
    "prelude-in-d-flat-major-op31-no1-scriabin",
]


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
    This is the key isolation that prevents the 86 GB committed-memory incident
    from recurring.
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
        description="Inference-only pass: run Stage B checkpoint on the 4 canonical "
                    "HF demo pieces and write predicted MusicXML files to disk. "
                    "Run eval/score_demo_eval.py afterwards to compute metrics."
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
        help="run name (used for output dir: eval/results/clarity_demo_<name>/)",
    )
    p.add_argument(
        "--beam-width", type=int, default=1,
        help="Stage-B beam width (default 1 = greedy; matches lieder eval default)",
    )
    p.add_argument(
        "--max-decode-steps", type=int, default=256,
        help="Stage-B max decode steps per staff (default 256; matches lieder eval default)",
    )
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"FATAL: checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"FATAL: config not found: {args.config}")

    print(f"Run name:   {args.name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(f"Pieces:     {len(DEMO_STEMS)}")
    print(f"Mode:       INFERENCE ONLY (no metric scoring)")
    print()

    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = (repo_root / "data/clarity_demo/pdf").resolve()
    out_dir = (repo_root / "eval/results" / f"clarity_demo_{args.name}").resolve()
    work_base = (repo_root / "eval/results" / f"clarity_demo_{args.name}_workdirs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(DEMO_STEMS)
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, stem in enumerate(DEMO_STEMS, 1):
        pdf = pdf_dir / f"{stem}.pdf"

        if not pdf.exists():
            print(f"[{i}/{n_total}] SKIP {stem}: PDF not found at {pdf}")
            n_skip += 1
            continue

        pred = out_dir / f"{stem}.musicxml"
        work_dir = work_base / stem

        try:
            if not pred.exists():
                print(f"[{i}/{n_total}] inference {stem} ...")
                run_inference(
                    args.checkpoint, args.config, pdf, pred, work_dir,
                    beam_width=args.beam_width,
                    max_decode_steps=args.max_decode_steps,
                )
                if pred.exists():
                    print(f"[{i}/{n_total}] done     {stem}  ({pred.stat().st_size // 1024} KB)")
                else:
                    print(f"[{i}/{n_total}] WARN     {stem}: inference completed but output not found")
            else:
                print(f"[{i}/{n_total}] cached   {stem}  ({pred.stat().st_size // 1024} KB)")
            n_ok += 1
        except Exception as e:
            print(f"[{i}/{n_total}] FAIL {stem}: {type(e).__name__}: {e}")
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
    print(f"  venv-cu132\\Scripts\\python -m eval.score_demo_eval \\")
    print(f"      --predictions-dir {out_dir} \\")
    print(f"      --reference-dir {(repo_root / 'data/clarity_demo/mxl').resolve()} \\")
    print(f"      --name {args.name}")


if __name__ == "__main__":
    main()
