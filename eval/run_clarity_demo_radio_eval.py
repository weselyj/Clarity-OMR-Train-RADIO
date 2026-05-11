"""Run the 4-piece upstream demo eval against a Stage 3 RADIO system-level
checkpoint, then score each prediction with eval/upstream_eval.py (the
upstream Clarity-OMR eval.py vendored verbatim).

The 4 pieces are the ones showcased on huggingface.co/clquwu/Clarity-OMR.
This replaces the archived per-staff archive/per_staff/eval/run_clarity_demo_eval.py
which targeted the legacy `src.pdf_to_musicxml` entrypoint.

Single-process by design: 4 pieces is small enough that loading the pipeline
once and looping is fine. Scoring is invoked as a subprocess per piece so
music21 state from one piece can't pollute the next.

Outputs (under eval/results/<--name>/):
    <stem>.musicxml + <stem>.musicxml.diagnostics.json   (inference pass)
    <stem>.score.json                                    (scoring pass)
    summary.json                                         (aggregate)

Example:
    venv-cu132\\Scripts\\python -m eval.run_clarity_demo_radio_eval \\
        --stage-b-ckpt checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt \\
        --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt \\
        --name stage3_v2_best
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Canonical demo stems -- match the upstream HuggingFace model card.
DEMO_STEMS = [
    "clair-de-lune-debussy",
    "fugue-no-2-bwv-847-in-c-minor",
    "gnossienne-no-1",
    "prelude-in-d-flat-major-op31-no1-scriabin",
]


def _run_inference(pipeline, pdf: Path, out: Path) -> dict:
    """Run inference for one piece and write MusicXML + diagnostics sidecar.

    Returns a small dict with timing + size info; never raises (failures are
    caught and recorded so the loop continues to the next piece).
    """
    from src.pipeline.export_musicxml import StageDExportDiagnostics

    t0 = time.time()
    try:
        diags = StageDExportDiagnostics()
        score = pipeline.run_pdf(pdf, diagnostics=diags)
        pipeline.export_musicxml(score, out, diagnostics=diags)
        dt = time.time() - t0
        size_kb = out.stat().st_size // 1024 if out.exists() else 0
        return {"ok": True, "seconds": dt, "size_kb": size_kb, "error": None}
    except Exception as exc:  # pragma: no cover - best-effort wrapper
        return {
            "ok": False,
            "seconds": time.time() - t0,
            "size_kb": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _score_piece(python_exe: Path, ref: Path, pred: Path, json_out: Path) -> dict:
    """Invoke eval/upstream_eval.py in a subprocess. Returns the parsed JSON
    of the first (only) candidate, or an `{"error": ...}` dict on failure."""
    cmd = [
        str(python_exe),
        "-m", "eval.upstream_eval",
        str(ref),
        str(pred),
        "--json", str(json_out),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "scoring timed out after 120s"}

    if proc.returncode != 0:
        return {"error": f"scoring rc={proc.returncode}: {proc.stderr.strip()[:400]}"}

    if not json_out.exists():
        return {"error": f"scoring produced no JSON output (stderr: {proc.stderr.strip()[:400]})"}

    try:
        payload = json.loads(json_out.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"failed to parse {json_out}: {exc}"}

    candidates = payload.get("candidates") or []
    if not candidates:
        return {"error": "scoring JSON had no candidates entry"}
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run system-level inference + upstream mir_eval scoring "
                    "on the 4 canonical Clarity-OMR demo pieces.",
    )
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--yolo-weights", type=Path, required=True)
    p.add_argument("--name", required=True, help="Run name (output dir suffix).")
    p.add_argument("--pdf-dir", type=Path, default=_REPO_ROOT / "data/clarity_demo/pdf")
    p.add_argument("--mxl-dir", type=Path, default=_REPO_ROOT / "data/clarity_demo/mxl")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: eval/results/clarity_demo_<name>).")
    p.add_argument("--beam-width", type=int, default=1)
    p.add_argument("--max-decode-steps", type=int, default=2048)
    p.add_argument("--page-dpi", type=int, default=300)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--python", type=Path, default=None,
                   help="Python executable for scoring subprocesses. Defaults to current.")
    p.add_argument("--stems", default=None,
                   help="Comma-separated subset of DEMO_STEMS to run (debug aid).")
    args = p.parse_args()

    global DEMO_STEMS
    if args.stems:
        wanted = {s.strip() for s in args.stems.split(",") if s.strip()}
        DEMO_STEMS = [s for s in DEMO_STEMS if s in wanted]
        if not DEMO_STEMS:
            raise SystemExit(f"FATAL: --stems matched no DEMO_STEMS (wanted={wanted})")

    for path in (args.stage_b_ckpt, args.yolo_weights, args.pdf_dir, args.mxl_dir):
        if not path.exists():
            raise SystemExit(f"FATAL: not found: {path}")

    out_dir = args.out_dir or (_REPO_ROOT / "eval/results" / f"clarity_demo_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    python_exe = args.python or Path(sys.executable)

    print(f"Run name:       {args.name}")
    print(f"Stage B ckpt:   {args.stage_b_ckpt}")
    print(f"YOLO weights:   {args.yolo_weights}")
    print(f"PDF dir:        {args.pdf_dir}")
    print(f"Ref MXL dir:    {args.mxl_dir}")
    print(f"Out dir:        {out_dir}")
    print(f"Device:         {args.device}  fp16={args.fp16}")
    print(f"Beam:           {args.beam_width}   max_steps={args.max_decode_steps}")
    print(f"Score python:   {python_exe}")
    print()

    # ---- Inference pass --------------------------------------------------
    print("== Inference pass ==")
    from src.inference.system_pipeline import SystemInferencePipeline

    t_pipe_load = time.time()
    pipeline = SystemInferencePipeline(
        yolo_weights=args.yolo_weights,
        stage_b_ckpt=args.stage_b_ckpt,
        device=args.device,
        beam_width=args.beam_width,
        max_decode_steps=args.max_decode_steps,
        page_dpi=args.page_dpi,
        use_fp16=args.fp16,
    )
    print(f"Pipeline loaded in {time.time() - t_pipe_load:.1f}s")
    print()

    per_piece: dict[str, dict] = {}
    for i, stem in enumerate(DEMO_STEMS, 1):
        pdf = args.pdf_dir / f"{stem}.pdf"
        ref = args.mxl_dir / f"{stem}.mxl"
        pred = out_dir / f"{stem}.musicxml"
        score_json = out_dir / f"{stem}.score.json"

        entry: dict = {"stem": stem, "pdf": str(pdf), "ref": str(ref), "pred": str(pred)}
        per_piece[stem] = entry

        if not pdf.exists():
            print(f"[{i}/4] SKIP {stem}: PDF not found at {pdf}")
            entry["inference"] = {"ok": False, "error": "pdf_missing"}
            continue
        if not ref.exists():
            print(f"[{i}/4] SKIP {stem}: reference MXL not found at {ref}")
            entry["inference"] = {"ok": False, "error": "ref_missing"}
            continue

        if pred.exists():
            print(f"[{i}/4] cached inference: {stem} ({pred.stat().st_size // 1024} KB)")
            entry["inference"] = {"ok": True, "cached": True, "size_kb": pred.stat().st_size // 1024}
        else:
            print(f"[{i}/4] inference: {stem} ...", flush=True)
            inf = _run_inference(pipeline, pdf, pred)
            entry["inference"] = inf
            if inf["ok"]:
                print(f"      done in {inf['seconds']:.1f}s  ({inf['size_kb']} KB)")
            else:
                print(f"      FAILED: {inf['error']}")
                continue

        print(f"[{i}/4] scoring:   {stem} ...", flush=True)
        score = _score_piece(python_exe, ref, pred, score_json)
        entry["score"] = score
        if score.get("error") is not None:
            print(f"      FAILED: {score['error']}")
        else:
            print(
                f"      onset_f1={score.get('onset_f1', float('nan')):.4f}   "
                f"f1={score.get('f1', float('nan')):.4f}   "
                f"overlap={score.get('overlap', float('nan')):.4f}   "
                f"quality={score.get('quality_score', float('nan')):.1f}/100"
            )

    # ---- Aggregate -------------------------------------------------------
    rows = []
    for stem in DEMO_STEMS:
        e = per_piece.get(stem, {})
        score = e.get("score", {})
        if score.get("error") is not None or "onset_f1" not in score:
            rows.append((stem, None, None, None, None))
            continue
        rows.append((
            stem,
            score.get("onset_f1"),
            score.get("f1"),
            score.get("overlap"),
            score.get("quality_score"),
        ))

    print()
    print("== Summary ==")
    print(f"{'piece':50s}  onset_f1   note_f1    overlap   quality")
    for stem, of1, f1, ov, q in rows:
        if of1 is None:
            print(f"{stem:50s}  --        --         --        --")
        else:
            print(f"{stem:50s}  {of1:.4f}   {f1:.4f}    {ov:.4f}   {q:5.1f}")

    valid = [(of1, f1, ov, q) for _, of1, f1, ov, q in rows if of1 is not None]
    if valid:
        of1s, f1s, ovs, qs = zip(*valid)
        print()
        print(f"mean onset_f1: {statistics.mean(of1s):.4f}   "
              f"mean note_f1: {statistics.mean(f1s):.4f}   "
              f"mean overlap: {statistics.mean(ovs):.4f}   "
              f"mean quality: {statistics.mean(qs):.1f}")

    # Persist a summary JSON next to predictions
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "name": args.name,
        "stage_b_ckpt": str(args.stage_b_ckpt),
        "yolo_weights": str(args.yolo_weights),
        "pieces": per_piece,
    }, indent=2), encoding="utf-8")
    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()
