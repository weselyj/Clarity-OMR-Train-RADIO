#!/usr/bin/env python3
"""Predict MusicXML for a single PDF using a trained Stage A + Stage B checkpoint.

Runs the full SystemInferencePipeline (YOLO system detection -> Stage B
seq2seq decode per system -> assembly -> Stage D MusicXML export) on one PDF
and writes the resulting MusicXML to the path you choose. Defaults to the
latest Stage 3 v3 checkpoint and the standard YOLO weights so the common case
is a one-line invocation.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.predict_pdf \\
        data\\scratch\\score.pdf \\
        data\\scratch\\score.musicxml

With overrides:
    venv-cu132\\Scripts\\python -m scripts.predict_pdf \\
        my.pdf out.musicxml \\
        --stage-b-ckpt checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --beam-width 5

Optional --diagnostics-out writes the Stage D export diagnostics JSON alongside
the MusicXML, useful for inspecting skipped notes / chords / measures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_STAGE_B_CKPT = (
    "checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt"
)
_DEFAULT_YOLO_WEIGHTS = "runs/detect/runs/yolo26m_systems/weights/best.pt"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdf", type=Path, help="Input PDF to transcribe.")
    p.add_argument("out", type=Path, help="Output MusicXML path (parent dir auto-created).")
    p.add_argument(
        "--stage-b-ckpt", type=Path, default=Path(_DEFAULT_STAGE_B_CKPT),
        help=f"Path to Stage B checkpoint (.pt). Default: {_DEFAULT_STAGE_B_CKPT}",
    )
    p.add_argument(
        "--yolo-weights", type=Path, default=Path(_DEFAULT_YOLO_WEIGHTS),
        help=f"Path to YOLO system-detection weights. Default: {_DEFAULT_YOLO_WEIGHTS}",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="Device for Stage B inference. Default: cuda.")
    p.add_argument("--beam-width", type=int, default=1,
                   help="Stage B beam width. Default 1 (greedy). 5 is ~5x slower per beam.")
    p.add_argument("--max-decode-steps", type=int, default=2048,
                   help="Max decode steps per system crop. Default 2048.")
    p.add_argument("--page-dpi", type=int, default=300,
                   help="PDF render DPI for Stage A. Default 300 (matches training).")
    p.add_argument("--fp16", action="store_true",
                   help="Use fp16 for Stage B inference (small accuracy risk, faster).")
    p.add_argument(
        "--diagnostics-out", type=Path, default=None,
        help="Optional path to write Stage D export diagnostics JSON sidecar.",
    )
    args = p.parse_args()

    if not args.pdf.is_file():
        p.error(f"PDF not found: {args.pdf}")
    if not args.stage_b_ckpt.is_file():
        p.error(
            f"Stage B checkpoint not found: {args.stage_b_ckpt}. "
            f"Pass --stage-b-ckpt to override or train Stage 3 first."
        )
    if not args.yolo_weights.is_file():
        p.error(
            f"YOLO weights not found: {args.yolo_weights}. "
            f"Pass --yolo-weights to override or train Stage A first."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"PDF:             {args.pdf}")
    print(f"Output MusicXML: {args.out}")
    print(f"Stage B ckpt:    {args.stage_b_ckpt}")
    print(f"YOLO weights:    {args.yolo_weights}")
    print(f"Device:          {args.device}  fp16={args.fp16}")
    print(f"Beam:            {args.beam_width}  max_decode_steps={args.max_decode_steps}")
    print()

    from src.inference.system_pipeline import SystemInferencePipeline
    from src.pipeline.export_musicxml import StageDExportDiagnostics

    print("Loading pipeline ...")
    t0 = time.time()
    pipeline = SystemInferencePipeline(
        yolo_weights=args.yolo_weights,
        stage_b_ckpt=args.stage_b_ckpt,
        device=args.device,
        beam_width=args.beam_width,
        max_decode_steps=args.max_decode_steps,
        page_dpi=args.page_dpi,
        use_fp16=args.fp16,
    )
    print(f"  ready in {time.time() - t0:.1f}s")

    diags = StageDExportDiagnostics()
    print("Running inference ...")
    t1 = time.time()
    score = pipeline.run_pdf(args.pdf, diagnostics=diags)
    n_staves = sum(len(system.staves) for system in score.systems)
    print(f"  decoded {len(score.systems)} systems, {n_staves} staves total "
          f"in {time.time() - t1:.1f}s")

    print(f"Exporting MusicXML to {args.out} ...")
    t2 = time.time()
    pipeline.export_musicxml(score, args.out, diagnostics=diags)
    size_kb = args.out.stat().st_size // 1024 if args.out.exists() else 0
    print(f"  wrote {size_kb} KB in {time.time() - t2:.1f}s")

    if args.diagnostics_out is not None:
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        # StageDExportDiagnostics is a dataclass-like object; serialize via vars()
        diag_payload = {k: v for k, v in vars(diags).items() if not k.startswith("_")}
        args.diagnostics_out.write_text(json.dumps(diag_payload, indent=2, default=str), encoding="utf-8")
        print(f"  diagnostics JSON: {args.diagnostics_out}")

    total = time.time() - t0
    print()
    print(f"Done in {total:.1f}s total. Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
