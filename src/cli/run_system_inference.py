"""Thin CLI wrapping SystemInferencePipeline for one-off and smoke runs.

Usage:
    python -m src.cli.run_system_inference \\
        --pdf path/to/score.pdf \\
        --out out.musicxml \\
        --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt \\
        --stage-b-ckpt checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_system_inference",
        description="Per-system end-to-end inference: PDF -> MusicXML + diagnostics sidecar.",
    )
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output .musicxml path; .musicxml.diagnostics.json written alongside.")
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--stage-b-ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--max-decode-steps", type=int, default=2048)
    parser.add_argument("--page-dpi", type=int, default=300)
    parser.add_argument("--length-penalty-alpha", type=float, default=0.4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    from src.inference.system_pipeline import SystemInferencePipeline
    from src.pipeline.export_musicxml import StageDExportDiagnostics

    pipeline = SystemInferencePipeline(
        yolo_weights=args.yolo_weights,
        stage_b_ckpt=args.stage_b_ckpt,
        device=args.device,
        beam_width=args.beam_width,
        max_decode_steps=args.max_decode_steps,
        page_dpi=args.page_dpi,
        length_penalty_alpha=args.length_penalty_alpha,
        use_fp16=args.fp16,
        quantize=args.quantize,
    )
    diags = StageDExportDiagnostics()
    score = pipeline.run_pdf(args.pdf, diagnostics=diags)
    pipeline.export_musicxml(score, args.out, diagnostics=diags)
    print(f"wrote {args.out} + {args.out}.diagnostics.json")


if __name__ == "__main__":
    main()
