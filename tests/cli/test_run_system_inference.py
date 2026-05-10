"""argparse smoke for src.cli.run_system_inference."""
from __future__ import annotations


def test_argparser_does_not_require_vocab_flag():
    """The CLI must NOT require a --vocab flag — vocabulary is built
    in-code via build_default_vocabulary()."""
    from src.cli.run_system_inference import build_argparser

    parser = build_argparser()
    args = parser.parse_args([
        "--pdf", "x.pdf",
        "--out", "out.musicxml",
        "--yolo-weights", "yolo.pt",
        "--stage-b-ckpt", "stage_b.pt",
    ])
    assert str(args.pdf) == "x.pdf"
    assert str(args.out) == "out.musicxml"


def test_argparser_optional_flags_have_defaults():
    from src.cli.run_system_inference import build_argparser

    parser = build_argparser()
    args = parser.parse_args([
        "--pdf", "x.pdf", "--out", "out.musicxml",
        "--yolo-weights", "yolo.pt", "--stage-b-ckpt", "stage_b.pt",
    ])
    assert args.device == "cuda"
    assert args.beam_width == 1
    assert args.max_decode_steps == 2048
    assert args.page_dpi == 300
    assert args.length_penalty_alpha == 0.4
    assert args.fp16 is False
    assert args.quantize is False
