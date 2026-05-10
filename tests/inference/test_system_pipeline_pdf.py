"""Integration test: SystemInferencePipeline.run_pdf + export_musicxml."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_run_pdf_produces_assembled_score_from_pdf():
    """run_pdf opens PDF via PyMuPDF, runs run_page on each page, assembles."""
    from PIL import Image as _Image

    fake_bundle = MagicMock(use_fp16=False, vocab=MagicMock(tokens=[]))
    fake_yolo_instance = MagicMock()
    fake_yolo_instance.detect_systems.return_value = [
        {"system_index": 0, "bbox_extended": (0.0, 0.0, 100.0, 50.0), "conf": 0.9},
    ]
    fake_token_seq = [
        "<bos>",
        "<staff_start>", "<staff_idx_0>",
        "<measure_start>", "note-C4-quarter", "<measure_end>", "<staff_end>",
        "<eos>",
    ]

    fake_pixmap = MagicMock(width=1000, height=800)
    fake_pixmap.samples = bytes([255]) * (1000 * 800 * 3)
    fake_page = MagicMock()
    fake_page.get_pixmap.return_value = fake_pixmap
    fake_doc = MagicMock(__iter__=lambda self: iter([fake_page, fake_page]))
    fake_doc.__enter__ = MagicMock(return_value=fake_doc)
    fake_doc.__exit__ = MagicMock(return_value=False)

    with patch("src.inference.system_pipeline.YoloStageASystems",
               return_value=fake_yolo_instance), \
         patch("src.inference.system_pipeline.load_stage_b_for_inference",
               return_value=fake_bundle), \
         patch("src.inference.system_pipeline._load_stage_b_crop_tensor",
               return_value=MagicMock()), \
         patch("src.inference.system_pipeline._encode_staff_image",
               return_value=MagicMock()), \
         patch("src.inference.system_pipeline._decode_stage_b_tokens",
               return_value=fake_token_seq), \
         patch("src.inference.system_pipeline.fitz") as fake_fitz:

        fake_fitz.open.return_value = fake_doc

        from src.inference.system_pipeline import SystemInferencePipeline

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt", stage_b_ckpt="stage_b.pt", device="cpu",
        )
        score = pipeline.run_pdf("fake.pdf")

    total_staves = sum(len(s.staves) for s in score.systems)
    assert total_staves == 2
