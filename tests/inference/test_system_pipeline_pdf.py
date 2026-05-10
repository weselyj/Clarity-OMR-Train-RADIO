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


def test_export_musicxml_writes_xml_and_diagnostics_sidecar(tmp_path):
    """export_musicxml writes both the .musicxml file and a
    .musicxml.diagnostics.json sidecar (matching the contract eval/_scoring_utils.py expects)."""
    fake_bundle = MagicMock(use_fp16=False, vocab=MagicMock(tokens=[]))

    fake_music_score = MagicMock()
    fake_music_score.write = MagicMock()

    with patch("src.inference.system_pipeline.YoloStageASystems"), \
         patch("src.inference.system_pipeline.load_stage_b_for_inference",
               return_value=fake_bundle), \
         patch("src.inference.system_pipeline.assembled_score_to_music21_with_diagnostics",
               return_value=fake_music_score) as fake_export:

        from src.inference.system_pipeline import SystemInferencePipeline
        from src.pipeline.export_musicxml import StageDExportDiagnostics

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt", stage_b_ckpt="stage_b.pt", device="cpu",
        )
        out_path = tmp_path / "out.musicxml"
        diags = StageDExportDiagnostics()
        diags.skipped_notes = 7
        fake_score = MagicMock()

        def _write_stub(_format, path):
            from pathlib import Path as _P
            _P(path).write_text("<score-partwise/>")
        fake_music_score.write.side_effect = _write_stub

        pipeline.export_musicxml(fake_score, out_path, diagnostics=diags)

    fake_export.assert_called_once_with(fake_score, diags, strict=False)
    assert out_path.exists()
    sidecar = out_path.with_suffix(out_path.suffix + ".diagnostics.json")
    assert sidecar.exists()

    import json
    payload = json.loads(sidecar.read_text())
    assert payload["skipped_notes"] == 7
