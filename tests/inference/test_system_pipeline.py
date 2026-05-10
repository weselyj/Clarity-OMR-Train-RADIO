"""Unit tests for src.inference.system_pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_pipeline_init_loads_yolo_and_stage_b_once():
    """__init__ should construct YoloStageASystems and call
    load_stage_b_for_inference exactly once each."""
    with patch("src.inference.system_pipeline.YoloStageASystems") as fake_yolo, \
         patch("src.inference.system_pipeline.load_stage_b_for_inference") as fake_loader:

        fake_loader.return_value = MagicMock(use_fp16=False)
        from src.inference.system_pipeline import SystemInferencePipeline

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt",
            stage_b_ckpt="stage_b.pt",
            device="cpu",
        )

    fake_yolo.assert_called_once_with("yolo.pt")
    fake_loader.assert_called_once()
    # Pipeline must NOT call _prepare_model_for_inference directly — the loader does.
    assert pipeline is not None


def test_run_system_crop_decodes_via_bundle_and_returns_staves():
    """run_system_crop saves the crop, loads it via _load_stage_b_crop_tensor,
    runs encoder + decoder, and returns N StaffRecognitionResult objects."""
    from PIL import Image as _Image

    fake_bundle = MagicMock(use_fp16=False, vocab=MagicMock(tokens=[]))
    fake_token_seq = [
        "<bos>",
        "<staff_start>", "<staff_idx_0>",
        "<measure_start>", "note-C4-quarter", "<measure_end>", "<staff_end>",
        "<eos>",
    ]

    with patch("src.inference.system_pipeline.YoloStageASystems"), \
         patch("src.inference.system_pipeline.load_stage_b_for_inference",
               return_value=fake_bundle), \
         patch("src.inference.system_pipeline._load_stage_b_crop_tensor",
               return_value=MagicMock()) as fake_load_crop, \
         patch("src.inference.system_pipeline._encode_staff_image",
               return_value=MagicMock()) as fake_encode, \
         patch("src.inference.system_pipeline._decode_stage_b_tokens",
               return_value=fake_token_seq) as fake_decode:

        from src.inference.system_pipeline import SystemInferencePipeline

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt", stage_b_ckpt="stage_b.pt", device="cpu",
        )

        crop = _Image.new("RGB", (200, 100), color="white")
        sys_loc = {
            "system_index": 0,
            "bbox": (0.0, 0.0, 200.0, 100.0),
            "page_index": 0,
            "conf": 0.9,
        }
        staves = pipeline.run_system_crop(crop, system_index=0, system_location=sys_loc)

    assert len(staves) == 1
    assert staves[0].system_index_hint == 0
    fake_load_crop.assert_called_once()
    fake_encode.assert_called_once()
    fake_decode.assert_called_once()
