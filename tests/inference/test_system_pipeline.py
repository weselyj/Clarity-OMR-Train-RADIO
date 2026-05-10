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
