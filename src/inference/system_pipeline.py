"""End-to-end per-system inference pipeline.

Library class that loads YOLO + Stage B once and exposes
`run_pdf` / `run_page` / `run_system_crop` / `export_musicxml`. See the
spec at docs/superpowers/specs/2026-05-10-radio-subproject4-design.md.

The class is designed so the eval driver can hold one instance for an
entire 50-piece run (Phase 1 inference only — scoring stays in subprocess).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from src.inference.checkpoint_load import StageBInferenceBundle, load_stage_b_for_inference
from src.models.yolo_stage_a_systems import YoloStageASystems


class SystemInferencePipeline:
    def __init__(
        self,
        yolo_weights,
        stage_b_ckpt,
        *,
        device: str = "cuda",
        beam_width: int = 1,
        max_decode_steps: int = 2048,
        page_dpi: int = 300,
        image_height: int = 250,
        image_max_width: int = 2500,
        length_penalty_alpha: float = 0.4,
        use_fp16: bool = False,
        quantize: bool = False,
    ):
        self._device = torch.device(device)
        self._stage_a = YoloStageASystems(yolo_weights)
        self._bundle: StageBInferenceBundle = load_stage_b_for_inference(
            stage_b_ckpt, self._device, use_fp16=use_fp16, quantize=quantize,
        )
        self._beam_width = beam_width
        self._max_decode_steps = max_decode_steps
        self._page_dpi = page_dpi
        self._image_height = image_height
        self._image_max_width = image_max_width
        self._length_penalty_alpha = length_penalty_alpha
