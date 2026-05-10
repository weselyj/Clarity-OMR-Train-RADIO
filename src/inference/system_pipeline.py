"""End-to-end per-system inference pipeline.

Library class that loads YOLO + Stage B once and exposes
`run_pdf` / `run_page` / `run_system_crop` / `export_musicxml`. See the
spec at docs/superpowers/specs/2026-05-10-radio-subproject4-design.md.

The class is designed so the eval driver can hold one instance for an
entire 50-piece run (Phase 1 inference only — scoring stays in subprocess).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from src.inference.checkpoint_load import StageBInferenceBundle, load_stage_b_for_inference
from src.inference.decoder_runtime import (
    _decode_stage_b_tokens,
    _encode_staff_image,
    _load_stage_b_crop_tensor,
)
from src.models.yolo_stage_a_systems import YoloStageASystems
from src.pipeline.assemble_score import (
    AssembledScore,
    StaffRecognitionResult,
    assemble_score_from_system_predictions,
)


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

    def run_system_crop(
        self,
        crop: Image.Image,
        system_index: int,
        system_location: dict,
    ) -> List[StaffRecognitionResult]:
        """Decode a single system crop and return per-staff
        StaffRecognitionResult objects."""
        tokens = self._decode_one_crop(crop)
        score = assemble_score_from_system_predictions(
            [tokens], [system_location],
        )
        # Flatten staves from the single-system AssembledScore.
        result: List[StaffRecognitionResult] = []
        for system in score.systems:
            for staff in system.staves:
                result.append(
                    StaffRecognitionResult(
                        sample_id=staff.sample_id,
                        tokens=staff.tokens,
                        location=staff.location,
                        system_index_hint=system.system_index,
                    )
                )
        return result

    def _decode_one_crop(self, crop: Image.Image) -> List[str]:
        """Save crop to a temp file, run encoder + decoder, return token list."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            crop.save(tmp_path)
        try:
            pixel_values = _load_stage_b_crop_tensor(
                tmp_path,
                image_height=self._image_height,
                image_max_width=self._image_max_width,
                device=self._device,
            )
            if self._bundle.use_fp16:
                pixel_values = pixel_values.half()
            memory = _encode_staff_image(self._bundle.decode_model, pixel_values)
            return _decode_stage_b_tokens(
                model=self._bundle.model,
                pixel_values=pixel_values,
                vocabulary=self._bundle.vocab,
                beam_width=self._beam_width,
                max_decode_steps=self._max_decode_steps,
                length_penalty_alpha=self._length_penalty_alpha,
                _precomputed={
                    "decode_model": self._bundle.decode_model,
                    "memory": memory,
                    "token_to_idx": self._bundle.token_to_idx,
                    "use_fp16": self._bundle.use_fp16,
                },
            )
        finally:
            tmp_path.unlink(missing_ok=True)
