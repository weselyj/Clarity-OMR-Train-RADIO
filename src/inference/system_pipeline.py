"""End-to-end per-system inference pipeline.

Library class that loads YOLO + Stage B once and exposes
`run_pdf` / `run_page` / `run_system_crop` / `export_musicxml`. See the
spec at docs/superpowers/specs/2026-05-10-radio-subproject4-design.md.

The class is designed so the eval driver can hold one instance for an
entire 50-piece run (Phase 1 inference only — scoring stays in subprocess).
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
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
from src.pipeline.export_musicxml import (
    StageDExportDiagnostics,
    assembled_score_to_music21_with_diagnostics,
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
        yolo_conf: float = 0.25,
    ):
        self._device = torch.device(device)
        self._stage_a = YoloStageASystems(yolo_weights, conf=yolo_conf)
        self._bundle: StageBInferenceBundle = load_stage_b_for_inference(
            stage_b_ckpt, self._device, use_fp16=use_fp16, quantize=quantize,
        )
        self._beam_width = beam_width
        self._max_decode_steps = max_decode_steps
        self._page_dpi = page_dpi
        self._image_height = image_height
        self._image_max_width = image_max_width
        self._length_penalty_alpha = length_penalty_alpha

    def run_pdf(
        self,
        pdf_path,
        *,
        diagnostics=None,
        token_log: Optional[list] = None,
        postprocess: bool = True,
        postprocess_repair_bass_clef: bool = False,
    ) -> AssembledScore:
        """Render PDF pages, run Stage A + Stage B per page, assemble.

        When `diagnostics` is supplied, it is forwarded to
        `assemble_score_from_system_predictions`, which increments
        `skipped_systems` for each system whose decoder output failed
        token validation (decoder truncation). It is also forwarded to
        `export_musicxml` for Stage-D skip recording.

        When `token_log` is supplied (a list), one dict per system is
        appended to it before assembly with keys ``system_index``,
        ``page_index``, ``bbox``, ``conf``, ``tokens``, ``n_tokens``.
        Intended for debugging clef / staff-ordering failures by
        inspecting the raw decoder output prior to Stage D export.
        """
        from src.pipeline.post_decode import clean_system_tokens

        all_token_lists = []
        all_locations = []
        with fitz.open(str(pdf_path)) as doc:
            for page_index, page in enumerate(doc):
                pix = page.get_pixmap(dpi=self._page_dpi)
                img = Image.frombytes(
                    "RGB", (pix.width, pix.height), pix.samples,
                )
                systems = self._stage_a.detect_systems(img)
                for sys in systems:
                    x1, y1, x2, y2 = sys["bbox_extended"]
                    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                    raw_tokens = self._decode_one_crop(crop)
                    tokens = (
                        clean_system_tokens(
                            raw_tokens,
                            repair_bass_clef=postprocess_repair_bass_clef,
                        )
                        if postprocess
                        else raw_tokens
                    )
                    all_token_lists.append(tokens)
                    all_locations.append({
                        "system_index": sys["system_index"],
                        "bbox": sys["bbox_extended"],
                        "page_index": page_index,
                        "conf": sys["conf"],
                    })
                    if token_log is not None:
                        token_log.append({
                            "system_index": int(sys["system_index"]),
                            "page_index": int(page_index),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "conf": float(sys["conf"]),
                            "tokens": list(tokens),
                            "raw_tokens": list(raw_tokens) if postprocess else None,
                            "n_tokens": len(tokens),
                        })
        return assemble_score_from_system_predictions(
            all_token_lists, all_locations, diagnostics=diagnostics,
        )

    def run_image(
        self,
        image,
        *,
        diagnostics=None,
        token_log: Optional[list] = None,
        postprocess: bool = True,
        postprocess_repair_bass_clef: bool = False,
    ) -> AssembledScore:
        """Run Stage A + Stage B on a single page image, assemble.

        Mirrors ``run_pdf`` but takes one already-rendered page image instead of
        a PDF. Useful for direct image input (scans, single-page JPG/PNG/etc.)
        without converting to PDF first. ``image`` may be a path
        (``str``/``Path``) or a PIL ``Image.Image``; the file extension is not
        checked, so any format PIL can decode works.

        ``token_log`` (optional list) is appended with one dict per system in
        the same shape as ``run_pdf`` — see that method for the schema.
        """
        from src.pipeline.post_decode import clean_system_tokens

        if isinstance(image, Image.Image):
            img = image.convert("RGB") if image.mode != "RGB" else image
        else:
            img = Image.open(str(image)).convert("RGB")
        all_token_lists = []
        all_locations = []
        systems = self._stage_a.detect_systems(img)
        for sys in systems:
            x1, y1, x2, y2 = sys["bbox_extended"]
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            raw_tokens = self._decode_one_crop(crop)
            tokens = (
                clean_system_tokens(
                    raw_tokens,
                    repair_bass_clef=postprocess_repair_bass_clef,
                )
                if postprocess
                else raw_tokens
            )
            all_token_lists.append(tokens)
            all_locations.append({
                "system_index": sys["system_index"],
                "bbox": sys["bbox_extended"],
                "page_index": 0,
                "conf": sys["conf"],
            })
            if token_log is not None:
                token_log.append({
                    "system_index": int(sys["system_index"]),
                    "page_index": 0,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(sys["conf"]),
                    "tokens": list(tokens),
                    "raw_tokens": list(raw_tokens) if postprocess else None,
                    "n_tokens": len(tokens),
                })
        return assemble_score_from_system_predictions(
            all_token_lists, all_locations, diagnostics=diagnostics,
        )

    def run_page(
        self,
        page_image: Image.Image,
        page_index: int = 0,
    ) -> List[StaffRecognitionResult]:
        """Detect systems on the page, decode each, return flat staves list."""
        systems = self._stage_a.detect_systems(page_image)
        all_staves: List[StaffRecognitionResult] = []
        for sys in systems:
            x1, y1, x2, y2 = sys["bbox_extended"]
            crop = page_image.crop((int(x1), int(y1), int(x2), int(y2)))
            sys_loc = {
                "system_index": sys["system_index"],
                "bbox": sys["bbox_extended"],
                "page_index": page_index,
                "conf": sys["conf"],
            }
            staves = self.run_system_crop(crop, sys["system_index"], sys_loc)
            all_staves.extend(staves)
        return all_staves

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
        # Flatten staves from the single-system AssembledScore. Use the caller's
        # system_index — assemble_score() re-numbers systems sequentially via
        # enumerate(), losing the per-page identity we need to round-trip.
        result: List[StaffRecognitionResult] = []
        for system in score.systems:
            for staff in system.staves:
                result.append(
                    StaffRecognitionResult(
                        sample_id=staff.sample_id,
                        tokens=staff.tokens,
                        location=staff.location,
                        system_index_hint=system_index,
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

    def export_musicxml(
        self,
        score: AssembledScore,
        out_path,
        *,
        diagnostics: Optional[StageDExportDiagnostics] = None,
    ) -> None:
        """Write the predicted MusicXML and the .diagnostics.json sidecar.

        Mirrors archive/per_staff/src/pdf_to_musicxml.py:395-465 (the
        production export pattern), but does not include the lenient
        re-export fallback. If music21.write fails, the exception
        propagates — the eval driver wraps individual pieces in try/except.
        """
        if diagnostics is None:
            diagnostics = StageDExportDiagnostics()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        music_score = assembled_score_to_music21_with_diagnostics(
            score, diagnostics, strict=False,
        )
        music_score.write("musicxml", str(out_path))

        diag_path = out_path.with_suffix(out_path.suffix + ".diagnostics.json")
        diag_dict = dataclasses.asdict(diagnostics)
        diag_path.write_text(json.dumps(diag_dict, indent=2), encoding="utf-8")
