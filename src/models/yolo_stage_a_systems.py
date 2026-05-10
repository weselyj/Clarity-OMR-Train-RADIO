"""System-level Stage A YOLO wrapper for inference.

Loads the system-detection YOLO checkpoint, runs it on a page image, and
returns sorted top-to-bottom system bboxes with the brace-margin extension
already applied. Designed for inference only; training-time data prep uses
src/data/yolo_aligned_systems.py with oracle matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from src.data.yolo_common import _yolo_predict_to_boxes
from src.models.system_postprocess import extend_left_for_brace

# YOLO is imported lazily in __init__ to avoid a torchvision/torch NMS
# registration crash that occurs on module import in CPU-only environments.
# The module-level name exists so that tests can patch it via
#   patch("src.models.yolo_stage_a_systems.YOLO", ...).
YOLO = None


def _get_yolo():
    """Return the YOLO class, importing ultralytics on first call."""
    global YOLO  # noqa: PLW0603
    if YOLO is None:
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO
    return YOLO


class YoloStageASystems:
    def __init__(self, weights_path: Path, *, conf: float = 0.25, imgsz: int = 1920):
        yolo_cls = YOLO if YOLO is not None else _get_yolo()
        self._model = yolo_cls(str(weights_path))
        self._conf = conf
        self._imgsz = imgsz

    def detect_systems(self, page_image: Image.Image) -> List[dict]:
        """Run YOLO on the page, sort top-to-bottom, apply brace margin extension.

        Returns: list of {system_index, bbox_extended, conf} dicts.
        bbox_extended is a 4-tuple (x1, y1, x2, y2). Sorted top-to-bottom by
        y_center; system_index is assigned 0..N-1 after sorting.
        """
        raw = _yolo_predict_to_boxes(
            self._model, page_image, imgsz=self._imgsz, conf=self._conf,
        )
        if not raw:
            return []

        # Sort top-to-bottom by y_center.
        raw.sort(key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2)

        # Apply brace margin extension in one batch.
        boxes_array = np.array([b["bbox"] for b in raw], dtype=np.float64)
        extended = extend_left_for_brace(boxes_array, page_w=page_image.width)

        return [
            {
                "system_index": idx,
                "bbox_extended": tuple(float(v) for v in extended[idx]),
                "conf": float(raw[idx]["conf"]),
            }
            for idx in range(len(raw))
        ]
