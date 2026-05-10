"""Shared YOLO geometry + adapter utilities.

Pure helpers reused by per-system Stage 3 data prep and the archived
per-staff variant. No per-staff or per-system assumptions — keep this
module limited to format-agnostic geometry and ultralytics adapters.
"""
from __future__ import annotations

from typing import Sequence

from PIL import Image


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU between two axis-aligned bboxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _yolo_predict_to_boxes(yolo_model, page_image: Image.Image, imgsz: int = 1920, conf: float = 0.25) -> list[dict]:
    """Adapter: ultralytics YOLO .predict() → list of {yolo_idx, bbox, conf} dicts."""
    results = yolo_model.predict(page_image, imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []
    r = results[0]
    xyxy = r.boxes.xyxy
    confs = r.boxes.conf
    # Tolerate either torch tensors or plain lists (latter is what tests pass).
    def to_list(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)
    xyxy_list = to_list(xyxy)
    conf_list = to_list(confs)
    out = []
    for i, (box, c) in enumerate(zip(xyxy_list, conf_list)):
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        out.append({"yolo_idx": i, "bbox": (x1, y1, x2, y2), "conf": float(c)})
    return out
