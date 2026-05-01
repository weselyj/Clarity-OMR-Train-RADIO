"""YOLO-aligned crop extraction for RADIO Stage 3 training.

Replaces oracle Verovio bboxes with YOLO predictions to close the train/eval
distribution mismatch on staff crops. Per the design spec
docs/superpowers/specs/2026-05-01-radio-stage3-yolo-aligned-design.md, the
α-policy applies: staves YOLO misses are dropped from training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


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


def match_yolo_to_oracle(
    yolo_boxes: Iterable[dict],
    oracle_staves: Iterable[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Match each YOLO prediction to its best-IoU oracle staff.

    α-policy: drop YOLO boxes that don't reach `iou_threshold` against any
    oracle (false positives) and oracle staves that no YOLO box matches
    (recall gaps).

    When two YOLO boxes match the same oracle, keep the higher-confidence one.

    Each yolo_box dict must contain: yolo_idx, bbox (x1,y1,x2,y2), conf.
    Each oracle dict must contain: staff_index, bbox (x1,y1,x2,y2).

    Returns a list of matches, each: {yolo_idx, staff_index, conf, iou,
    yolo_bbox, oracle_bbox}.
    """
    yolo_list = list(yolo_boxes)
    oracle_list = list(oracle_staves)
    candidates = []
    for y in yolo_list:
        best_oracle = None
        best_iou = 0.0
        for o in oracle_list:
            i = iou_xyxy(y["bbox"], o["bbox"])
            if i > best_iou:
                best_iou = i
                best_oracle = o
        if best_oracle is not None and best_iou >= iou_threshold:
            candidates.append({
                "yolo_idx": y["yolo_idx"],
                "staff_index": best_oracle["staff_index"],
                "conf": y["conf"],
                "iou": best_iou,
                "yolo_bbox": y["bbox"],
                "oracle_bbox": best_oracle["bbox"],
            })
    # Resolve duplicates: when multiple YOLO boxes match the same oracle, keep highest conf.
    by_oracle: dict[int, dict] = {}
    for c in candidates:
        sid = c["staff_index"]
        if sid not in by_oracle or c["conf"] > by_oracle[sid]["conf"]:
            by_oracle[sid] = c
    return sorted(by_oracle.values(), key=lambda c: c["staff_index"])


def load_oracle_bboxes_from_yolo_label(
    label_path: Path, page_width: int, page_height: int
) -> list[dict]:
    """Read a YOLO-format label file and return oracle bboxes in pixel xyxy.

    YOLO format: each line is `class cx cy w h`, all normalized to [0,1].
    `staff_index` is assigned by sorting on y_center top→bottom (matches
    the convention used by synthetic_token_manifest.jsonl).
    """
    rows = []
    text = Path(label_path).read_text() if Path(label_path).exists() else ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # class_id, cx, cy, w, h
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2) * page_width
        y1 = (cy - h / 2) * page_height
        x2 = (cx + w / 2) * page_width
        y2 = (cy + h / 2) * page_height
        rows.append({"y_center": cy, "bbox": (x1, y1, x2, y2)})
    rows.sort(key=lambda r: r["y_center"])
    return [{"staff_index": i, "bbox": r["bbox"]} for i, r in enumerate(rows)]
