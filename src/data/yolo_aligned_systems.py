"""Multi-staff system-level helpers for Stage 3 data prep.

Mirrors `src/data/yolo_aligned_crops.py` but operates on system bboxes
(one label per system, multiple staves per system) rather than per-staff.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.data.yolo_aligned_crops import iou_xyxy  # reuse


def load_oracle_system_bboxes(
    label_path: Path, page_width: int, page_height: int
) -> list[dict]:
    """Read YOLO-format system label file → list of `{system_index, bbox}` dicts.

    YOLO format: each line is `class cx cy w h`, normalized to [0, 1].
    Output sorted top-to-bottom by y_center (matches the convention used by
    the companion `<page>.staves.json` file).
    """
    rows: list[dict] = []
    text = Path(label_path).read_text() if Path(label_path).exists() else ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2) * page_width
        y1 = (cy - h / 2) * page_height
        x2 = (cx + w / 2) * page_width
        y2 = (cy + h / 2) * page_height
        rows.append({"y_center": cy, "bbox": (x1, y1, x2, y2)})
    rows.sort(key=lambda r: r["y_center"])
    return [{"system_index": i, "bbox": r["bbox"]} for i, r in enumerate(rows)]


def load_staves_per_system(json_path: Path) -> list[int]:
    """Read the companion `<page>.staves.json` → list of integers (staves per system)."""
    p = Path(json_path)
    if not p.exists():
        return []
    return list(json.loads(p.read_text()))


def staff_indices_for_system(system_index: int, staves_per_system: list[int]) -> list[int]:
    """Return the per-staff indices that fall within `system_index`.

    Per-staff indices are page-global, top-to-bottom. Cumulative sum of
    `staves_per_system` gives each system's starting staff index.
    """
    if system_index < 0 or system_index >= len(staves_per_system):
        return []
    start = sum(staves_per_system[:system_index])
    count = staves_per_system[system_index]
    return list(range(start, start + count))


def match_yolo_to_oracle_systems(
    yolo_boxes: Iterable[dict],
    oracle_systems: Iterable[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Match each YOLO system prediction to its best-IoU oracle system.

    α-policy: drop YOLO boxes that don't reach `iou_threshold` against any
    oracle (false positives) and oracle systems that no YOLO box matches
    (recall gaps). When two YOLO boxes match the same oracle, keep the
    higher-confidence one.

    Each yolo_box dict must contain: yolo_idx, bbox (x1,y1,x2,y2), conf.
    Each oracle dict must contain: system_index, bbox.

    Returns: sorted list of `{yolo_idx, system_index, conf, iou, yolo_bbox, oracle_bbox}`.
    """
    yolo_list = list(yolo_boxes)
    oracle_list = list(oracle_systems)
    candidates: list[dict] = []
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
                "system_index": best_oracle["system_index"],
                "conf": y["conf"],
                "iou": best_iou,
                "yolo_bbox": y["bbox"],
                "oracle_bbox": best_oracle["bbox"],
            })
    by_oracle: dict[int, dict] = {}
    for c in candidates:
        sid = c["system_index"]
        if sid not in by_oracle or c["conf"] > by_oracle[sid]["conf"]:
            by_oracle[sid] = c
    return sorted(by_oracle.values(), key=lambda c: c["system_index"])
