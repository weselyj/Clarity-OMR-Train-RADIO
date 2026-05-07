"""Multi-staff system-level helpers for Stage 3 data prep.

Mirrors `src/data/yolo_aligned_crops.py` but operates on system bboxes
(one label per system, multiple staves per system) rather than per-staff.
"""
from __future__ import annotations

from pathlib import Path


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
