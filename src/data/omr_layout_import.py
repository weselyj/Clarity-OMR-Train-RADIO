"""Convert omr-layout-analysis annotations to single-class YOLO labels.

The upstream dataset stores per-page custom JSON files with arrays-per-class
(staves, systems, stave_measures, system_measures, grand_staff) using absolute
pixel bbox coordinates [left, top, width, height]. Stage A is single-class —
this converter extracts only `staves`, normalizes to YOLO format, remaps to
class id 0, and writes one .txt label file per input JSON.

Pages with no `staves` entries produce NO output file (caller can detect
and skip the corresponding image).
"""
from __future__ import annotations

import json
from pathlib import Path


def convert_annotation_to_yolo(annotation_path: Path, out_label_path: Path) -> int:
    """Convert one omr-layout-analysis JSON to a YOLO .txt label file.

    Args:
        annotation_path: path to a per-page JSON annotation file
        out_label_path: target .txt file (created only if there are staves to write)

    Returns:
        Number of staff bboxes written. Zero when the page has no staves —
        in that case no output file is created.
    """
    data = json.loads(annotation_path.read_text())
    width = data["width"]
    height = data["height"]
    staves = data.get("staves", [])

    yolo_lines: list[str] = []
    for stave in staves:
        w = stave["width"]
        h = stave["height"]
        if w <= 0 or h <= 0:
            continue
        x_center = (stave["left"] + w / 2) / width
        y_center = (stave["top"] + h / 2) / height
        w_n = w / width
        h_n = h / height
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_n:.6f} {h_n:.6f}")

    if not yolo_lines:
        return 0

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.write_text("\n".join(yolo_lines) + "\n")
    return len(yolo_lines)
