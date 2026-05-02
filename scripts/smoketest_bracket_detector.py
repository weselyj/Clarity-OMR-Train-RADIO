"""Smoke-test bracket detector + grouping on a few known pages.

Prints detected brackets, then runs grouping using existing per-staff labels,
and prints the resulting system grouping. Verifies the detector + grouping
work end-to-end before integrating into the full derivation pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.bracket_detector import (  # noqa: E402
    detect_brackets_on_page,
    group_staves_by_brackets,
)


CASES = [
    {
        "name": "sparse_augment lc5079502 p5 (3 brackets x 3 staves expected)",
        "image": Path("data/processed/sparse_augment/images/dpi300/leipzig-default/data__processed__sparse_augment__mxl__lc5079502_mbrest.musicxml__leipzig-default__p005.png"),
        "labels": Path("data/processed/sparse_augment/labels/leipzig-default/data__processed__sparse_augment__mxl__lc5079502_mbrest.musicxml__leipzig-default__p005.txt"),
    },
    {
        "name": "AudioLabs Schubert D911-04 p4 (5+ systems x 3 staves expected)",
        "image": Path("data/processed/omr_layout_real/images/Schubert_D911-04__Schubert_D911-04_004.png"),
        "labels": Path("data/processed/omr_layout_real/labels/Schubert_D911-04__Schubert_D911-04_004.txt"),
    },
    {
        "name": "AudioLabs Wagner WWV086A p2 (4 systems expected)",
        "image": Path("data/processed/omr_layout_real/images/Wagner_WWV086A__Wagner_WWV086A_002.png"),
        "labels": Path("data/processed/omr_layout_real/labels/Wagner_WWV086A__Wagner_WWV086A_002.png"),
    },
]


def yolo_to_pixel_staves(label_path: Path, page_w: int, page_h: int) -> list:
    out = []
    if not label_path.exists():
        # try .txt if image suffix was used
        if label_path.suffix != ".txt":
            label_path = label_path.with_suffix(".txt")
            if not label_path.exists():
                return out
        else:
            return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2) * page_w
        y1 = (cy - h / 2) * page_h
        x2 = (cx + w / 2) * page_w
        y2 = (cy + h / 2) * page_h
        out.append({"bbox": (x1, y1, x2, y2)})
    return out


def main() -> int:
    from PIL import Image

    for case in CASES:
        print("=" * 80)
        print(case["name"])
        print("=" * 80)
        image_path = case["image"]
        if not image_path.exists():
            # Try jpg
            for ext in (".jpg", ".jpeg"):
                alt = image_path.with_suffix(ext)
                if alt.exists():
                    image_path = alt
                    break
        if not image_path.exists():
            print(f"  SKIP: image not found at {case['image']}")
            continue
        with Image.open(image_path) as img:
            page_w, page_h = img.size
        print(f"  Image: {image_path.name} ({page_w}x{page_h})")

        brackets = detect_brackets_on_page(image_path)
        print(f"  Detected {len(brackets)} brackets:")
        for i, b in enumerate(brackets):
            print(f"    {i}: x={b['x']:.0f}, y_top={b['y_top']:.0f}, y_bot={b['y_bottom']:.0f}, h={b['height']:.0f}")

        label_path = case["labels"] if case["labels"].suffix == ".txt" else case["labels"].with_suffix(".txt")
        staves = yolo_to_pixel_staves(label_path, page_w, page_h)
        print(f"  Per-staff bboxes: {len(staves)}")
        if not staves:
            print("    (no labels found, skip grouping)")
            continue

        systems = group_staves_by_brackets(staves, brackets)
        print(f"  Groups: {len(systems)} systems with {[s['staves_in_system'] for s in systems]} staves each")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
