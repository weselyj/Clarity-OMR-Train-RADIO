"""Debug: print svg_layout + staff_boxes + assignment for the Burleigh synthetic page."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.generate_synthetic import (  # noqa: E402
    _build_system_yolo_objects,
    _extract_system_layout_from_svg,
)


def main() -> int:
    from PIL import Image

    page_id = "data__openscore_lieder__scores__Burleigh___Harry_Thacker__5_Songs_of_Laurence_Hope__1_Worth_While__lc6518082.mxl__leipzig-default__p002"
    svg_path = Path(f"data/processed/synthetic_v2/pages/leipzig-default/{page_id}.svg")
    img_path = Path(f"data/processed/synthetic_v2/images/dpi300/leipzig-default/{page_id}.png")
    label_path = Path(f"data/processed/synthetic_v2/labels/leipzig-default/{page_id}.txt")

    print(f"SVG exists: {svg_path.exists()}")
    print(f"Image exists: {img_path.exists()}")
    print(f"Label exists: {label_path.exists()}")

    with Image.open(img_path) as img:
        page_w, page_h = img.size
    print(f"Page dims: {page_w} x {page_h}")

    # Read per-staff labels (normalized) -> pixel-space (x, y, w, h)
    staff_boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        w_px = w * page_w
        h_px = h * page_h
        x_px = cx * page_w - w_px / 2
        y_px = cy * page_h - h_px / 2
        staff_boxes.append((x_px, y_px, w_px, h_px))
    print(f"\nPer-staff bboxes (pixel x, y, w, h):")
    for i, sb in enumerate(staff_boxes):
        print(f"  {i}: x={sb[0]:.0f}, y={sb[1]:.0f}, w={sb[2]:.0f}, h={sb[3]:.0f}, y_center={sb[1] + sb[3]/2:.0f}")

    # Extract Verovio system layout
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_layout = _extract_system_layout_from_svg(svg_text)
    print(f"\nVerovio svg_layout: {len(svg_layout) if svg_layout else 0} systems")
    if svg_layout:
        for i, sys in enumerate(svg_layout):
            print(f"  sys {i}: y_top={sys.y_top:.0f}, y_bottom={sys.y_bottom:.0f}, "
                  f"staves_per_system={sys.staves_per_system}, x_left={sys.x_left}")

    # Run the assignment
    label_objects, staves_in_system = _build_system_yolo_objects(staff_boxes, svg_layout)
    print(f"\nResult: {len(label_objects)} label objects, staves_in_system={staves_in_system}")
    for i, (cls, bbox) in enumerate(label_objects):
        print(f"  obj {i}: bbox={bbox}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
