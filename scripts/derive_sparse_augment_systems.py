"""Derive system-level YOLO labels for sparse_augment from existing per-staff labels.

Input: data/processed/sparse_augment/labels/{style}/*.txt + corresponding images.
       data/processed/sparse_augment/pages/{style}/*.svg  (Verovio SVG, same stem)
Output: data/processed/sparse_augment/labels_systems/{style}/*.txt + .staves.json sidecars.

v15 algorithm: SVG-hierarchy grouping (same as generate_synthetic.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.generate_synthetic import (  # noqa: E402
    _build_system_yolo_objects_v15,
    write_yolo_labels,
)

DEFAULT_LABELS_DIR = Path("data/processed/sparse_augment/labels")
DEFAULT_PAGES_DIR  = Path("data/processed/sparse_augment/pages")
DEFAULT_IMAGES_DIR = Path("data/processed/sparse_augment/images")
DEFAULT_OUT_DIR    = Path("data/processed/sparse_augment/labels_systems")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--pages-dir",  type=Path, default=DEFAULT_PAGES_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required: pip install Pillow", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(args.labels_dir.rglob("*.txt"))
    print(f"Processing {len(label_files)} per-staff label files (sparse_augment v15)...", flush=True)

    n_pages = 0
    n_systems = 0
    n_skipped_no_image = 0
    n_skipped_no_svg = 0

    for label_path in label_files:
        relative  = label_path.relative_to(args.labels_dir)
        page_id   = relative.stem
        style_id  = relative.parent.name if relative.parent.name else ""

        # ----------------------------------------------------------------
        # Resolve the PNG (need page dimensions)
        # ----------------------------------------------------------------
        image_path = None
        for dpi_dir in args.images_dir.iterdir():
            if not dpi_dir.is_dir():
                continue
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = dpi_dir / style_id / f"{page_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path:
                break

        if image_path is None:
            n_skipped_no_image += 1
            continue

        with Image.open(image_path) as img:
            page_w, page_h = img.size

        # ----------------------------------------------------------------
        # Resolve the SVG (v15 hierarchy grouping needs it)
        # ----------------------------------------------------------------
        svg_path = args.pages_dir / style_id / f"{page_id}.svg"
        if not svg_path.exists():
            n_skipped_no_svg += 1
            continue

        # ----------------------------------------------------------------
        # Read per-staff YOLO labels → pixel (x, y, w, h)
        # ----------------------------------------------------------------
        staff_boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4]))
            x1 = (cx - w / 2) * page_w
            y1 = (cy - h / 2) * page_h
            w_px = w * page_w
            h_px = h * page_h
            staff_boxes.append((x1, y1, w_px, h_px))

        if not staff_boxes:
            n_skipped_no_image += 1  # treat empty label as nothing to do
            continue

        # ----------------------------------------------------------------
        # v15 system grouping
        # ----------------------------------------------------------------
        system_label_objects, staves_in_system = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, page_w, page_h
        )

        if not system_label_objects:
            continue

        # ----------------------------------------------------------------
        # Write output — same flat YOLO format + .staves.json sidecar
        # ----------------------------------------------------------------
        out_path = args.out_dir / style_id / f"{page_id}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_yolo_labels(out_path, system_label_objects,
                          page_width=page_w, page_height=page_h)
        out_path.with_suffix(".staves.json").write_text(
            json.dumps(staves_in_system), encoding="utf-8"
        )

        n_pages += 1
        n_systems += len(system_label_objects)

    print(
        f"Done. {n_pages} pages, {n_systems} systems. "
        f"Skipped: {n_skipped_no_image} (no image), {n_skipped_no_svg} (no SVG).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
