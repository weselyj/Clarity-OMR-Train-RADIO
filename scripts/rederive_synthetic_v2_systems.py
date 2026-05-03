"""Re-derive labels_systems/ for an existing synthetic_v2 corpus IN PLACE.

Used after fixes to _build_system_yolo_objects (e.g., adding bracket inclusion
via the SVG system rect's x_left). Re-reads existing per-staff labels + SVGs,
re-emits labels_systems/ files. Avoids the 6h cost of a full re-render.

Per-page workflow:
  1. Read per-staff YOLO labels (already correct; normalized format)
  2. Look up page_width/height from the corresponding image
  3. Convert per-staff labels back to pixel-space staff_boxes
  4. Re-parse the SVG to extract _SvgSystemInfo (now includes x_left)
  5. Run _build_system_yolo_objects(staff_boxes, svg_layout)
  6. Write new labels_systems/<style>/<page>.txt + .staves.json sidecar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.generate_synthetic import (  # noqa: E402
    _build_system_yolo_objects_v15,
    write_yolo_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("data/processed/synthetic_v2"),
        help="Root of the synthetic corpus (contains labels/, labels_systems/, pages/, images/).",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default="leipzig-default,bravura-compact,gootville-wide",
        help="Comma-separated style ids to process.",
    )
    parser.add_argument(
        "--dpi-for-dims",
        type=int,
        default=300,
        help="DPI subdir to read page dimensions from (any DPI works; labels are DPI-invariant).",
    )
    args = parser.parse_args()

    from PIL import Image

    corpus = args.corpus_root.resolve()
    labels_root = corpus / "labels"
    labels_systems_root = corpus / "labels_systems"
    pages_root = corpus / "pages"
    images_root = corpus / "images" / f"dpi{args.dpi_for_dims}"

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    print(f"Re-deriving systems labels under {labels_systems_root}", flush=True)
    print(f"Styles: {styles}", flush=True)

    n_pages = 0
    n_systems = 0
    n_no_svg = 0
    n_no_image = 0
    n_no_layout = 0
    n_no_labels = 0
    t0 = time.time()

    for style in styles:
        style_labels_dir = labels_root / style
        style_systems_dir = labels_systems_root / style
        style_pages_dir = pages_root / style
        style_images_dir = images_root / style

        if not style_labels_dir.exists():
            print(f"  skip style {style}: no per-staff labels dir", flush=True)
            continue

        style_systems_dir.mkdir(parents=True, exist_ok=True)
        label_files = sorted(style_labels_dir.glob("*.txt"))
        print(f"  {style}: {len(label_files)} per-staff label files", flush=True)

        n_processed_for_style = 0
        for label_path in label_files:
            page_id = label_path.stem
            svg_path = style_pages_dir / f"{page_id}.svg"
            image_path = style_images_dir / f"{page_id}.png"

            if not svg_path.exists():
                n_no_svg += 1
                continue
            if not image_path.exists():
                n_no_image += 1
                continue

            with Image.open(image_path) as img:
                page_w, page_h = img.size

            # Read per-staff YOLO labels and convert back to pixel-space (x, y, w, h).
            staff_boxes = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                w_px = w * page_w
                h_px = h * page_h
                x_px = cx * page_w - w_px / 2
                y_px = cy * page_h - h_px / 2
                staff_boxes.append((x_px, y_px, w_px, h_px))

            if not staff_boxes:
                n_no_labels += 1
                continue

            label_objects, staves_in_system = _build_system_yolo_objects_v15(
                svg_path, staff_boxes, page_w, page_h
            )
            if not label_objects:
                continue

            out_path = style_systems_dir / f"{page_id}.txt"
            write_yolo_labels(out_path, label_objects, page_width=page_w, page_height=page_h)
            out_path.with_suffix(".staves.json").write_text(
                json.dumps(staves_in_system), encoding="utf-8",
            )

            n_pages += 1
            n_systems += len(label_objects)
            n_processed_for_style += 1

        print(f"  {style}: re-derived {n_processed_for_style} pages", flush=True)

    elapsed = time.time() - t0
    print(
        f"Done in {elapsed:.1f}s. {n_pages} pages, {n_systems} systems. "
        f"Skipped: no_svg={n_no_svg}, no_image={n_no_image}, no_layout={n_no_layout}, no_labels={n_no_labels}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
