"""Derive system-level YOLO labels for sparse_augment from existing per-staff labels.

Input: data/processed/sparse_augment/labels/{style}/*.txt + corresponding images.
Output: data/processed/sparse_augment/labels_systems/{style}/*.txt + .staves.json sidecars.

sparse_augment layout: labels nested per-style (labels/<style>/<page>.txt);
images keyed by DPI then style (images/dpi{N}/<style>/<page>.png).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.bracket_detector import detect_brackets_on_page, group_staves_by_brackets  # noqa: E402
from src.data.derive_systems_from_staves import group_staves_into_systems  # noqa: E402
from scripts.derive_audiolabs_systems import write_yolo_systems  # noqa: E402

DEFAULT_LABELS_DIR = Path("data/processed/sparse_augment/labels")
DEFAULT_IMAGES_DIR = Path("data/processed/sparse_augment/images")
DEFAULT_OUT_DIR = Path("data/processed/sparse_augment/labels_systems")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vertical-gap-factor", type=float, default=2.5)
    parser.add_argument("--x-overlap-threshold", type=float, default=0.80)
    parser.add_argument(
        "--use-bracket-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    from PIL import Image

    args.out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(args.labels_dir.rglob("*.txt"))
    print(f"Processing {len(label_files)} per-staff label files (sparse_augment)...", flush=True)

    n_pages = 0
    n_systems = 0
    n_skipped = 0
    n_bracket_detected = 0
    n_fallback_spatial = 0

    for label_path in label_files:
        relative = label_path.relative_to(args.labels_dir)
        page_id = relative.stem
        style_id = relative.parent.name if relative.parent.name else ""

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
            n_skipped += 1
            continue

        with Image.open(image_path) as img:
            page_w, page_h = img.size

        staves = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            x1 = (cx - w / 2) * page_w
            y1 = (cy - h / 2) * page_h
            x2 = (cx + w / 2) * page_w
            y2 = (cy + h / 2) * page_h
            staves.append({"bbox": (x1, y1, x2, y2)})

        systems = []
        if args.use_bracket_detection:
            brackets = detect_brackets_on_page(image_path, staves)
            if brackets:
                systems = group_staves_by_brackets(staves, brackets)
                n_bracket_detected += 1
        if not systems:
            systems = group_staves_into_systems(
                staves,
                vertical_gap_factor=args.vertical_gap_factor,
                x_overlap_threshold=args.x_overlap_threshold,
            )
            n_fallback_spatial += 1

        out_path = args.out_dir / style_id / f"{page_id}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_yolo_systems(systems, page_w, page_h, out_path)

        n_pages += 1
        n_systems += len(systems)

    print(
        f"Done. {n_pages} pages, {n_systems} systems, {n_skipped} skipped (no image). "
        f"Bracket-detected: {n_bracket_detected}, Spatial-fallback: {n_fallback_spatial}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
