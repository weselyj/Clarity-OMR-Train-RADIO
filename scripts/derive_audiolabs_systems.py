"""Derive system-level YOLO labels for AudioLabs v2 from existing per-staff labels.

Input: data/processed/omr_layout_real/labels/*.txt + corresponding images for dims.
Output: data/processed/omr_layout_real/labels_systems/*.txt + .staves.json sidecars
        (each line: "0 cx cy w h", normalized to [0,1]; sidecar = JSON list of stave counts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.derive_systems_from_staves import group_staves_into_systems  # noqa: E402

DEFAULT_LABELS_DIR = Path("data/processed/omr_layout_real/labels")
DEFAULT_IMAGES_DIR = Path("data/processed/omr_layout_real/images")
DEFAULT_OUT_DIR = Path("data/processed/omr_layout_real/labels_systems")


def write_yolo_systems(systems, page_w, page_h, out_path: Path) -> None:
    """Write YOLO-format system labels (class 0) + .staves.json sidecar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    staves_per_bbox = []
    for s in systems:
        x1, y1, x2, y2 = s["bbox"]
        cx = ((x1 + x2) / 2) / max(page_w, 1)
        cy = ((y1 + y2) / 2) / max(page_h, 1)
        w = (x2 - x1) / max(page_w, 1)
        h = (y2 - y1) / max(page_h, 1)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        staves_per_bbox.append(int(s["staves_in_system"]))
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    out_path.with_suffix(".staves.json").write_text(json.dumps(staves_per_bbox), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vertical-gap-factor", type=float, default=1.5)
    parser.add_argument("--x-overlap-threshold", type=float, default=0.80)
    args = parser.parse_args()

    from PIL import Image

    args.out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(args.labels_dir.glob("*.txt"))
    print(f"Processing {len(label_files)} per-staff label files...", flush=True)

    n_pages = 0
    n_systems = 0
    n_staves_grouped = 0
    n_skipped_no_image = 0

    for label_path in label_files:
        page_id = label_path.stem
        image_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = args.images_dir / f"{page_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            n_skipped_no_image += 1
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

        systems = group_staves_into_systems(
            staves,
            vertical_gap_factor=args.vertical_gap_factor,
            x_overlap_threshold=args.x_overlap_threshold,
        )

        out_path = args.out_dir / f"{page_id}.txt"
        write_yolo_systems(systems, page_w, page_h, out_path)

        n_pages += 1
        n_systems += len(systems)
        n_staves_grouped += sum(s["staves_in_system"] for s in systems)

    print(
        f"Done. {n_pages} pages, {n_systems} systems, {n_staves_grouped} staves grouped, "
        f"{n_skipped_no_image} skipped (no image).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
