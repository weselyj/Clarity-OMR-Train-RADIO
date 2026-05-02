"""Visualize AudioLabs system-level labels overlaid on page images for spot-check.

Usage: python scripts/visualize_audiolabs_systems.py --num-samples 50 --out-dir /tmp/audiolabs_spotcheck
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_LABELS_DIR = Path("data/processed/omr_layout_real/labels_systems")
DEFAULT_IMAGES_DIR = Path("data/processed/omr_layout_real/images")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    label_files = sorted(args.labels_dir.glob("*.txt"))
    sample = rng.sample(label_files, min(args.num_samples, len(label_files)))

    n_written = 0
    for label_path in sample:
        page_id = label_path.stem
        image_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = args.images_dir / f"{page_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        page_w, page_h = img.size

        # Read .staves.json sidecar so we can label each box
        sidecar_path = label_path.with_suffix(".staves.json")
        staves_per: list[int] = []
        if sidecar_path.exists():
            try:
                staves_per = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception:
                staves_per = []

        bbox_idx = 0
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
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

            label = f"sys{bbox_idx}"
            if bbox_idx < len(staves_per):
                label += f" ({staves_per[bbox_idx]} staves)"
            draw.text((x1 + 8, max(0, y1 - 18)), label, fill="red")
            bbox_idx += 1

        img.save(args.out_dir / f"{page_id}_systems.png")
        n_written += 1

    print(f"Wrote {n_written} visualizations to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
