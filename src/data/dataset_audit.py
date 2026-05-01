"""Audit a YOLO-format dataset and emit a JSON report.

Captures: image-size histogram (500px buckets), bbox-count stats per image,
bbox-area stats, and class distribution. Run before training to verify the
dataset composition matches expectations.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image


def audit_dataset(dataset_root: Path) -> dict:
    """Walk dataset_root/{train,val}/{images,labels} and produce a summary dict."""
    report: dict = {}
    for split in ("train", "val"):
        split_dir = dataset_root / split
        imgs = list((split_dir / "images").glob("*.png"))
        size_buckets: dict[str, int] = defaultdict(int)
        bbox_areas: list[float] = []
        bbox_count_per_image: list[int] = []
        class_counter: Counter = Counter()

        for img_path in imgs:
            with Image.open(img_path) as img:
                w, h = img.size
            bucket = f"{(w // 500) * 500}x{(h // 500) * 500}"
            size_buckets[bucket] += 1

            lbl = split_dir / "labels" / f"{img_path.stem}.txt"
            if not lbl.exists():
                bbox_count_per_image.append(0)
                continue
            lines = lbl.read_text().strip().splitlines()
            bbox_count_per_image.append(len(lines))
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls, _, _, w_n, h_n = parts[:5]
                class_counter[int(cls)] += 1
                bbox_areas.append(float(w_n) * float(h_n))

        report[split] = {
            "n_images": len(imgs),
            "size_histogram": dict(sorted(size_buckets.items())),
            "bbox_count_mean": (mean(bbox_count_per_image) if bbox_count_per_image else 0),
            "bbox_count_median": (median(bbox_count_per_image) if bbox_count_per_image else 0),
            "bbox_area_mean": (mean(bbox_areas) if bbox_areas else 0),
            "bbox_area_median": (median(bbox_areas) if bbox_areas else 0),
            "class_distribution": dict(class_counter),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("dataset_audit.json"))
    args = parser.parse_args()

    report = audit_dataset(args.dataset_root)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
