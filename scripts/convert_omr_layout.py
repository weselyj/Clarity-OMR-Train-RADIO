"""One-shot: convert all omr-layout-analysis JSONs to YOLO labels + collect images.

Walks a local clone of the omr-layout-analysis dataset (path provided via
``--source-root``) and processes every .json annotation file found under a
``json/`` subdirectory. For each annotation that has at least one staff,
copies the matching image into the target images/ dir and writes a
single-class YOLO label into the target labels/ dir, both keyed by a unique
piece-prefixed filename.

Dataset layout (al2_extracted only, single top-level source):
    al2_extracted/<piece>/json/<stem>.json
    al2_extracted/<piece>/img/<stem>.png

The `coco/` directory contains a single combined COCO-format JSON (not per-page);
those files are automatically skipped because their parent dir is "coco", not "json".
"""
import argparse
import shutil
from pathlib import Path

from src.data.omr_layout_import import convert_annotation_to_yolo

DEFAULT_TARGET_LABELS = Path("data/processed/omr_layout_real/labels")
DEFAULT_TARGET_IMAGES = Path("data/processed/omr_layout_real/images")


def find_image_for(json_path: Path) -> Path | None:
    """Resolve the image file matching a given annotation JSON.

    Convention: replace the `json` component with `img` in the path,
    and swap the .json extension for .png.

    Example:
        al2_extracted/Schubert_D911-01/json/Schubert_D911-01_001.json
        -> al2_extracted/Schubert_D911-01/img/Schubert_D911-01_001.png
    """
    # Primary: json/ -> img/, .json -> .png
    candidate = json_path.parent.parent / "img" / json_path.with_suffix(".png").name
    if candidate.exists():
        return candidate

    # Fallback variants in case some pieces use different names
    for img_dir in ("images", "png", "PNG", "Images"):
        candidate = json_path.parent.parent / img_dir / json_path.with_suffix(".png").name
        if candidate.exists():
            return candidate

    # Try .jpg as well
    for img_dir in ("img", "images", "png"):
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            candidate = json_path.parent.parent / img_dir / json_path.with_suffix(ext).name
            if candidate.exists():
                return candidate

    return None


def piece_prefix(json_path: Path) -> str:
    """Return a stable unique identifier for the source piece.

    The piece directory is two levels above the JSON file:
        al2_extracted/<piece>/json/<stem>.json
    so we take json_path.parent.parent.name -> e.g. "Schubert_D911-01"
    """
    return json_path.parent.parent.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True,
        help="Path to a local clone of the omr-layout-analysis dataset "
             "(directory containing the al2_extracted/ tree).",
    )
    parser.add_argument(
        "--target-labels", type=Path, default=DEFAULT_TARGET_LABELS,
        help="Directory to write YOLO label .txt files (default: %(default)s).",
    )
    parser.add_argument(
        "--target-images", type=Path, default=DEFAULT_TARGET_IMAGES,
        help="Directory to copy matching source images into (default: %(default)s).",
    )
    args = parser.parse_args()

    source_root = args.source_root
    target_labels = args.target_labels
    target_images = args.target_images

    if not source_root.exists():
        raise SystemExit(f"--source-root does not exist: {source_root}")

    target_labels.mkdir(parents=True, exist_ok=True)
    target_images.mkdir(parents=True, exist_ok=True)

    # Only process JSONs that live inside a `json/` subdirectory to
    # avoid picking up the coco/all_annotations.json or other non-page files.
    json_files = [
        p for p in source_root.rglob("*.json")
        if p.parent.name == "json"
    ]
    print(f"Found {len(json_files)} per-page JSON annotations")

    written = 0
    skipped_no_staves = 0
    skipped_no_image = 0

    for json_path in sorted(json_files):
        prefix = piece_prefix(json_path)
        out_stem = f"{prefix}__{json_path.stem}"
        label_path = target_labels / f"{out_stem}.txt"

        n = convert_annotation_to_yolo(json_path, label_path)
        if n == 0:
            skipped_no_staves += 1
            continue

        image_src = find_image_for(json_path)
        if image_src is None or not image_src.exists():
            # Roll back the label so we don't have orphans
            label_path.unlink(missing_ok=True)
            skipped_no_image += 1
            print(f"  WARNING: no image for {json_path}")
            continue

        image_dst = target_images / f"{out_stem}{image_src.suffix.lower()}"
        if not image_dst.exists():
            shutil.copy2(image_src, image_dst)
        written += 1

    print(
        f"\nDone. Wrote {written} pairs; "
        f"skipped {skipped_no_staves} (no staves), "
        f"{skipped_no_image} (no matching image)"
    )


if __name__ == "__main__":
    main()
