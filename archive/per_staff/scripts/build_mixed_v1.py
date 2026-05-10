"""Build the v1 mixed YOLO training dataset.

Sources
-------
- synthetic_multi_dpi: nested layout  images/dpi{94,150,300}/{style}/*.png
                                      labels/{style}/*.txt  (DPI-invariant)
  9 (dpi, style) combinations; approx 19,400 images total, 6,550 labels.

- sparse_augment:       same nested layout as synthetic_multi_dpi
  9 combinations; approx 2,013 images total, 662 labels.

- omr_layout_real:      flat layout   images/*.png  +  labels/*.txt
  919 image+label pairs.

Total sources: 9 + 9 + 1 = 19 stratified sources.

Output
------
  data/processed/mixed_v1/
    train/{images,labels}/
    val/{images,labels}/
    data.yaml

Option chosen: C (extended build_mixed_dataset)
-----------------------------------------------
We extended src/data/build_mixed_dataset.py to accept either a plain Path
(flat-layout, backwards-compatible) or a dict {"images": Path, "labels": Path}
(explicit dirs, used here for the nested synthetic/sparse corpora).

This approach avoids any staging copies (no extra ~60 GB disk use) and keeps
the stratified 80/20 val split working across all 19 sources independently.
All 5 existing unit tests continue to pass unchanged.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from repo root: python scripts/build_mixed_v1.py
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.data.build_mixed_dataset import build_mixed_dataset  # noqa: E402

PROCESSED = REPO / "data" / "processed"
SYNTHETIC = PROCESSED / "synthetic_multi_dpi"
SPARSE = PROCESSED / "sparse_augment"
REAL = PROCESSED / "omr_layout_real"
OUT = PROCESSED / "mixed_v1"

DPIS = ["dpi94", "dpi150", "dpi300"]
STYLES = ["bravura-compact", "gootville-wide", "leipzig-default"]

VAL_RATIO = 0.2
SEED = 42


def make_sources() -> dict:
    """Build the 19-entry sources dict for build_mixed_dataset."""
    sources: dict = {}

    # 9 synthetic_multi_dpi (dpi, style) combos
    for dpi in DPIS:
        for style in STYLES:
            key = f"syn_{dpi}_{style.replace('-', '_')}"
            sources[key] = {
                "images": SYNTHETIC / "images" / dpi / style,
                "labels": SYNTHETIC / "labels" / style,
            }

    # 9 sparse_augment (dpi, style) combos
    for dpi in DPIS:
        for style in STYLES:
            key = f"aug_{dpi}_{style.replace('-', '_')}"
            sources[key] = {
                "images": SPARSE / "images" / dpi / style,
                "labels": SPARSE / "labels" / style,
            }

    # 1 flat real-scan corpus
    sources["real"] = REAL

    return sources


def main() -> None:
    print("=" * 60)
    print("build_mixed_v1.py — assembling mixed_v1 training dataset")
    print("=" * 60)

    if OUT.exists():
        print(f"\nWARNING: output directory already exists: {OUT}")
        print("Existing files will NOT be overwritten (shutil.copy2 skips identical).")
        print("Delete the directory and re-run for a clean build.\n")

    sources = make_sources()

    # Pre-flight: report source image counts
    print(f"\nSources ({len(sources)} total):")
    grand_total_imgs = 0
    grand_total_lbls = 0
    for name, spec in sources.items():
        if isinstance(spec, dict):
            imgs_dir = Path(spec["images"])
            lbls_dir = Path(spec["labels"])
        else:
            imgs_dir = Path(spec) / "images"
            lbls_dir = Path(spec) / "labels"
        n_imgs = len(list(imgs_dir.glob("*.png"))) if imgs_dir.exists() else 0
        n_lbls = len(list(lbls_dir.glob("*.txt"))) if lbls_dir.exists() else 0
        grand_total_imgs += n_imgs
        grand_total_lbls += n_lbls
        print(f"  {name:<40} {n_imgs:>5} images  {n_lbls:>5} labels")
    print(f"\n  TOTAL                                    {grand_total_imgs:>5} images  {grand_total_lbls:>5} labels")

    print(f"\nOutput dir : {OUT}")
    print(f"Val ratio  : {VAL_RATIO}  (per source, stratified)")
    print(f"RNG seed   : {SEED}")
    print("\nStarting build (this may take several minutes for file copies)...\n")

    t0 = time.time()
    yaml_path = build_mixed_dataset(sources, out_dir=OUT, val_ratio=VAL_RATIO, seed=SEED)
    elapsed = time.time() - t0

    # Post-build summary
    train_imgs = list((OUT / "train" / "images").glob("*.png"))
    val_imgs = list((OUT / "val" / "images").glob("*.png"))
    train_lbls = list((OUT / "train" / "labels").glob("*.txt"))
    val_lbls = list((OUT / "val" / "labels").glob("*.txt"))

    print("=" * 60)
    print(f"Build complete in {elapsed:.1f}s")
    print(f"  train images : {len(train_imgs):,}")
    print(f"  train labels : {len(train_lbls):,}")
    print(f"  val   images : {len(val_imgs):,}")
    print(f"  val   labels : {len(val_lbls):,}")
    print(f"  total pairs  : {len(train_imgs) + len(val_imgs):,}")
    print(f"  data.yaml    : {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
