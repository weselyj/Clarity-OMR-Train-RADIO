"""Build the v1 mixed *systems* YOLO training dataset.

Mirrors build_mixed_v1.py but consumes system-level labels:
- synthetic_v2 (re-rendered with both label types — uses labels_systems/)
- sparse_augment (with derived labels_systems/ from derive_sparse_augment_systems)
- omr_layout_real (with derived labels_systems/ from derive_audiolabs_systems)

Output layout matches mixed_v1's convention; the only difference is data.yaml's
``names`` field is overridden to ["system"] (still nc=1 single-class detection).

Also writes audit.json with per-source counts + train/val split sizes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.data.build_mixed_dataset import build_mixed_dataset  # noqa: E402

PROCESSED = REPO / "data" / "processed"
SYNTHETIC = PROCESSED / "synthetic_v2"
SPARSE = PROCESSED / "sparse_augment"
REAL = PROCESSED / "omr_layout_real"

DPIS = ["dpi94", "dpi150", "dpi300"]
STYLES = ["bravura-compact", "gootville-wide", "leipzig-default"]

VAL_RATIO = 0.2
SEED = 20260502


def make_sources() -> dict:
    """Build the 19-entry sources dict for build_mixed_dataset using labels_systems."""
    sources: dict = {}

    # 9 synthetic_v2 (dpi, style) combos
    for dpi in DPIS:
        for style in STYLES:
            key = f"syn_{dpi}_{style.replace('-', '_')}"
            sources[key] = {
                "images": SYNTHETIC / "images" / dpi / style,
                "labels": SYNTHETIC / "labels_systems" / style,
            }

    # 9 sparse_augment (dpi, style) combos
    for dpi in DPIS:
        for style in STYLES:
            key = f"aug_{dpi}_{style.replace('-', '_')}"
            sources[key] = {
                "images": SPARSE / "images" / dpi / style,
                "labels": SPARSE / "labels_systems" / style,
            }

    # 1 flat real-scan corpus (AudioLabs)
    sources["real"] = {
        "images": REAL / "images",
        "labels": REAL / "labels_systems",
    }

    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED / "mixed_systems_v1")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=" * 60)
    print("build_mixed_v2_systems.py - assembling mixed_systems_v1 training dataset")
    print("=" * 60)

    if args.out_dir.exists():
        print(f"\nWARNING: output directory already exists: {args.out_dir}")
        print("Existing files will NOT be overwritten (shutil.copy2 skips identical).")
        print("Delete the directory and re-run for a clean build.\n")

    sources = make_sources()

    # Pre-flight: report per-source counts (image+label pairings)
    print(f"\nSources ({len(sources)} total):")
    grand_total_imgs = 0
    grand_total_lbls = 0
    per_source_summary = {}
    for name, spec in sources.items():
        imgs_dir = Path(spec["images"]) if isinstance(spec, dict) else Path(spec) / "images"
        lbls_dir = Path(spec["labels"]) if isinstance(spec, dict) else Path(spec) / "labels"
        n_imgs = len(list(imgs_dir.glob("*.png"))) if imgs_dir.exists() else 0
        n_lbls = len(list(lbls_dir.glob("*.txt"))) if lbls_dir.exists() else 0
        # Pair count = images that have a matching label
        n_paired = 0
        if imgs_dir.exists() and lbls_dir.exists():
            for img in imgs_dir.glob("*.png"):
                if (lbls_dir / f"{img.stem}.txt").exists():
                    n_paired += 1
        grand_total_imgs += n_imgs
        grand_total_lbls += n_lbls
        per_source_summary[name] = {
            "images_dir": str(imgs_dir),
            "labels_dir": str(lbls_dir),
            "n_images": n_imgs,
            "n_labels": n_lbls,
            "n_paired": n_paired,
        }
        print(f"  {name:<40} {n_imgs:>5} images  {n_lbls:>5} labels  {n_paired:>5} paired")
    print(f"\n  TOTAL                                    {grand_total_imgs:>5} images  {grand_total_lbls:>5} labels")

    print(f"\nOutput dir : {args.out_dir}")
    print(f"Val ratio  : {args.val_ratio}  (per source, stratified)")
    print(f"RNG seed   : {args.seed}")
    print("\nStarting build (this may take several minutes for file copies)...\n")

    t0 = time.time()
    yaml_path = build_mixed_dataset(sources, out_dir=args.out_dir, val_ratio=args.val_ratio, seed=args.seed)
    elapsed = time.time() - t0

    # Override class name in data.yaml: this dataset is single-class "system" not "staff"
    data_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data_yaml["names"] = ["system"]
    yaml_path.write_text(yaml.safe_dump(data_yaml), encoding="utf-8")

    # Post-build summary
    train_imgs = list((args.out_dir / "train" / "images").glob("*.png"))
    val_imgs = list((args.out_dir / "val" / "images").glob("*.png"))
    train_lbls = list((args.out_dir / "train" / "labels").glob("*.txt"))
    val_lbls = list((args.out_dir / "val" / "labels").glob("*.txt"))

    print("=" * 60)
    print(f"Build complete in {elapsed:.1f}s")
    print(f"  train images : {len(train_imgs):,}")
    print(f"  train labels : {len(train_lbls):,}")
    print(f"  val   images : {len(val_imgs):,}")
    print(f"  val   labels : {len(val_lbls):,}")
    print(f"  total pairs  : {len(train_imgs) + len(val_imgs):,}")
    print(f"  data.yaml    : {yaml_path}")
    print("=" * 60)

    audit = {
        "dataset_name": "mixed_systems_v1",
        "build_seed": args.seed,
        "val_ratio": args.val_ratio,
        "elapsed_seconds": round(elapsed, 1),
        "n_train_pairs": len(train_imgs),
        "n_val_pairs": len(val_imgs),
        "n_total_pairs": len(train_imgs) + len(val_imgs),
        "data_yaml": str(yaml_path),
        "data_yaml_contents": data_yaml,
        "per_source": per_source_summary,
    }
    audit_path = args.out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"  audit.json   : {audit_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
