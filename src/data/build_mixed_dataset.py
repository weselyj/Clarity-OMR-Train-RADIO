"""Assemble synthetic + real + sparse-augment YOLO datasets into a unified dataset.

Stratifies the val split per source corpus so each training run sees a fair
mix at validation time. Handles filename collisions by namespacing copied
files with the source corpus name.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Mapping

import yaml


def build_mixed_dataset(
    sources: Mapping[str, Path],
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Build a unified train/val dataset from multiple sources.

    Args:
        sources: mapping of corpus name -> dir containing `images/` and `labels/`
                 subdirs (the corpus name is used to namespace output filenames
                 to avoid collisions across corpora).
        out_dir: target directory; will be populated with train/{images,labels}
                 and val/{images,labels} subdirectories and a `data.yaml` file.
        val_ratio: fraction of each corpus to put in val (per-source stratified).
        seed: RNG seed for reproducibility.

    Returns:
        Path to the generated `data.yaml` file.
    """
    rng = random.Random(seed)
    train_imgs = out_dir / "train" / "images"
    train_lbls = out_dir / "train" / "labels"
    val_imgs = out_dir / "val" / "images"
    val_lbls = out_dir / "val" / "labels"
    for d in (train_imgs, train_lbls, val_imgs, val_lbls):
        d.mkdir(parents=True, exist_ok=True)

    for corpus_name, src_dir in sources.items():
        imgs = sorted((src_dir / "images").glob("*.png"))
        # Pair each image with its label; skip orphans.
        paired: list[tuple[Path, Path]] = []
        for img in imgs:
            lbl = src_dir / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                paired.append((img, lbl))
        rng.shuffle(paired)

        n_val = int(len(paired) * val_ratio)
        for i, (img, lbl) in enumerate(paired):
            target_imgs = val_imgs if i < n_val else train_imgs
            target_lbls = val_lbls if i < n_val else train_lbls
            # Namespace by corpus to avoid filename collisions across sources
            new_stem = f"{corpus_name}__{img.stem}"
            shutil.copy2(img, target_imgs / f"{new_stem}{img.suffix}")
            shutil.copy2(lbl, target_lbls / f"{new_stem}.txt")

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["staff"],
    }))
    return yaml_path
