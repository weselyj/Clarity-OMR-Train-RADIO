"""Assemble synthetic + real + sparse-augment YOLO datasets into a unified dataset.

Stratifies the val split per source corpus so each training run sees a fair
mix at validation time. Handles filename collisions by namespacing copied
files with the source corpus name.

Source spec
-----------
Each entry in ``sources`` maps a corpus name to either:

* a ``Path`` — treated as a directory containing ``images/`` and ``labels/``
  subdirectories (original / backwards-compatible behaviour), or
* a ``dict`` with keys ``"images"`` and ``"labels"`` — each pointing directly to
  the directory that holds the flat ``*.png`` / ``*.txt`` files.  This allows
  callers to wire up nested layouts (e.g. ``images/dpi94/bravura-compact/``)
  without staging copies.

Examples::

    # Legacy: flat source directory
    build_mixed_dataset({"real": Path("data/omr_layout_real")}, ...)

    # Extended: explicit image + label dirs (useful for nested multi-DPI trees)
    build_mixed_dataset({
        "syn_dpi94_bravura": {
            "images": Path("data/synthetic_multi_dpi/images/dpi94/bravura-compact"),
            "labels": Path("data/synthetic_multi_dpi/labels/bravura-compact"),
        },
    }, ...)
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Mapping, Union

import yaml

# A source spec is either a plain Path or a dict with 'images'/'labels' keys.
SourceSpec = Union[Path, "dict[str, Path]"]


def _resolve_source_dirs(spec: SourceSpec) -> tuple[Path, Path]:
    """Return (images_dir, labels_dir) for a source spec."""
    if isinstance(spec, dict):
        return Path(spec["images"]), Path(spec["labels"])
    # Plain Path — legacy flat-layout convention
    return spec / "images", spec / "labels"


def build_mixed_dataset(
    sources: Mapping[str, SourceSpec],
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Build a unified train/val dataset from multiple sources.

    Args:
        sources: mapping of corpus name -> source spec (see module docstring).
                 The corpus name is used to namespace output filenames to avoid
                 collisions across corpora.
        out_dir: target directory; will be populated with train/{images,labels}
                 and val/{images,labels} subdirectories and a ``data.yaml`` file.
        val_ratio: fraction of each corpus to put in val (per-source stratified).
        seed: RNG seed for reproducibility.

    Returns:
        Path to the generated ``data.yaml`` file.
    """
    rng = random.Random(seed)
    train_imgs = out_dir / "train" / "images"
    train_lbls = out_dir / "train" / "labels"
    val_imgs = out_dir / "val" / "images"
    val_lbls = out_dir / "val" / "labels"
    for d in (train_imgs, train_lbls, val_imgs, val_lbls):
        d.mkdir(parents=True, exist_ok=True)

    for corpus_name, spec in sources.items():
        images_dir, labels_dir = _resolve_source_dirs(spec)
        imgs = sorted(images_dir.glob("*.png"))
        # Pair each image with its label; skip orphans.
        paired: list[tuple[Path, Path]] = []
        for img in imgs:
            lbl = labels_dir / f"{img.stem}.txt"
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
