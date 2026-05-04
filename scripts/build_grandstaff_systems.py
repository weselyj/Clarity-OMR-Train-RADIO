#!/usr/bin/env python3
"""Build the grandstaff_systems manifest from GrandStaff .krn files using the corrected kern converter."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Make src/ importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.convert_tokens import convert_kern_file


@dataclass
class BuilderStats:
    entries_written: int = 0
    skipped_oversized: int = 0
    skipped_no_image: int = 0
    skipped_empty_tokens: int = 0
    spine_count_histogram: Dict[int, int] = field(default_factory=lambda: Counter())


def _spine_count_from_tokens(tokens: List[str]) -> int:
    starts = tokens.count("<staff_start>")
    return max(1, starts)


def _image_variants(krn_path: Path) -> List[Tuple[str, Path]]:
    base = krn_path.with_suffix("")
    out: List[Tuple[str, Path]] = []
    clean = base.with_suffix(".jpg")
    if clean.exists():
        out.append(("clean", clean))
    distorted = base.parent / f"{base.name}_distorted.jpg"
    if distorted.exists():
        out.append(("distorted", distorted))
    return out


def _assign_split(group_id: str, split_ratios: Dict[str, float], seed: int) -> str:
    """Hash group_id deterministically and assign train/val/test by ratio buckets."""
    import hashlib

    h = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(h[:8], "big") / 2**64  # uniform [0, 1)
    cum = 0.0
    for split_name, ratio in split_ratios.items():
        cum += ratio
        if bucket < cum:
            return split_name
    # Floating-point rounding fallback.
    return list(split_ratios.keys())[-1]


def build_manifest_entries(
    grandstaff_root: Path,
    *,
    max_sequence_length: int,
    project_root: Path,
    split_ratios: Dict[str, float] | None = None,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, object]], BuilderStats]:
    """Build manifest entries. Splits are assigned per-group_id (so all variants of one piece
    land in the same split — prevents val/train leakage across clean/distorted pairs).
    """
    if split_ratios is None:
        split_ratios = {"train": 0.90, "val": 0.05, "test": 0.05}
    # Validate ratios sum to 1.0 (within float tolerance).
    if not (0.999 <= sum(split_ratios.values()) <= 1.001):
        raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios.values())}")

    stats = BuilderStats()
    entries: List[Dict[str, object]] = []
    if not grandstaff_root.exists():
        return entries, stats

    for krn_path in sorted(grandstaff_root.rglob("*.krn")):
        tokens = convert_kern_file(krn_path)
        if not tokens:
            stats.skipped_empty_tokens += 1
            continue
        if len(tokens) > max_sequence_length:
            stats.skipped_oversized += 1
            continue
        spine_count = _spine_count_from_tokens(tokens)
        stats.spine_count_histogram[spine_count] += 1
        variants = _image_variants(krn_path)
        if not variants:
            stats.skipped_no_image += 1
            continue
        try:
            dir_rel = krn_path.parent.relative_to(grandstaff_root).as_posix()
        except ValueError:
            dir_rel = krn_path.parent.name
        base_stem = krn_path.stem
        group_id = f"{dir_rel}/{base_stem}"
        split = _assign_split(group_id, split_ratios, split_seed)
        for variant, image_path in variants:
            try:
                image_rel = image_path.relative_to(project_root).as_posix()
                krn_rel = krn_path.relative_to(project_root).as_posix()
            except ValueError:
                image_rel = str(image_path)
                krn_rel = str(krn_path)
            entries.append(
                {
                    "sample_id": f"grandstaff_systems:{group_id}:{image_path.stem}",
                    "dataset": "grandstaff_systems",
                    "group_id": group_id,
                    "modality": "image+notation",
                    "variant": variant,
                    "split": split,
                    "image_path": image_rel,
                    "krn_path": krn_rel,
                    "token_sequence": tokens,
                    "staves_in_system": spine_count,
                }
            )
            stats.entries_written += 1
    return entries, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grandstaff-root", type=Path, default=Path("data/grandstaff"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/grandstaff_systems/manifests/synthetic_token_manifest.jsonl"),
    )
    parser.add_argument("--max-sequence-length", type=int, default=768)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()

    entries, stats = build_manifest_entries(
        args.grandstaff_root,
        max_sequence_length=args.max_sequence_length,
        project_root=args.project_root,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"[builder] wrote {stats.entries_written} entries to {args.output}")
    print(f"[builder] spine-count histogram: {dict(stats.spine_count_histogram)}")
    print(f"[builder] skipped: oversized={stats.skipped_oversized}, no_image={stats.skipped_no_image}, empty={stats.skipped_empty_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
