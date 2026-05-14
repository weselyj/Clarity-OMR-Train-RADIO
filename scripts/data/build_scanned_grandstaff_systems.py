"""Build the scanned_grandstaff_systems corpus from grandstaff_systems.

For each source image in grandstaff_systems, apply src.data.scan_degradation.apply_scan_degradation
with a seed derived from source_path. Copy token labels verbatim — only the image changes.

Output manifest mirrors grandstaff's schema with updated image_path.

Usage (on seder, where grandstaff lives):
  python scripts\\data\\build_scanned_grandstaff_systems.py
  python scripts\\data\\build_scanned_grandstaff_systems.py --limit 100  # dry-run subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from src.data.scan_degradation import apply_scan_degradation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data/processed/grandstaff_systems"
DEFAULT_OUT = REPO_ROOT / "data/processed/scanned_grandstaff_systems"


def _seed_from_path(source_path: str) -> int:
    h = hashlib.sha256(source_path.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def build_corpus(src_root: Path, out_root: Path, limit: int | None = None) -> None:
    src_manifest = src_root / "manifests/synthetic_token_manifest.jsonl"
    out_images = out_root / "images"
    out_manifests = out_root / "manifests"
    out_images.mkdir(parents=True, exist_ok=True)
    out_manifests.mkdir(parents=True, exist_ok=True)

    out_manifest_path = out_manifests / "synthetic_token_manifest.jsonl"
    written = 0
    skipped = 0
    with src_manifest.open() as fin, out_manifest_path.open("w") as fout:
        for line in fin:
            if limit is not None and written >= limit:
                break
            entry = json.loads(line)
            src_img_path = Path(entry["image_path"])
            if not src_img_path.is_absolute():
                src_img_path = src_root / src_img_path.name  # fallback if relative
            stem = src_img_path.stem
            out_img_path = out_images / f"{stem}.png"

            if out_img_path.exists():
                skipped += 1
            else:
                with Image.open(src_img_path) as src_img:
                    degraded = apply_scan_degradation(src_img.convert("L"), seed=_seed_from_path(entry["source_path"]))
                degraded.save(out_img_path, format="PNG")

            new_entry = dict(entry)
            new_entry["image_path"] = str(out_img_path)
            new_entry["original_image_path"] = str(src_img_path)
            fout.write(json.dumps(new_entry) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"  ... {written} entries written")
    print(f"Wrote {written} entries to {out_manifest_path} ({skipped} images skipped because already present)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: limit number of entries (for dry-run testing)")
    args = ap.parse_args()
    build_corpus(args.src_root, args.out_root, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
