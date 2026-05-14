"""Build the scanned_grandstaff_systems corpus from grandstaff_systems.

For each entry in grandstaff_systems' manifest, apply
src.data.scan_degradation.apply_scan_degradation to the source image and emit a
parallel manifest entry pointing at the degraded image. Token labels and
provenance are preserved verbatim; only the image changes, and the corpus tag
flips to scanned_grandstaff_systems.

Usage (on seder, where grandstaff lives):
  python scripts\\data\\build_scanned_grandstaff_systems.py
  python scripts\\data\\build_scanned_grandstaff_systems.py --limit 100  # dry-run subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from src.data.scan_degradation import apply_scan_degradation  # noqa: E402

DEFAULT_SRC = REPO_ROOT / "data/processed/grandstaff_systems"
DEFAULT_OUT = REPO_ROOT / "data/processed/scanned_grandstaff_systems"
SOURCE_CORPUS_TAG = "grandstaff_systems"
TARGET_CORPUS_TAG = "scanned_grandstaff_systems"


def _seed_from_path(source_path: str) -> int:
    h = hashlib.sha256(source_path.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def _seed_input(entry: dict) -> str:
    """Pick the most stable unique-per-source string available."""
    return entry.get("sample_id") or entry.get("source_path") or entry["image_path"]


def _resolve_source_image(image_path_str: str, repo_root: Path) -> Path:
    p = Path(image_path_str)
    return p if p.is_absolute() else repo_root / p


def _output_image_path(image_path_str: str, out_images_root: Path) -> Path:
    """Mirror the source structure under out_images_root, suffixed .png.

    For absolute source paths (test mode), flatten to basename to keep tmp_path
    fixtures simple. For relative source paths (production), preserve the full
    relative structure so per-piece stems (e.g., 'original_m-0-5') don't
    collide across composers.
    """
    p = Path(image_path_str)
    rel = Path(p.name) if p.is_absolute() else p
    return out_images_root / rel.with_suffix(".png")


def _new_sample_id(old_sample_id: str | None) -> str | None:
    if not old_sample_id:
        return old_sample_id
    return old_sample_id.replace(f"{SOURCE_CORPUS_TAG}:", f"{TARGET_CORPUS_TAG}:", 1)


def build_corpus(
    src_root: Path,
    out_root: Path,
    limit: int | None = None,
    repo_root: Path = REPO_ROOT,
) -> None:
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

            src_img_path = _resolve_source_image(entry["image_path"], repo_root)
            out_img_path = _output_image_path(entry["image_path"], out_images)
            out_img_path.parent.mkdir(parents=True, exist_ok=True)

            if out_img_path.exists():
                skipped += 1
            else:
                with Image.open(src_img_path) as src_img:
                    degraded = apply_scan_degradation(
                        src_img.convert("L"),
                        seed=_seed_from_path(_seed_input(entry)),
                    )
                degraded.save(out_img_path, format="PNG")

            new_entry = dict(entry)
            new_entry["original_image_path"] = entry["image_path"]
            try:
                new_entry["image_path"] = out_img_path.relative_to(repo_root).as_posix()
            except ValueError:
                new_entry["image_path"] = out_img_path.as_posix()
            if new_entry.get("dataset") == SOURCE_CORPUS_TAG:
                new_entry["dataset"] = TARGET_CORPUS_TAG
            if new_entry.get("variant") == "clean":
                new_entry["variant"] = "scanned"
            new_entry["sample_id"] = _new_sample_id(new_entry.get("sample_id"))
            fout.write(json.dumps(new_entry) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"  ... {written} entries written")
    print(
        f"Wrote {written} entries to {out_manifest_path} "
        f"({skipped} images skipped because already present)"
    )


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
