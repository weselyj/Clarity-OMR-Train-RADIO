#!/usr/bin/env python3
"""Filter low-ink synthetic pages from YOLO split files.

This removes samples where rendered PNG pages are effectively blank (or unreadable),
which otherwise introduces label noise during YOLO training.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class InkResult:
    split: str
    image_relpath: str
    ink_ratio: Optional[float]
    reason: str


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _ink_ratio_png_stdlib(path: Path, *, sample_step: int, gray_threshold: int) -> float:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")

    pos = 8
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    interlace = 0
    idat_chunks: List[bytes] = []

    while pos + 8 <= len(data):
        chunk_len = int.from_bytes(data[pos : pos + 4], "big")
        pos += 4
        chunk_type = data[pos : pos + 4]
        pos += 4
        chunk_data = data[pos : pos + chunk_len]
        pos += chunk_len
        pos += 4  # crc

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _comp, _flt, interlace = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if bit_depth != 8:
        raise ValueError(f"Unsupported bit depth ({bit_depth}) in {path}")
    if interlace != 0:
        raise ValueError(f"Unsupported interlace ({interlace}) in {path}")

    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"Unsupported color type ({color_type}) in {path}")

    raw = zlib.decompress(b"".join(idat_chunks))
    stride = width * channels
    bytes_per_pixel = channels

    prev = bytearray(stride)
    idx = 0
    ink = 0
    total = 0
    sampled_rows = set(range(0, height, max(1, sample_step)))

    for y in range(height):
        filter_type = raw[idx]
        idx += 1
        scanline = bytearray(raw[idx : idx + stride])
        idx += stride

        if filter_type == 1:  # Sub
            for x in range(stride):
                left = scanline[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
                scanline[x] = (scanline[x] + left) & 255
        elif filter_type == 2:  # Up
            for x in range(stride):
                scanline[x] = (scanline[x] + prev[x]) & 255
        elif filter_type == 3:  # Average
            for x in range(stride):
                left = scanline[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
                up = prev[x]
                scanline[x] = (scanline[x] + ((left + up) >> 1)) & 255
        elif filter_type == 4:  # Paeth
            for x in range(stride):
                left = scanline[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
                up = prev[x]
                up_left = prev[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
                scanline[x] = (scanline[x] + _paeth(left, up, up_left)) & 255
        elif filter_type != 0:
            raise ValueError(f"Unsupported PNG filter ({filter_type}) in {path}")

        if y in sampled_rows:
            if channels in (1, 2):
                for x in range(0, width, max(1, sample_step)):
                    gray = scanline[x * channels]
                    if gray < gray_threshold:
                        ink += 1
                    total += 1
            else:
                for x in range(0, width, max(1, sample_step)):
                    base = x * channels
                    r = scanline[base]
                    g = scanline[base + 1]
                    b = scanline[base + 2]
                    gray = (299 * r + 587 * g + 114 * b) // 1000
                    if gray < gray_threshold:
                        ink += 1
                    total += 1

        prev = scanline

    return (ink / total) if total else 0.0


def compute_ink_ratio(path: Path, *, sample_step: int, gray_threshold: int) -> float:
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore[assignment]

    if Image is not None:
        with Image.open(path) as image:
            gray = image.convert("L")
            width, height = gray.size
            step = max(1, sample_step)
            sample_w = max(1, width // step)
            sample_h = max(1, height // step)
            sampled = gray.resize((sample_w, sample_h))
            hist = sampled.histogram()
            ink = sum(hist[: max(0, min(255, int(gray_threshold)))])
            total = sample_w * sample_h
            return (ink / total) if total else 0.0

    return _ink_ratio_png_stdlib(path, sample_step=sample_step, gray_threshold=gray_threshold)


def _load_split(split_name: str, split_path: Path) -> List[str]:
    if not split_path.exists():
        return []
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_split(split_path: Path, image_paths: Sequence[str]) -> None:
    split_path.write_text("\n".join(image_paths) + ("\n" if image_paths else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Filter low-ink synthetic pages from split txt files.")
    parser.add_argument("--project-root", type=Path, default=project_root, help="Repository root.")
    parser.add_argument("--train-split", type=Path, default=project_root / "info" / "train.txt", help="Train split txt.")
    parser.add_argument("--val-split", type=Path, default=project_root / "info" / "val.txt", help="Val split txt.")
    parser.add_argument("--test-split", type=Path, default=project_root / "info" / "test.txt", help="Test split txt.")
    parser.add_argument(
        "--ink-threshold",
        type=float,
        default=0.005,
        help="Remove samples with ink_ratio <= threshold (or unreadable image).",
    )
    parser.add_argument("--sample-step", type=int, default=12, help="Pixel sampling stride for ink estimation.")
    parser.add_argument("--gray-threshold", type=int, default=245, help="Pixel < threshold is counted as ink.")
    parser.add_argument(
        "--removed-report-json",
        type=Path,
        default=project_root / "info" / "removed_low_ink_samples.json",
        help="JSON report path for removed samples.",
    )
    parser.add_argument(
        "--removed-report-txt",
        type=Path,
        default=project_root / "info" / "removed_low_ink_samples.txt",
        help="TXT report path for removed samples.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze only; do not rewrite split files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    split_paths = {
        "train": args.train_split.resolve(),
        "val": args.val_split.resolve(),
        "test": args.test_split.resolve(),
    }

    split_items = {name: _load_split(name, path) for name, path in split_paths.items()}
    removed: List[InkResult] = []
    kept: dict[str, List[str]] = {name: [] for name in split_paths}

    for split_name, images in split_items.items():
        for rel in images:
            image_path = (project_root / rel).resolve()
            try:
                ratio = compute_ink_ratio(
                    image_path,
                    sample_step=max(1, int(args.sample_step)),
                    gray_threshold=max(0, min(255, int(args.gray_threshold))),
                )
            except Exception:
                removed.append(
                    InkResult(
                        split=split_name,
                        image_relpath=rel,
                        ink_ratio=None,
                        reason="image_load_error",
                    )
                )
                continue

            if ratio <= float(args.ink_threshold):
                removed.append(
                    InkResult(
                        split=split_name,
                        image_relpath=rel,
                        ink_ratio=ratio,
                        reason="low_ink",
                    )
                )
            else:
                kept[split_name].append(rel)

    if not args.dry_run:
        for split_name, split_path in split_paths.items():
            _write_split(split_path, kept[split_name])

    by_reason = Counter(item.reason for item in removed)
    by_split = Counter(item.split for item in removed)

    report = {
        "threshold": float(args.ink_threshold),
        "sample_step": int(args.sample_step),
        "gray_threshold": int(args.gray_threshold),
        "dry_run": bool(args.dry_run),
        "split_counts_before": {name: len(items) for name, items in split_items.items()},
        "split_counts_after": {name: len(items) for name, items in kept.items()},
        "removed_count": len(removed),
        "removed_by_reason": dict(by_reason),
        "removed_by_split": dict(by_split),
        "removed_samples": [
            {
                "split": item.split,
                "image_relpath": item.image_relpath,
                "ink_ratio": item.ink_ratio,
                "reason": item.reason,
            }
            for item in removed
        ],
    }

    args.removed_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_lines = [f"{item.split}\t{item.reason}\t{item.ink_ratio}\t{item.image_relpath}" for item in removed]
    args.removed_report_txt.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
