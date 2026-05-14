"""Offline scan-realistic degradation for engraved music score images.

Used to derive scanned_grandstaff_systems from grandstaff_systems. Deterministic
per source via seeded RNG so the corpus is reproducible across machines.

Pipeline (applied in order):
    1. Rotation: ±2.3° uniform (border replicate)
    2. Perspective warp: corners offset up to 3% of dimension
    3. Brightness ±0.10, contrast ±0.12
    4. Paper-texture overlay: low-amplitude noise field, σ=0.06
    5. Light Gaussian blur: kernel 3, σ uniform[0.3, 0.9]
    6. Salt-and-pepper noise: fraction 0.0015
    7. JPEG round-trip: quality uniform[60, 90]

Parameter ranges tuned to span the Phase 1a calibration distribution
(docs/audits/2026-05-13-real-scan-degradation-calibration.md). JPEG lower
bound widened to 60 (cf. calibration's [50, 60] from a single q=29 outlier);
60 keeps a single heavily-compressed scan in distribution without
over-degrading the rest of the synthetic corpus.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_scan_degradation(img: Image.Image, seed: int) -> Image.Image:
    """Apply the full degradation pipeline. Deterministic for fixed (img, seed)."""
    rng = np.random.default_rng(seed)
    out = img

    # 1. Rotation
    rot_deg = float(rng.uniform(-2.3, 2.3))
    out = out.rotate(rot_deg, resample=Image.BICUBIC, fillcolor=255)

    # 2. Perspective warp (mild)
    out = _apply_perspective(out, rng, max_offset_frac=0.03)

    # 3. Brightness + contrast
    out = ImageEnhance.Brightness(out).enhance(1.0 + float(rng.uniform(-0.10, 0.10)))
    out = ImageEnhance.Contrast(out).enhance(1.0 + float(rng.uniform(-0.12, 0.12)))

    # 4. Paper-texture overlay
    arr = np.asarray(out, dtype=np.float32) / 255.0
    noise = rng.normal(0.0, 0.06, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    out = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L")

    # 5. Light Gaussian blur
    sigma = float(rng.uniform(0.3, 0.9))
    out = out.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 6. Salt-and-pepper noise
    out = _apply_salt_pepper(out, rng, fraction=0.0015)

    # 7. JPEG round-trip
    quality = int(rng.integers(60, 91))
    buf = BytesIO()
    out.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("L").copy()

    return out


def _apply_perspective(img: Image.Image, rng: np.random.Generator, max_offset_frac: float) -> Image.Image:
    """Apply a mild perspective warp by jittering the four corners."""
    import cv2
    w, h = img.size
    max_off_x = max_offset_frac * w
    max_off_y = max_offset_frac * h
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + rng.uniform(-1.0, 1.0, src.shape).astype(np.float32) * np.array(
        [max_off_x, max_off_y], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(src, dst)
    arr = np.asarray(img, dtype=np.uint8)
    warped = cv2.warpPerspective(
        arr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(warped, mode=img.mode)


def _apply_salt_pepper(img: Image.Image, rng: np.random.Generator, fraction: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.uint8).copy()
    flat = arr.reshape(-1)
    n = max(1, int(round(flat.size * fraction)))
    indices = rng.choice(flat.size, size=n, replace=False)
    split = n // 2
    flat[indices[:split]] = 0
    flat[indices[split:]] = 255
    return Image.fromarray(arr, mode=img.mode)
