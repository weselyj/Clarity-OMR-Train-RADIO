"""Measure scan-quality statistics from real scanned piano scores.

Inputs: a directory of images (Bethlehem.jpg, TimeMachine page renders, plus
~5 IMSLP scans). Outputs per-image and aggregated statistics on rotation, noise,
brightness/contrast, blur, and JPEG quality estimate.

Writes:
  - docs/audits/2026-05-13-real-scan-degradation-calibration.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image, ImageFilter


def estimate_rotation_degrees(img: Image.Image) -> float:
    """Return estimated rotation in degrees via Hough transform on edges."""
    import cv2
    gray = np.asarray(img.convert("L"))
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return 0.0
    angles_deg = []
    for line in lines[:50]:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90.0  # 0 = horizontal
        if -10.0 < angle < 10.0:
            angles_deg.append(angle)
    if not angles_deg:
        return 0.0
    # Negate to match PIL.Image.rotate's CCW-positive convention.
    # Image coordinates have y-axis flipped vs. math, so Hough's median angle
    # is the math-space rotation — the negation of the visual rotation PIL applies.
    return -float(np.median(angles_deg))


def estimate_noise_sigma(img: Image.Image) -> float:
    """Return high-frequency residual stddev — proxy for noise level (normalized 0-1)."""
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    blurred = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32) / 255.0
    residual = gray - blurred
    return float(residual.std())


def estimate_blur_laplacian_var(img: Image.Image) -> float:
    """Laplacian variance — higher = sharper, lower = blurrier."""
    import cv2
    gray = np.asarray(img.convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_jpeg_quality(img_path: Path) -> int | None:
    """Heuristic JPEG quality estimate from quantization tables. Returns None for non-JPEG."""
    if img_path.suffix.lower() not in {".jpg", ".jpeg"}:
        return None
    try:
        with Image.open(img_path) as img:
            qtables = getattr(img, "quantization", None)
            if not qtables:
                return None
            # Lower mean → higher quality. Calibrate roughly: q50 mean ≈ 30, q90 ≈ 6.
            mean_q = np.mean(qtables[0])
            return int(round(max(0.0, min(100.0, 100.0 - (mean_q - 6.0) * 100.0 / 24.0))))
    except Exception:
        return None


def measure(img_path: Path) -> Dict:
    img = Image.open(img_path)
    return {
        "path": str(img_path),
        "size": img.size,
        "rotation_deg": estimate_rotation_degrees(img),
        "noise_sigma": estimate_noise_sigma(img),
        "blur_laplacian_var": estimate_blur_laplacian_var(img),
        "jpeg_quality": estimate_jpeg_quality(img_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", type=Path, required=True,
                    help="Directory of real scan images")
    ap.add_argument("--output", type=Path,
                    default=Path("docs/audits/2026-05-13-real-scan-degradation-calibration.md"))
    args = ap.parse_args()

    results = []
    for img_path in sorted(args.scan_dir.glob("*.[jp][pn]g")):
        results.append(measure(img_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real-Scan Degradation Calibration",
        "",
        f"**Date:** 2026-05-13",
        f"**Inputs:** {len(results)} scans from `{args.scan_dir}`",
        "",
        "## Per-scan measurements",
        "",
        "| File | rotation (°) | noise σ | blur var | JPEG q |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{Path(r['path']).name}` | {r['rotation_deg']:.2f} | "
            f"{r['noise_sigma']:.4f} | {r['blur_laplacian_var']:.1f} | "
            f"{r['jpeg_quality'] or 'n/a'} |"
        )

    if results:
        rotations = [r["rotation_deg"] for r in results]
        noises = [r["noise_sigma"] for r in results]
        blurs = [r["blur_laplacian_var"] for r in results]
        jpegs = [r["jpeg_quality"] for r in results if r["jpeg_quality"] is not None]
        lines += [
            "",
            "## Aggregate",
            "",
            f"- Rotation range: [{min(rotations):.2f}, {max(rotations):.2f}], median {np.median(rotations):.2f}",
            f"- Noise σ range: [{min(noises):.4f}, {max(noises):.4f}], median {np.median(noises):.4f}",
            f"- Blur var range: [{min(blurs):.1f}, {max(blurs):.1f}], median {np.median(blurs):.1f}",
        ]
        if jpegs:
            lines.append(f"- JPEG quality range: [{min(jpegs)}, {max(jpegs)}], median {int(np.median(jpegs))}")

        lines += [
            "",
            "## Recommended degradation pipeline parameters",
            "",
            "Set `src/data/scan_degradation.py` defaults to span the observed distribution:",
            f"- Rotation: ±{max(abs(min(rotations)), abs(max(rotations))) + 0.3:.1f}° uniform",
            f"- Noise σ: uniform [{max(0.01, min(noises)):.3f}, {max(noises) + 0.02:.3f}]",
            f"- Blur kernel σ: scaled so output blur_laplacian_var matches [{min(blurs):.0f}, {max(blurs):.0f}]",
        ]
        if jpegs:
            lines.append(f"- JPEG quality: uniform [{max(50, min(jpegs) - 5)}, {min(95, max(jpegs) + 5)}]")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
