"""Spatial-grouping heuristic to derive system-level bboxes from per-staff YOLO labels.

Used for AudioLabs v2 (real scans) and sparse_augment corpora where Verovio's
authoritative system markup is not available.

Two-stage threshold for per-page adaptive grouping:

1. **Median-based** (primary, used when ≥3 gaps): threshold = 1.3 × median(gaps).
   On pages with bimodal gap distributions (intra-system gaps small + inter-system
   gaps large), the median sits at an intra-system gap, so 1.3× of it lands in
   the gap "valley" and cleanly splits inter-system boundaries. Robust to gap
   variance within a system.

2. **Factor fallback** (used when ≤2 gaps): threshold = vertical_gap_factor *
   avg_staff_height. Few-gap pages don't have enough samples for a stable median.

Both stages handle the SATB-choral edge case (uniform large gaps) by leaving
all staves in one system: when median ≈ inter-system gap, 1.3× exceeds the
actual gap and no splits are made — the heuristic can't distinguish "one wide
system" from "many adjacent systems" without bracket detection.
"""
from __future__ import annotations

from typing import Iterable, List, Dict, Optional


def _median(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _adaptive_threshold(
    gaps: List[float],
    median_factor: float,
) -> Optional[float]:
    """Per-page system-boundary threshold from the gap distribution.

    Returns ``median_factor × median(gaps)`` when there are enough gaps to
    estimate a stable median (≥ 3). Returns None otherwise so the caller can
    fall back to a fixed-factor heuristic.
    """
    if len(gaps) < 3:
        return None
    return median_factor * _median(gaps)


def group_staves_into_systems(
    staves: Iterable[Dict],
    vertical_gap_factor: float = 2.5,
    x_overlap_threshold: float = 0.80,
    median_factor: float = 1.3,
) -> List[Dict]:
    """Group per-staff bboxes into system bboxes by spatial proximity.

    Each input staff dict must contain ``"bbox"`` as ``(x1, y1, x2, y2)`` in pixel coords.
    Returns a list of system dicts, each containing:
      - ``"bbox"``: ``(x1, y1, x2, y2)`` — bounding box of the union of staves in the system
      - ``"staves_in_system"``: int — number of staves grouped into this system
    Output is sorted by y1 (top-to-bottom).

    Adaptive threshold (per-page):
      - ≥3 gaps: ``median_factor × median(gaps)``. This sits between intra and
        inter-system gaps for pages with bimodal distributions.
      - ≤2 gaps: ``vertical_gap_factor × avg_staff_height`` (fallback).
    """
    sorted_staves = sorted(staves, key=lambda s: s["bbox"][1])
    n = len(sorted_staves)
    if n == 0:
        return []
    if n == 1:
        x1, y1, x2, y2 = sorted_staves[0]["bbox"]
        return [{"bbox": (x1, y1, x2, y2), "staves_in_system": 1}]

    # Compute consecutive gaps + average staff height
    gaps: List[float] = []
    for i in range(n - 1):
        _sx1, sy1, _sx2, sy2 = sorted_staves[i]["bbox"]
        _nx1, ny1, _nx2, _ny2 = sorted_staves[i + 1]["bbox"]
        gaps.append(max(0.0, ny1 - sy2))

    avg_h = sum((s["bbox"][3] - s["bbox"][1]) for s in sorted_staves) / n
    threshold = _adaptive_threshold(gaps, median_factor)
    if threshold is None:
        threshold = vertical_gap_factor * avg_h

    # Walk staves in order; start new system when gap >= threshold OR x-overlap fails
    systems: List[List[Dict]] = [[sorted_staves[0]]]
    for i in range(1, n):
        cur = sorted_staves[i]
        prev = systems[-1][-1]
        sx1, sy1, sx2, _sy2 = cur["bbox"]
        lx1, _ly1, lx2, ly2 = prev["bbox"]

        gap = sy1 - ly2
        x_overlap_start = max(sx1, lx1)
        x_overlap_end = min(sx2, lx2)
        x_overlap = max(0.0, x_overlap_end - x_overlap_start)
        larger_width = max(sx2 - sx1, lx2 - lx1)
        x_overlap_frac = x_overlap / max(larger_width, 1.0)

        if gap < threshold and x_overlap_frac >= x_overlap_threshold:
            systems[-1].append(cur)
        else:
            systems.append([cur])

    out = []
    for system_staves in systems:
        x1 = min(s["bbox"][0] for s in system_staves)
        y1 = min(s["bbox"][1] for s in system_staves)
        x2 = max(s["bbox"][2] for s in system_staves)
        y2 = max(s["bbox"][3] for s in system_staves)
        out.append({
            "bbox": (x1, y1, x2, y2),
            "staves_in_system": len(system_staves),
        })
    return out
