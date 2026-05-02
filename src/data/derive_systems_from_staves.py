"""Spatial-grouping heuristic to derive system-level bboxes from per-staff YOLO labels.

Used for AudioLabs v2 (real scans) and sparse_augment corpora where Verovio's
authoritative system markup is not available.

The grouping uses per-page gap clustering: for pages with ≥2 inter-staff gaps,
sort the gaps and find the largest "jump" between consecutive sorted values.
If the jump is ≥ ``bimodal_jump_ratio`` (default 1.5x), use the midpoint as
the system-boundary threshold. This naturally separates intra-system gaps
(small, between staves of one system) from inter-system gaps (large, between
systems). Pages without a clean bimodal split fall back to the factor-based
threshold ``vertical_gap_factor × avg_staff_height``.

This handles piano-vocal lieder (3 staves per system, intra ~30-40 px,
inter ~50-80 px — gaps are too close for a fixed factor to distinguish) and
SATB choral (uniform large gaps — bimodal-detection misses, fallback merges).
"""
from __future__ import annotations

from typing import Iterable, List, Dict, Optional


def _detect_bimodal_threshold(
    gaps: List[float],
    bimodal_jump_ratio: float,
) -> Optional[float]:
    """Find a per-page system-boundary threshold from the gap distribution.

    Returns the midpoint between the two clusters when the largest jump in
    sorted gaps is >= bimodal_jump_ratio. Returns None when the gaps are
    monomodal (no clear split — caller should fall back to factor heuristic).
    """
    sorted_gaps = sorted(gaps)
    n = len(sorted_gaps)
    if n < 2:
        return None
    best_ratio = 1.0
    best_split_idx = -1
    for i in range(n - 1):
        if sorted_gaps[i] <= 0:
            continue
        ratio = sorted_gaps[i + 1] / sorted_gaps[i]
        if ratio > best_ratio:
            best_ratio = ratio
            best_split_idx = i
    if best_split_idx >= 0 and best_ratio >= bimodal_jump_ratio:
        return (sorted_gaps[best_split_idx] + sorted_gaps[best_split_idx + 1]) / 2.0
    return None


def group_staves_into_systems(
    staves: Iterable[Dict],
    vertical_gap_factor: float = 2.5,
    x_overlap_threshold: float = 0.80,
    bimodal_jump_ratio: float = 1.5,
) -> List[Dict]:
    """Group per-staff bboxes into system bboxes by spatial proximity.

    Each input staff dict must contain ``"bbox"`` as ``(x1, y1, x2, y2)`` in pixel coords.
    Returns a list of system dicts, each containing:
      - ``"bbox"``: ``(x1, y1, x2, y2)`` — bounding box of the union of staves in the system
      - ``"staves_in_system"``: int — number of staves grouped into this system
    Output is sorted by y1 (top-to-bottom).

    Adaptive threshold:
      - For each page, compute all consecutive (sorted-by-y) inter-staff gaps.
      - If gap distribution is bimodal (largest sorted-gap jump >= bimodal_jump_ratio),
        use the midpoint as the system-boundary threshold (per-page adaptive).
      - Otherwise fall back to ``vertical_gap_factor * avg_staff_height``.
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
    threshold = _detect_bimodal_threshold(gaps, bimodal_jump_ratio)
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
