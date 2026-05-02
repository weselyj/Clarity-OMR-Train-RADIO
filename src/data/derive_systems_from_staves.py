"""Spatial-grouping heuristic to derive system-level bboxes from per-staff YOLO labels.

Used for AudioLabs v2 (real scans) and sparse_augment corpora where Verovio's
authoritative system markup is not available.

Per-page adaptive grouping in two stages:

1. **Local-maxima + median** (primary, used when ≥3 gaps): split at gaps that
   are BOTH (a) larger than ``median_factor × median(gaps)`` AND (b) local
   maxima (strictly greater than both neighbors). The median requirement filters
   out tiny "wiggles" between similar small gaps; the local-max requirement
   captures the actual inter-system boundaries even when their absolute values
   overlap with intra-system gap variance (the common Schubert lieder failure
   mode where intra ~30-40 px and inter ~50-60 px).

2. **Factor fallback** (used when ≤2 gaps): ``vertical_gap_factor *
   avg_staff_height``. Few-gap pages don't support stable peak detection.

Pages with uniform gap distributions (e.g., SATB choral with even spacing)
have no local maxima → no splits → one system. The heuristic cannot solve
these without bracket detection.
"""
from __future__ import annotations

from typing import Iterable, List, Dict


def _median(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _local_max_split_indices(gaps: List[float], min_gap: float) -> set:
    """Return indices of gaps that are local maxima AND > min_gap.

    A local max is strictly greater than its neighbors. Endpoints count if
    they're larger than their single neighbor.
    """
    n = len(gaps)
    splits: set = set()
    for i, g in enumerate(gaps):
        if g <= min_gap:
            continue
        left_ok = (i == 0) or (g > gaps[i - 1])
        right_ok = (i == n - 1) or (g > gaps[i + 1])
        if left_ok and right_ok:
            splits.add(i)
    return splits


def group_staves_into_systems(
    staves: Iterable[Dict],
    vertical_gap_factor: float = 2.5,
    x_overlap_threshold: float = 0.80,
    median_factor: float = 1.0,
) -> List[Dict]:
    """Group per-staff bboxes into system bboxes by spatial proximity.

    Each input staff dict must contain ``"bbox"`` as ``(x1, y1, x2, y2)`` in pixel coords.
    Returns a list of system dicts, each containing:
      - ``"bbox"``: ``(x1, y1, x2, y2)`` — bounding box of the union of staves in the system
      - ``"staves_in_system"``: int — number of staves grouped into this system
    Output is sorted by y1 (top-to-bottom).

    Per-page adaptive split detection (≥3 gaps): split at indices that are
    local maxima in the gap sequence AND larger than ``median_factor × median(gaps)``.
    Few-gap pages (≤2) fall back to ``vertical_gap_factor × avg_staff_height``.
    """
    sorted_staves = sorted(staves, key=lambda s: s["bbox"][1])
    n = len(sorted_staves)
    if n == 0:
        return []
    if n == 1:
        x1, y1, x2, y2 = sorted_staves[0]["bbox"]
        return [{"bbox": (x1, y1, x2, y2), "staves_in_system": 1}]

    gaps: List[float] = []
    for i in range(n - 1):
        _sx1, sy1, _sx2, sy2 = sorted_staves[i]["bbox"]
        _nx1, ny1, _nx2, _ny2 = sorted_staves[i + 1]["bbox"]
        gaps.append(max(0.0, ny1 - sy2))

    avg_h = sum((s["bbox"][3] - s["bbox"][1]) for s in sorted_staves) / n
    if len(gaps) >= 3:
        split_indices = _local_max_split_indices(
            gaps, min_gap=median_factor * _median(gaps),
        )
    else:
        # Few-gap fallback: simple per-pair factor heuristic
        threshold = vertical_gap_factor * avg_h
        split_indices = {i for i, g in enumerate(gaps) if g >= threshold}

    systems: List[List[Dict]] = [[sorted_staves[0]]]
    for i in range(1, n):
        cur = sorted_staves[i]
        prev = systems[-1][-1]
        sx1, sy1, sx2, _sy2 = cur["bbox"]
        lx1, _ly1, lx2, ly2 = prev["bbox"]

        x_overlap_start = max(sx1, lx1)
        x_overlap_end = min(sx2, lx2)
        x_overlap = max(0.0, x_overlap_end - x_overlap_start)
        larger_width = max(sx2 - sx1, lx2 - lx1)
        x_overlap_frac = x_overlap / max(larger_width, 1.0)

        gap_idx = i - 1  # the gap between prev and cur
        is_split = (gap_idx in split_indices) or (x_overlap_frac < x_overlap_threshold)
        if is_split:
            systems.append([cur])
        else:
            systems[-1].append(cur)

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
