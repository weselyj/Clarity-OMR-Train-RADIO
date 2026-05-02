"""Spatial-grouping heuristic to derive system-level bboxes from per-staff YOLO labels.

Used for AudioLabs v2 (real scans) and sparse_augment corpora where Verovio's
authoritative system markup is not available.

Heuristic: group adjacent staves vertically (gap < vertical_gap_factor * staff_height) AND
x-extent overlap >= x_overlap_threshold. Produces one system bbox per group, sorted top-to-bottom.
"""
from __future__ import annotations

from typing import Iterable, List, Dict


def group_staves_into_systems(
    staves: Iterable[Dict],
    vertical_gap_factor: float = 1.5,
    x_overlap_threshold: float = 0.80,
) -> List[Dict]:
    """Group per-staff bboxes into system bboxes by spatial proximity.

    Each input staff dict must contain ``"bbox"`` as ``(x1, y1, x2, y2)`` in pixel coords.
    Returns a list of system dicts, each containing:
      - ``"bbox"``: ``(x1, y1, x2, y2)`` — bounding box of the union of staves in the system
      - ``"staves_in_system"``: int — number of staves grouped into this system
    Output is sorted by y1 (top-to-bottom).
    """
    sorted_staves = sorted(staves, key=lambda s: s["bbox"][1])

    systems: List[List[Dict]] = []
    for staff in sorted_staves:
        sx1, sy1, sx2, sy2 = staff["bbox"]
        staff_h = sy2 - sy1

        if systems:
            last_system = systems[-1]
            last_staff = last_system[-1]
            lx1, ly1, lx2, ly2 = last_staff["bbox"]
            last_h = ly2 - ly1
            avg_h = (staff_h + last_h) / 2

            vertical_gap = sy1 - ly2

            x_overlap_start = max(sx1, lx1)
            x_overlap_end = min(sx2, lx2)
            x_overlap = max(0, x_overlap_end - x_overlap_start)
            # Larger width as denominator: a small staff inset into a wider one
            # should NOT be grouped (treat as a partial-width inset/branch system).
            larger_width = max(sx2 - sx1, lx2 - lx1)
            x_overlap_frac = x_overlap / max(larger_width, 1)

            if vertical_gap < vertical_gap_factor * avg_h and x_overlap_frac >= x_overlap_threshold:
                last_system.append(staff)
                continue

        systems.append([staff])

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
