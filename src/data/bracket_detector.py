"""Visual bracket/brace detection for grouping staves into systems.

Heuristic spatial grouping of staves (used for AudioLabs and sparse_augment) hits
a fundamental limit: vocal-to-piano gaps can exceed inter-system gaps, causing
brackets to be split across labels. The fix is to *see* the bracket itself —
brackets and braces are vertical structures on the LEFT side of every system,
spanning exactly the staves they group.

Pipeline:
1. ``detect_brackets_on_page(image_path)`` -> list of bracket regions
   {"x", "y_top", "y_bottom", "height"} sorted top-to-bottom. Uses OpenCV
   morphological vertical-line extraction on the leftmost ~10% of the page.
2. ``group_staves_by_brackets(staves, brackets)`` -> system bboxes by assigning
   each staff to the bracket whose y-range contains the staff's y-center.
   Bracket position becomes the bbox left edge (no need for the leftward margin
   hack any more); bracket vertical extent is the y-range source of truth.
3. Falls back to spatial heuristic when no brackets are detected (e.g.,
   vocal-only single-staff "systems" with no bracket).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

# Defaults derived from typical Verovio + scan layouts:
# - Brackets/braces sit within the leftmost ~10% of page width
# - Minimum bracket height is ~half a staff height (so a brace on a single
#   piano grand staff is detectable); skip shorter components as noise
# - Aspect ratio (h/w) >= 5 is "vertical enough"; brackets are tall thin lines
DEFAULT_SEARCH_LEFT_FRAC = 0.10
DEFAULT_THRESHOLD = 200  # ink pixels are darker than this on a 0-255 scale
DEFAULT_MIN_HEIGHT_FRAC = 0.02  # of page height
DEFAULT_MIN_ASPECT = 4.0  # height / width


def detect_brackets_on_page(
    image_path: Path,
    *,
    search_left_frac: float = DEFAULT_SEARCH_LEFT_FRAC,
    threshold: int = DEFAULT_THRESHOLD,
    min_height_frac: float = DEFAULT_MIN_HEIGHT_FRAC,
    min_aspect: float = DEFAULT_MIN_ASPECT,
) -> List[Dict]:
    """Detect vertical brackets/braces on the left side of a music page.

    Returns a list of bracket dicts: {"x", "y_top", "y_bottom", "height"} in
    page-pixel coordinates, sorted top-to-bottom. Empty list if no brackets
    are detected (page may have unbracketed systems → caller falls back).
    """
    import cv2
    import numpy as np  # noqa: F401  (used implicitly via cv2)

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, w = img.shape

    search_width = max(8, int(w * search_left_frac))
    left_strip = img[:, :search_width]

    # Binary: ink (dark) -> white (255) on black bg
    _, binary = cv2.threshold(left_strip, threshold, 255, cv2.THRESH_BINARY_INV)

    # Vertical-line morphology: tall thin kernel. Anything that survives is
    # a vertical structure (bracket spine, brace, barline, slur — most are filtered
    # later by the leftmost-x check or aspect ratio).
    vert_kernel_h = max(20, int(h * min_height_frac * 0.6))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_h))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=4)

    brackets = []
    min_h_px = max(8, int(h * min_height_frac))
    for label_idx in range(1, n_labels):  # 0 is background
        x, y, w_box, h_box, _area = stats[label_idx]
        if h_box < min_h_px:
            continue
        if h_box / max(w_box, 1) < min_aspect:
            continue
        brackets.append({
            "x": float(x),
            "y_top": float(y),
            "y_bottom": float(y + h_box),
            "height": float(h_box),
        })

    brackets.sort(key=lambda b: b["y_top"])
    # Conservative merge: only bridge tiny anti-aliasing gaps. Larger gaps
    # are likely separate brackets for different systems; merging them
    # over-merges adjacent systems on tightly-packed lieder pages.
    brackets = _merge_close_brackets(
        brackets,
        x_tolerance_px=max(4, int(w * 0.008)),
        y_gap_tolerance_px=max(10, int(h * 0.012)),
    )
    return brackets


def _merge_close_brackets(
    brackets: List[Dict],
    *,
    x_tolerance_px: int,
    y_gap_tolerance_px: int,
) -> List[Dict]:
    """Merge brackets at similar x with small y-gap (likely one bracket split by anti-aliasing)."""
    if not brackets:
        return []
    # Group by approximate x cluster
    by_x: Dict[int, List[Dict]] = {}
    for b in sorted(brackets, key=lambda b: b["x"]):
        # Bucket by x_tolerance_px
        placed = False
        for key in list(by_x.keys()):
            if abs(b["x"] - key) <= x_tolerance_px:
                by_x[key].append(b)
                placed = True
                break
        if not placed:
            by_x[int(b["x"])] = [b]

    merged: List[Dict] = []
    for x_key, group in by_x.items():
        group.sort(key=lambda b: b["y_top"])
        cur = dict(group[0])
        for nxt in group[1:]:
            gap = nxt["y_top"] - cur["y_bottom"]
            if gap <= y_gap_tolerance_px:
                cur["y_bottom"] = max(cur["y_bottom"], nxt["y_bottom"])
                cur["height"] = cur["y_bottom"] - cur["y_top"]
            else:
                merged.append(cur)
                cur = dict(nxt)
        merged.append(cur)

    merged.sort(key=lambda b: b["y_top"])
    return merged


def group_staves_by_brackets(
    staves: Sequence[Dict],
    brackets: Sequence[Dict],
    *,
    upward_attach_max_gap_frac: float = 1.5,
) -> List[Dict]:
    """Group staves by which bracket's y-range contains each staff's y-center.

    Two assignment passes:
      1. Containment: staves whose y-center falls within a bracket's y-range
         are assigned to that bracket.
      2. Upward attachment: a staff sitting just ABOVE a bracket (no other
         bracket contains it, gap < ``upward_attach_max_gap_frac × staff_height``)
         is attached to the bracket below. This handles lieder convention
         where the brace covers piano grand staff only and the vocal sits above.

    Staves with no nearby bracket become their own 1-staff systems
    (vocal-only or ungrouped).

    Returns a list of system dicts: {"bbox": (x1, y1, x2, y2), "staves_in_system"}.
    """
    if not staves:
        return []
    sorted_staves = sorted(staves, key=lambda s: s["bbox"][1])
    if not brackets:
        return [
            {
                "bbox": tuple(s["bbox"]),
                "staves_in_system": 1,
            }
            for s in sorted_staves
        ]

    by_bracket: Dict[int, List[Dict]] = {}
    unassigned_pass1: List[Dict] = []
    for staff in sorted_staves:
        x1, y1, x2, y2 = staff["bbox"]
        y_center = (y1 + y2) / 2
        assigned_idx = None
        for b_idx, b in enumerate(brackets):
            if b["y_top"] <= y_center <= b["y_bottom"]:
                assigned_idx = b_idx
                break
        if assigned_idx is None:
            unassigned_pass1.append(staff)
        else:
            by_bracket.setdefault(assigned_idx, []).append(staff)

    # Pass 2: upward attachment. For each unassigned staff, look for a bracket
    # below within reasonable distance. Lieder vocal staff sits just above the
    # piano brace and isn't structurally contained.
    unassigned_pass2: List[Dict] = []
    for staff in unassigned_pass1:
        _x1, y1, _x2, y2 = staff["bbox"]
        staff_h = max(1.0, y2 - y1)
        max_gap = upward_attach_max_gap_frac * staff_h
        attached_idx = None
        # Sort brackets by y_top, find the first one whose y_top is below this staff
        for b_idx, b in enumerate(brackets):
            gap = b["y_top"] - y2
            if 0 <= gap <= max_gap:
                attached_idx = b_idx
                break
        if attached_idx is None:
            unassigned_pass2.append(staff)
        else:
            by_bracket.setdefault(attached_idx, []).append(staff)

    out: List[Dict] = []
    for b_idx in sorted(by_bracket.keys()):
        sub_staves = by_bracket[b_idx]
        b = brackets[b_idx]
        x1 = min(s["bbox"][0] for s in sub_staves)
        y1 = min(s["bbox"][1] for s in sub_staves)
        x2 = max(s["bbox"][2] for s in sub_staves)
        y2 = max(s["bbox"][3] for s in sub_staves)
        x1 = min(x1, b["x"])
        y1 = min(y1, b["y_top"])
        y2 = max(y2, b["y_bottom"])
        out.append({
            "bbox": (x1, y1, x2, y2),
            "staves_in_system": len(sub_staves),
        })

    for staff in unassigned_pass2:
        x1, y1, x2, y2 = staff["bbox"]
        out.append({"bbox": (x1, y1, x2, y2), "staves_in_system": 1})

    out.sort(key=lambda s: s["bbox"][1])
    return out
