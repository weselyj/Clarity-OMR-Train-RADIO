"""Visual system-delimiter detection for grouping staves into systems.

Heuristic spatial grouping of staves (used for AudioLabs and sparse_augment) hits
a fundamental limit: vocal-to-piano gaps can exceed inter-system gaps. The fix
is to *see* a visual signal that consistently marks system boundaries.

We use the **first barline of each system**: in standard music engraving, the
first barline (just after the clef/key/time-signature, before measure 1) always
spans ALL staves of a system, regardless of bracket/brace style above it. More
reliable than detecting brackets directly because:
  - Lieder vocal+piano grand staff: bracket only covers piano, but first barline
    spans vocal+piano.
  - SATB choral: bracket spans all staves; first barline does too.
  - Solo piano: brace only; first barline spans both staves.
  - Orchestral: nested brackets/braces; first barline spans all.

Pipeline:
1. ``detect_brackets_on_page(image_path, staves)`` -> list of system delimiters
   {"x", "y_top", "y_bottom", "height"}. Looks for vertical lines at x ~ leftmost
   staff start, where the first barline of each system sits. Uses ``staves`` to
   know where to look.
2. ``group_staves_by_brackets(staves, delimiters)`` -> system bboxes by assigning
   each staff to the delimiter whose y-range overlaps the staff most. Falls back
   to per-staff-as-system when no delimiters detected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_THRESHOLD = 200  # ink pixels are darker than this on a 0-255 scale
DEFAULT_MIN_ASPECT = 4.0  # height / width: barlines are tall thin lines
DEFAULT_BARLINE_SEARCH_WIDTH_FRAC = 0.020  # ±2% of page width around staff x_min


def detect_brackets_on_page(
    image_path: Path,
    staves: Sequence[Dict] = (),
    *,
    threshold: int = DEFAULT_THRESHOLD,
    min_aspect: float = DEFAULT_MIN_ASPECT,
    barline_search_width_frac: float = DEFAULT_BARLINE_SEARCH_WIDTH_FRAC,
) -> List[Dict]:
    """Detect vertical system delimiters (first-barlines) on a music page.

    Looks for vertical lines at x ~= leftmost staff start, where the first
    barline of each system sits. Each detected line spans the full vertical
    extent of one system (engraving convention).

    ``staves`` is the list of per-staff bboxes (each {"bbox": (x1, y1, x2, y2)}
    in pixel coords). Used to find the leftmost staff x and median staff height.

    Returns a list of delimiter dicts {"x", "y_top", "y_bottom", "height"}
    sorted top-to-bottom. Empty list if no delimiters detected.
    """
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, w = img.shape

    if not staves:
        return []

    staff_xmins = [s["bbox"][0] for s in staves]
    staff_heights = sorted(s["bbox"][3] - s["bbox"][1] for s in staves)
    staff_xmin = min(staff_xmins)
    median_staff_h = staff_heights[len(staff_heights) // 2]

    # Crop a thin vertical strip at staff_xmin where the first barline sits.
    delta = max(8, int(w * barline_search_width_frac))
    x_lo = max(0, int(staff_xmin) - delta)
    x_hi = min(w, int(staff_xmin) + delta)
    if x_hi <= x_lo:
        return []
    strip = img[:, x_lo:x_hi]

    _, binary = cv2.threshold(strip, threshold, 255, cv2.THRESH_BINARY_INV)

    # Kernel >= 1.5x median staff height: only catches system-spanning lines,
    # not single-staff barlines or short vertical decorations.
    vert_kernel_h = max(20, int(median_staff_h * 1.5))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_h))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=4)

    delimiters = []
    # Lower bound: 0.9x median staff height. Catches small braces (single-system
    # piano grand staff) but still rejects spurious short verticals (note stems,
    # measure barlines).
    min_h_px = max(15, int(median_staff_h * 0.9))
    for label_idx in range(1, n_labels):
        x, y, w_box, h_box, _area = stats[label_idx]
        if h_box < min_h_px:
            continue
        if h_box / max(w_box, 1) < min_aspect:
            continue
        delimiters.append({
            "x": float(x_lo + x),  # translate strip x back to page x
            "y_top": float(y),
            "y_bottom": float(y + h_box),
            "height": float(h_box),
        })

    delimiters.sort(key=lambda b: b["y_top"])
    delimiters = _merge_close_brackets(
        delimiters,
        x_tolerance_px=max(4, int(w * 0.008)),
        y_gap_tolerance_px=max(10, int(median_staff_h * 0.5)),
    )
    return delimiters


def _merge_close_brackets(
    brackets: List[Dict],
    *,
    x_tolerance_px: int,
    y_gap_tolerance_px: int,
) -> List[Dict]:
    """Merge brackets at similar x with small y-gap (one bracket split by anti-aliasing)."""
    if not brackets:
        return []
    by_x: Dict[int, List[Dict]] = {}
    for b in sorted(brackets, key=lambda b: b["x"]):
        placed = False
        for key in list(by_x.keys()):
            if abs(b["x"] - key) <= x_tolerance_px:
                by_x[key].append(b)
                placed = True
                break
        if not placed:
            by_x[int(b["x"])] = [b]

    merged: List[Dict] = []
    for _x_key, group in by_x.items():
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
    upward_attach_max_gap_factor: float = 2.5,
) -> List[Dict]:
    """Group staves by which delimiter's y-range overlaps the staff.

    Two-pass assignment:
      1. **Overlap**: assign each staff to the delimiter with most y-overlap.
      2. **Upward attach**: for unassigned staves, look for a delimiter BELOW
         within ``upward_attach_max_gap_factor × staff_height``. This handles
         vocal staves that sit above piano-only braces (lieder/choral
         convention where the first barline only spans piano staves).

    Each system bbox extends:
      - Left to the delimiter's x position
      - Vertically to MIN(staff y_top, delimiter y_top) and MAX(staff y_bottom, delimiter y_bottom)

    Staves with no nearby delimiter become their own 1-staff systems.
    """
    if not staves:
        return []
    sorted_staves = sorted(staves, key=lambda s: s["bbox"][1])
    if not brackets:
        return [
            {"bbox": tuple(s["bbox"]), "staves_in_system": 1}
            for s in sorted_staves
        ]

    by_bracket: Dict[int, List[Dict]] = {}
    unassigned_pass1: List[Dict] = []
    for staff in sorted_staves:
        _x1, y1, _x2, y2 = staff["bbox"]
        best_idx = None
        best_overlap = 0.0
        for b_idx, b in enumerate(brackets):
            ov_top = max(y1, b["y_top"])
            ov_bot = min(y2, b["y_bottom"])
            overlap = max(0.0, ov_bot - ov_top)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = b_idx
        if best_idx is None or best_overlap <= 0:
            unassigned_pass1.append(staff)
        else:
            by_bracket.setdefault(best_idx, []).append(staff)

    # Pass 2: upward attach for orphaned staves above a delimiter.
    unassigned_pass2: List[Dict] = []
    for staff in unassigned_pass1:
        _x1, y1, _x2, y2 = staff["bbox"]
        staff_h = max(1.0, y2 - y1)
        max_gap = upward_attach_max_gap_factor * staff_h
        attached_idx = None
        # Find first delimiter whose y_top is below the staff's y_bottom (within max_gap)
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

    # For staves still unassigned after both passes, run spatial-grouping on
    # the remainder so missed-bracket systems are still grouped (e.g., Wagner
    # title page where the first system's brace isn't detected).
    if unassigned_pass2:
        from src.data.derive_systems_from_staves import group_staves_into_systems
        spatial_systems = group_staves_into_systems(unassigned_pass2)
        out.extend(spatial_systems)

    out.sort(key=lambda s: s["bbox"][1])
    return out
