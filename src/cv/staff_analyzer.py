"""CV-based staff crop analyzer for OMR prior extraction.

Uses homr's segnet model for notehead detection (semantic segmentation),
combined with traditional CV for staff lines and barlines.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.cv.priors import (
    BarlineDetection,
    NoteheadDetection,
    OnsetCluster,
    MeasureSkeleton,
    StaffLineInfo,
    StaffSkeleton,
    staff_position_to_pitch,
)

# ---------------------------------------------------------------------------
# Segnet singleton (lazy-loaded to avoid startup cost)
# ---------------------------------------------------------------------------

_segnet_model = None


def _get_segnet():
    """Lazy-load the homr segnet ONNX model (singleton)."""
    global _segnet_model
    if _segnet_model is not None:
        return _segnet_model

    # Add homr-main to path so we can import its segmentation module
    homr_root = os.path.join(os.path.dirname(__file__), "..", "..", "homr-main")
    homr_root = os.path.abspath(homr_root)
    if homr_root not in sys.path:
        sys.path.insert(0, homr_root)

    from homr.segmentation.inference_segnet import Segnet

    _segnet_model = Segnet(use_gpu_inference=False)
    return _segnet_model


@dataclass
class SegnetMasks:
    """All semantic segmentation masks from segnet."""
    noteheads: np.ndarray    # class 2: binary uint8, 255 = notehead
    clefs_keys: np.ndarray   # class 3: binary uint8, 255 = clef/key/time symbol
    stems_rests: np.ndarray  # class 1: binary uint8, 255 = stem or rest


def _run_segnet(gray: np.ndarray, step_size: int = 160, win_size: int = 320) -> SegnetMasks:
    """Run segnet on a grayscale image and return all semantic masks.

    Uses 50% overlapping patches (step_size=160) for better boundary accuracy.
    Returns SegnetMasks with noteheads, clefs_keys, and stems_rests masks.
    """
    model = _get_segnet()

    # Segnet expects grayscale input, converts internally to BGR
    image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image_chw = np.transpose(image_bgr, (2, 0, 1)).astype(np.float32)

    c, h, w = image_chw.shape
    patches: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    batch_size = 16

    from homr.segmentation.inference_segnet import extract_patch

    for y_loop in range(0, max(h, win_size), step_size):
        y = min(y_loop, h - win_size)
        for x_loop in range(0, max(w, win_size), step_size):
            x = min(x_loop, w - win_size)
            hop = extract_patch(image_chw, y, x, win_size)
            batch.append(hop)
            if len(batch) == batch_size:
                batch_out = model.run(np.stack(batch, axis=0))
                for out in batch_out:
                    patches.append(np.argmax(out, axis=0))
                batch.clear()

    if batch:
        batch_out = model.run(np.stack(batch, axis=0))
        for out in batch_out:
            patches.append(np.argmax(out, axis=0))

    # Merge patches
    from homr.segmentation.inference_segnet import merge_patches
    merged = merge_patches(patches, (h, w), win_size, step_size)

    # Extract individual class masks
    notehead_mask = np.where(merged == 2, 255, 0).astype(np.uint8)
    clefs_keys_mask = np.where(merged == 3, 255, 0).astype(np.uint8)
    stems_rests_mask = np.where(merged == 1, 255, 0).astype(np.uint8)

    return SegnetMasks(
        noteheads=notehead_mask,
        clefs_keys=clefs_keys_mask,
        stems_rests=stems_rests_mask,
    )


# ---------------------------------------------------------------------------
# Staff line detection
# ---------------------------------------------------------------------------

def detect_staff_lines(gray: np.ndarray, *, min_line_density: float = 0.3) -> List[int]:
    """Detect staff line y-positions via horizontal projection.

    Returns y-coordinates of detected staff lines (expect 5 per staff).
    """
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = np.sum(binary > 0, axis=1).astype(np.float64)
    projection /= max(w, 1)

    # Staff lines have high horizontal density
    threshold = max(min_line_density, np.mean(projection) + 1.5 * np.std(projection))
    line_mask = projection > threshold

    # Group consecutive True rows into line segments
    lines: List[int] = []
    in_line = False
    start = 0
    for row in range(h):
        if line_mask[row] and not in_line:
            start = row
            in_line = True
        elif not line_mask[row] and in_line:
            center = (start + row - 1) // 2
            lines.append(center)
            in_line = False
    if in_line:
        lines.append((start + h - 1) // 2)

    # If we got more than 5 lines, keep the 5 most evenly spaced
    if len(lines) > 5:
        lines = _select_best_5_lines(lines)

    return lines


def _select_best_5_lines(candidates: List[int]) -> List[int]:
    """From a set of candidate lines, pick the 5 that form the most regular staff."""
    if len(candidates) <= 5:
        return candidates
    best_score = float("inf")
    best_group: List[int] = candidates[:5]
    n = len(candidates)
    # Try all combinations of 5 consecutive candidates
    for start in range(n - 4):
        group = candidates[start : start + 5]
        spacings = [group[i + 1] - group[i] for i in range(4)]
        mean_sp = sum(spacings) / 4.0
        variance = sum((s - mean_sp) ** 2 for s in spacings)
        if variance < best_score:
            best_score = variance
            best_group = group
    return best_group


def estimate_staff_info(lines: List[int]) -> Optional[StaffLineInfo]:
    """Build StaffLineInfo from detected line positions."""
    if len(lines) < 3:
        return None
    spacings = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
    avg_spacing = sum(spacings) / len(spacings)
    return StaffLineInfo(
        y_positions=lines,
        spacing=avg_spacing,
        top=lines[0],
        bottom=lines[-1],
    )


# ---------------------------------------------------------------------------
# Barline detection
# ---------------------------------------------------------------------------

def detect_barlines(
    binary: np.ndarray,
    staff_info: StaffLineInfo,
    *,
    min_height_ratio: float = 0.6,
) -> List[BarlineDetection]:
    """Detect barlines as tall vertical structures spanning the staff."""
    staff_height = staff_info.bottom - staff_info.top
    min_h = int(staff_height * min_height_ratio)
    v_kernel_len = max(3, min_h)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(vertical)
    img_w = binary.shape[1]
    barlines: List[BarlineDetection] = []
    for i in range(1, num_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        cx = float(centroids[i][0])
        # Barlines are thin and tall
        if comp_h >= min_h and comp_w < staff_info.spacing * 0.8:
            # Skip if too close to image edges (likely staff-line remnants)
            if cx < staff_info.spacing * 0.5 or cx > img_w - staff_info.spacing * 0.5:
                continue
            confidence = min(1.0, comp_h / staff_height)
            barlines.append(BarlineDetection(
                x=cx,
                x_normalized=cx / max(img_w, 1),
                confidence=confidence,
            ))

    barlines.sort(key=lambda b: b.x)
    # Deduplicate barlines that are very close
    if len(barlines) > 1:
        barlines = _deduplicate_barlines(barlines, staff_info.spacing)
    return barlines


def _deduplicate_barlines(barlines: List[BarlineDetection], spacing: float) -> List[BarlineDetection]:
    merged: List[BarlineDetection] = [barlines[0]]
    for b in barlines[1:]:
        if b.x - merged[-1].x < spacing * 0.5:
            if b.confidence > merged[-1].confidence:
                merged[-1] = b
        else:
            merged.append(b)
    return merged


# ---------------------------------------------------------------------------
# Notehead detection via segnet
# ---------------------------------------------------------------------------

def _overlaps_mask(mask: np.ndarray, cx: float, cy: float, radius: float) -> bool:
    """Check if a circle around (cx, cy) overlaps with nonzero pixels in mask."""
    h, w = mask.shape[:2]
    x0 = max(0, int(cx - radius))
    x1 = min(w, int(cx + radius) + 1)
    y0 = max(0, int(cy - radius))
    y1 = min(h, int(cy + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(np.any(mask[y0:y1, x0:x1] > 0))


def _has_nearby_stem(stems_mask: np.ndarray, cx: float, cy: float, sp: float) -> bool:
    """Check if there is a stem/rest pixel within ~1.5 staff spacings of a notehead."""
    search_radius = sp * 1.5
    return _overlaps_mask(stems_mask, cx, cy, search_radius)


def detect_noteheads_segnet(
    gray: np.ndarray,
    staff_info: StaffLineInfo,
    *,
    min_area: int = 20,
    margin_above: float = 3.5,
    margin_below: float = 2.5,
    header_zone_ratio: float = 0.15,
) -> List[NoteheadDetection]:
    """Detect noteheads using homr's segnet semantic segmentation.

    Runs the segnet model to get all masks, then extracts individual noteheads
    via connected components with cross-mask validation:
    - Suppresses detections that overlap with clefs_keys mask
    - Validates detections have nearby stems/rests

    margin_above/margin_below are in staff spacings. Below is tighter (2.5)
    to avoid picking up chord symbol text below the staff.
    """
    sp = staff_info.spacing
    img_h, img_w = gray.shape[:2]

    # Run segnet to get all masks
    masks = _run_segnet(gray)

    # Connected components on the notehead mask
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(masks.noteheads)

    # Vertical ROI around the staff — tighter below to avoid chord text
    roi_top = max(0, staff_info.top - int(sp * margin_above))
    roi_bottom = min(img_h, staff_info.bottom + int(sp * margin_below))

    # Header zone: skip blobs in leftmost portion (clef/key/time symbols)
    header_x = int(img_w * header_zone_ratio)

    ref_area = sp * sp
    max_area = ref_area * 3.0

    noteheads: List[NoteheadDetection] = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        cy = float(centroids[i][1])
        if cy < roi_top or cy > roi_bottom:
            continue

        cx = float(centroids[i][0])
        if cx < header_x:
            continue

        comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])

        if area > max_area:
            continue

        # Reject very elongated blobs (stem/beam fragments)
        aspect = comp_w / max(comp_h, 1)
        if aspect < 0.3 or aspect > 3.5:
            continue
        if comp_h > sp * 2.5:
            continue

        # Suppress false positives that overlap with clef/key/time mask
        clef_overlap_radius = sp * 0.8
        if _overlaps_mask(masks.clefs_keys, cx, cy, clef_overlap_radius):
            # Check overlap ratio — if most of the blob's bbox is in the clefs zone, skip
            bx0 = int(stats[i, cv2.CC_STAT_LEFT])
            by0 = int(stats[i, cv2.CC_STAT_TOP])
            bx1 = bx0 + comp_w
            by1 = by0 + comp_h
            bx0c = max(0, min(bx0, img_w))
            bx1c = max(0, min(bx1, img_w))
            by0c = max(0, min(by0, img_h))
            by1c = max(0, min(by1, img_h))
            blob_region = masks.clefs_keys[by0c:by1c, bx0c:bx1c]
            if blob_region.size > 0:
                overlap_ratio = float(np.count_nonzero(blob_region)) / max(1, blob_region.size)
                if overlap_ratio > 0.3:
                    continue

        # Validate that a stem or rest is nearby (isolated blobs are likely noise)
        if not _has_nearby_stem(masks.stems_rests, cx, cy, sp):
            # Only reject if confidence would be low — strong detections survive
            if area < ref_area * 0.6:
                continue

        # Split wide blobs that likely contain multiple noteheads
        if comp_w > sp * 1.8 and area > ref_area * 0.8:
            sub_notes = _split_wide_blob(
                _labels, i, stats, staff_info, header_x, roi_top, roi_bottom
            )
            if sub_notes:
                noteheads.extend(sub_notes)
                continue

        staff_pos = _y_to_staff_position(cy, staff_info)
        blob_density = area / max(comp_w * comp_h, 1)
        is_filled = blob_density > 0.45

        confidence = min(1.0, 0.6 + area / (ref_area * 2.0))

        noteheads.append(NoteheadDetection(
            x=cx,
            y=cy,
            w=comp_w,
            h=comp_h,
            area=area,
            staff_position=staff_pos,
            confidence=confidence,
            is_filled=is_filled,
        ))

    noteheads.sort(key=lambda n: n.x)
    return noteheads


def _split_wide_blob(
    labels: np.ndarray,
    label_id: int,
    stats: np.ndarray,
    staff_info: StaffLineInfo,
    header_x: int,
    roi_top: int,
    roi_bottom: int,
) -> List[NoteheadDetection]:
    """Try to split a wide connected component into individual noteheads.

    Uses distance transform + watershed for robust separation of touching
    noteheads (e.g. seconds in chords), falling back to erosion if watershed
    doesn't produce a split.
    """
    sp = staff_info.spacing
    x0 = stats[label_id, cv2.CC_STAT_LEFT]
    y0 = stats[label_id, cv2.CC_STAT_TOP]
    w0 = stats[label_id, cv2.CC_STAT_WIDTH]
    h0 = stats[label_id, cv2.CC_STAT_HEIGHT]

    blob_mask = (labels[y0:y0+h0, x0:x0+w0] == label_id).astype(np.uint8) * 255

    # Try distance transform + watershed first
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    dist_threshold = max(0.3 * dist.max(), sp * 0.15)
    _, sure_fg = cv2.threshold(dist, dist_threshold, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    n_markers, marker_labels = cv2.connectedComponents(sure_fg)

    if n_markers > 2:
        # Watershed needs a 3-channel image
        blob_bgr = cv2.cvtColor(blob_mask, cv2.COLOR_GRAY2BGR)
        markers = marker_labels.astype(np.int32)
        # Mark unknown region (blob minus sure foreground) as 0
        unknown = cv2.subtract(blob_mask, sure_fg)
        markers[unknown == 255] = 0
        cv2.watershed(blob_bgr, markers)

        results: List[NoteheadDetection] = []
        ref_area = sp * sp
        for marker_id in range(1, n_markers):
            region = (markers == marker_id).astype(np.uint8)
            sub_area = int(np.sum(region))
            if sub_area < 15:
                continue
            ys, xs = np.where(region > 0)
            sub_cx = float(np.mean(xs)) + x0
            sub_cy = float(np.mean(ys)) + y0
            sub_w = int(np.max(xs) - np.min(xs) + 1)
            sub_h = int(np.max(ys) - np.min(ys) + 1)

            if sub_cy < roi_top or sub_cy > roi_bottom or sub_cx < header_x:
                continue

            staff_pos = _y_to_staff_position(sub_cy, staff_info)
            blob_density = sub_area / max(sub_w * sub_h, 1)
            is_filled = blob_density > 0.45
            confidence = min(1.0, 0.5 + sub_area / (ref_area * 2.0))

            results.append(NoteheadDetection(
                x=sub_cx, y=sub_cy, w=sub_w, h=sub_h,
                area=sub_area, staff_position=staff_pos,
                confidence=confidence, is_filled=is_filled,
            ))
        if results:
            return results

    # Fallback: erosion-based splitting
    erode_k = max(1, int(sp * 0.15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    eroded = cv2.erode(blob_mask, kernel)

    n_sub, _sub_labels, sub_stats, sub_centroids = cv2.connectedComponentsWithStats(eroded)
    if n_sub <= 2:
        return []

    results = []
    ref_area = sp * sp
    for j in range(1, n_sub):
        sub_area = int(sub_stats[j, cv2.CC_STAT_AREA])
        if sub_area < 15:
            continue
        sub_cx = float(sub_centroids[j][0]) + x0
        sub_cy = float(sub_centroids[j][1]) + y0
        sub_w = int(sub_stats[j, cv2.CC_STAT_WIDTH])
        sub_h = int(sub_stats[j, cv2.CC_STAT_HEIGHT])

        if sub_cy < roi_top or sub_cy > roi_bottom or sub_cx < header_x:
            continue

        staff_pos = _y_to_staff_position(sub_cy, staff_info)
        blob_density = sub_area / max(sub_w * sub_h, 1)
        is_filled = blob_density > 0.45
        confidence = min(1.0, 0.5 + sub_area / (ref_area * 2.0))

        results.append(NoteheadDetection(
            x=sub_cx, y=sub_cy, w=sub_w, h=sub_h,
            area=sub_area, staff_position=staff_pos,
            confidence=confidence, is_filled=is_filled,
        ))

    return results


def _y_to_staff_position(y: float, staff_info: StaffLineInfo) -> float:
    """Convert y-coordinate to staff position.

    Position 0 = bottom line, 2 = second line from bottom, etc.
    Fractional positions indicate positions between lines/spaces.
    """
    if not staff_info.y_positions:
        return 0.0
    bottom_y = staff_info.y_positions[-1]  # bottom line has highest y
    half_space = staff_info.spacing / 2.0
    # In image coordinates, y increases downward. Staff position increases upward.
    position = (bottom_y - y) / half_space
    return round(position * 2) / 2  # round to nearest half position


# ---------------------------------------------------------------------------
# Onset clustering
# ---------------------------------------------------------------------------

def cluster_onsets(
    noteheads: List[NoteheadDetection],
    staff_spacing: float,
    *,
    x_tolerance_ratio: float = 0.6,
) -> List[OnsetCluster]:
    """Group noteheads by x-position into onset clusters.

    Noteheads within x_tolerance pixels are considered simultaneous (chord).
    """
    if not noteheads:
        return []
    x_tol = staff_spacing * x_tolerance_ratio
    sorted_heads = sorted(noteheads, key=lambda n: n.x)

    clusters: List[OnsetCluster] = []
    current_group: List[NoteheadDetection] = [sorted_heads[0]]

    for nh in sorted_heads[1:]:
        if nh.x - current_group[-1].x < x_tol:
            current_group.append(nh)
        else:
            clusters.append(_make_onset_cluster(current_group))
            current_group = [nh]
    clusters.append(_make_onset_cluster(current_group))
    return clusters


def _make_onset_cluster(noteheads: List[NoteheadDetection]) -> OnsetCluster:
    x_center = sum(n.x for n in noteheads) / len(noteheads)
    avg_conf = sum(n.confidence for n in noteheads) / len(noteheads)
    return OnsetCluster(
        x_center=x_center,
        noteheads=list(noteheads),
        is_chord=len(noteheads) > 1,
        note_count=len(noteheads),
        confidence=avg_conf,
    )


# ---------------------------------------------------------------------------
# Measure construction
# ---------------------------------------------------------------------------

def build_measures(
    barlines: List[BarlineDetection],
    onset_clusters: List[OnsetCluster],
    image_width: int,
) -> List[MeasureSkeleton]:
    """Assign onset clusters to measures defined by barlines."""
    if not barlines:
        # No barlines detected — everything is one measure
        return [MeasureSkeleton(
            index=0,
            start_x=0.0,
            end_x=float(image_width),
            onsets=list(onset_clusters),
            note_count=sum(c.note_count for c in onset_clusters),
        )]

    boundaries = [0.0] + [b.x for b in barlines] + [float(image_width)]
    measures: List[MeasureSkeleton] = []
    for i in range(len(boundaries) - 1):
        start_x = boundaries[i]
        end_x = boundaries[i + 1]
        onsets = [c for c in onset_clusters if start_x <= c.x_center < end_x]
        measures.append(MeasureSkeleton(
            index=i,
            start_x=start_x,
            end_x=end_x,
            onsets=onsets,
            note_count=sum(c.note_count for c in onsets),
        ))
    return measures


# ---------------------------------------------------------------------------
# Clef estimation (simple heuristic based on notehead pitch range)
# ---------------------------------------------------------------------------

def estimate_clef_from_noteheads(
    noteheads: List[NoteheadDetection],
    staff_info: StaffLineInfo,
) -> Tuple[Optional[str], float]:
    """Guess the clef based on where noteheads cluster on the staff."""
    if not noteheads or staff_info is None:
        return None, 0.0
    positions = [n.staff_position for n in noteheads]
    mean_pos = sum(positions) / len(positions)
    # Staff has 5 lines: positions 0-8. Middle = 4.
    if mean_pos >= 2.0:
        return "clef-G2", min(1.0, 0.5 + abs(mean_pos - 4.0) * 0.1)
    else:
        return "clef-F4", min(1.0, 0.5 + abs(mean_pos - 4.0) * 0.1)


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_staff(
    image,
    *,
    clef_hint: Optional[str] = None,
) -> StaffSkeleton:
    """Run full CV analysis on a staff crop image.

    Args:
        image: file path (str/Path), or numpy array (uint8 grayscale or BGR).
        clef_hint: if known, use this clef for pitch estimation.

    Returns:
        StaffSkeleton with all detected elements.
    """
    # Load image
    if isinstance(image, (str, Path)):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    h, w = gray.shape[:2]
    skeleton = StaffSkeleton(image_width=w, image_height=h)

    # 1. Detect staff lines
    lines = detect_staff_lines(gray)
    staff_info = estimate_staff_info(lines)
    if staff_info is None:
        return skeleton
    skeleton.staff_lines = staff_info

    # 2. Binarize (for barline detection)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Detect barlines
    barlines = detect_barlines(binary, staff_info)
    skeleton.barlines = barlines
    skeleton.estimated_measure_count = max(1, len(barlines) + 1)
    skeleton.measure_count_confidence = min(1.0, 0.5 + len(barlines) * 0.15)

    # 4. Detect noteheads via segnet
    noteheads = detect_noteheads_segnet(gray, staff_info)
    skeleton.noteheads = noteheads
    skeleton.total_note_count = len(noteheads)
    skeleton.note_count_confidence = min(1.0, 0.4 + len(noteheads) * 0.03)

    # 5. Estimate clef
    if clef_hint:
        skeleton.estimated_clef = clef_hint
        skeleton.clef_confidence = 1.0
    else:
        clef, conf = estimate_clef_from_noteheads(noteheads, staff_info)
        skeleton.estimated_clef = clef
        skeleton.clef_confidence = conf

    # 6. Estimate pitch for each notehead
    if skeleton.estimated_clef:
        for nh in noteheads:
            nh.estimated_pitch = staff_position_to_pitch(
                nh.staff_position, skeleton.estimated_clef
            )

    # 7. Cluster noteheads into onsets
    onset_clusters = cluster_onsets(noteheads, staff_info.spacing)
    skeleton.onset_clusters = onset_clusters

    # 8. Build measure skeletons
    skeleton.measures = build_measures(barlines, onset_clusters, w)

    return skeleton


# ---------------------------------------------------------------------------
# Visualization (for debugging / tuning)
# ---------------------------------------------------------------------------

def draw_analysis(image, skeleton: StaffSkeleton) -> np.ndarray:
    """Draw detected elements on the image for visual debugging.

    Returns a BGR color image with annotations.
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Draw staff lines (blue)
    if skeleton.staff_lines:
        for y in skeleton.staff_lines.y_positions:
            cv2.line(img, (0, y), (img.shape[1], y), (255, 150, 0), 1)

    # Draw barlines (red)
    for bl in skeleton.barlines:
        x = int(bl.x)
        if skeleton.staff_lines:
            y1 = skeleton.staff_lines.top - 10
            y2 = skeleton.staff_lines.bottom + 10
        else:
            y1, y2 = 0, img.shape[0]
        cv2.line(img, (x, y1), (x, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{bl.confidence:.2f}", (x - 10, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Draw noteheads (green circles with pitch labels)
    for nh in skeleton.noteheads:
        cx, cy = int(nh.x), int(nh.y)
        radius = max(3, int(nh.w / 2))
        color = (0, 200, 0) if nh.confidence > 0.5 else (0, 100, 200)
        cv2.circle(img, (cx, cy), radius, color, 2)
        label = nh.estimated_pitch or f"p{nh.staff_position:.1f}"
        cv2.putText(img, label, (cx + radius + 2, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Draw onset clusters (vertical lines, orange for chords)
    for oc in skeleton.onset_clusters:
        x = int(oc.x_center)
        if oc.is_chord:
            cv2.line(img, (x, 0), (x, img.shape[0]), (0, 140, 255), 1)
            cv2.putText(img, f"chord({oc.note_count})", (x - 15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 140, 255), 1)

    # Draw measure indices
    for m in skeleton.measures:
        mx = int((m.start_x + m.end_x) / 2)
        cv2.putText(img, f"M{m.index}", (mx - 5, img.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 0, 180), 1)

    return img
