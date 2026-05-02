#!/usr/bin/env python3
"""Generate synthetic full-page OMR training assets from symbolic scores."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_STYLE_PRESETS: Dict[str, Dict[str, object]] = {
    "leipzig-default": {
        "font": "Leipzig",
        "pageWidth": 2200,
        "pageHeight": 3000,
        "scale": 40,
        "breaks": "auto",
        "svgBoundingBoxes": True,
        "svgViewBox": True,
    },
    "bravura-compact": {
        "font": "Bravura",
        "pageWidth": 2100,
        "pageHeight": 2970,
        "scale": 38,
        "breaks": "auto",
        "svgBoundingBoxes": True,
        "svgViewBox": True,
    },
    "gootville-wide": {
        "font": "Gootville",
        "pageWidth": 2350,
        "pageHeight": 3050,
        "scale": 42,
        "breaks": "auto",
        "svgBoundingBoxes": True,
        "svgViewBox": True,
    },
}

YOLO_CLASS_NAMES = ["staff"]
YOLO_CLASS_TO_ID = {name: idx for idx, name in enumerate(YOLO_CLASS_NAMES)}

SCORE_TYPE_TARGET_DISTRIBUTION = {
    "piano": 0.40,
    "orchestral": 0.25,
    "chamber": 0.20,
    "choral": 0.10,
    "solo_instrument_with_piano": 0.05,
}


@dataclass
class RenderJob:
    source_path: Path
    source_relpath: str
    style_id: str
    score_type: str


@dataclass
class RenderSourceTask:
    source_path: Path
    source_relpath: str
    score_type: str


_WARNING_QUOTED_ID_PATTERN = re.compile(r"'[^']+'")
_WARNING_INTEGER_PATTERN = re.compile(r"\b\d+\b")


class _NativeStderrCapture:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.lines: List[str] = []
        self._saved_fd: Optional[int] = None
        self._temp_file = None

    def __enter__(self) -> "_NativeStderrCapture":
        if not self.enabled:
            return self
        self._saved_fd = os.dup(2)
        self._temp_file = tempfile.TemporaryFile(mode="w+b")
        os.dup2(self._temp_file.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False
        assert self._saved_fd is not None
        assert self._temp_file is not None
        os.dup2(self._saved_fd, 2)
        os.close(self._saved_fd)
        self._temp_file.seek(0)
        captured = self._temp_file.read().decode("utf-8", errors="replace")
        self.lines = [line for line in captured.splitlines() if line.strip()]
        self._temp_file.close()
        self._saved_fd = None
        self._temp_file = None
        return False


def _normalize_verovio_warning(raw_line: str) -> Optional[str]:
    line = raw_line.strip()
    if not line:
        return None
    if line.startswith("[Warning]"):
        line = line[len("[Warning]") :].strip()
    line = _WARNING_QUOTED_ID_PATTERN.sub("'<id>'", line)
    line = _WARNING_INTEGER_PATTERN.sub("<n>", line)
    return line or None


def _record_verovio_warnings(raw_lines: Iterable[str], warning_counts: Counter[str]) -> None:
    for raw_line in raw_lines:
        normalized = _normalize_verovio_warning(raw_line)
        if normalized is None:
            continue
        warning_counts[normalized] += 1


def relpath(project_root: Path, target: Path) -> str:
    resolved_target = target.resolve()
    resolved_root = project_root.resolve()
    try:
        return str(resolved_target.relative_to(resolved_root)).replace("\\", "/")
    except ValueError:
        return str(resolved_target)


def sanitize_relpath_for_id(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", path)


def load_manifest_sources(project_root: Path, manifest_path: Path) -> List[Path]:
    sources: Dict[str, Path] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest_path}:{line_no}") from exc

            for key in ("mscx_path", "musicxml_path"):
                source_relpath = entry.get(key)
                if not source_relpath:
                    continue
                source_abs = (project_root / Path(str(source_relpath))).resolve()
                if not source_abs.exists():
                    continue
                stem_key = str(source_abs.with_suffix("")).lower()
                if stem_key not in sources:
                    sources[stem_key] = source_abs
                if source_abs.suffix.lower() in {".mxl", ".musicxml", ".xml"}:
                    sources[stem_key] = source_abs
    return sorted(sources.values())


def scan_default_sources(data_root: Path) -> List[Path]:
    lieder_scores = data_root / "Lieder-main" / "scores"
    if not lieder_scores.exists():
        raise FileNotFoundError(f"Default source directory not found: {lieder_scores}")

    selected: Dict[str, Path] = {}
    for ext in (".mxl", ".mscx"):
        for file_path in sorted(lieder_scores.rglob(f"*{ext}")):
            if file_path.name.startswith("."):
                continue
            stem_key = str(file_path.with_suffix("")).lower()
            if stem_key not in selected:
                selected[stem_key] = file_path
            if ext == ".mxl":
                selected[stem_key] = file_path
    return sorted(selected.values())


def infer_score_type(source_path: Path) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", " ".join(part.lower() for part in source_path.parts))

    def contains_any(keywords: Sequence[str]) -> bool:
        return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)

    has_piano = contains_any(("piano", "klavier", "pf"))
    has_voice = contains_any(("lied", "song", "voice", "soprano", "tenor", "alto", "baritone"))
    has_solo_instrument = any(
        re.search(rf"\b{re.escape(keyword)}\b", text)
        for keyword in (
            "violin",
            "viola",
            "cello",
            "flute",
            "oboe",
            "clarinet",
            "bassoon",
            "trumpet",
            "trombone",
            "horn",
            "sax",
            "guitar",
            "mandolin",
        )
    )

    if has_piano and has_voice:
        return "piano"
    if has_piano and has_solo_instrument:
        return "solo_instrument_with_piano"
    if any(keyword in text for keyword in ("symph", "orchestra", "concerto", "overture")):
        return "orchestral"
    if any(keyword in text for keyword in ("chor", "choir", "mass", "motet", "cantata", "hymn", "requiem")):
        return "choral"
    if any(keyword in text for keyword in ("quartet", "quintet", "trio", "duo", "sextet", "chamber", "sonata")):
        return "chamber"
    if has_piano:
        return "piano"
    return "piano"


def _compute_target_counts(total: int, distribution: Dict[str, float]) -> Dict[str, int]:
    raw = {name: total * ratio for name, ratio in distribution.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(distribution.keys(), key=lambda key: raw[key] - counts[key], reverse=True)
    for key in ranked[:remainder]:
        counts[key] += 1
    return counts


def select_sources(
    project_root: Path,
    data_root: Path,
    input_manifest: Optional[Path],
    max_scores: Optional[int],
    seed: int,
) -> Tuple[List[Path], Dict[str, int]]:
    if input_manifest is not None and input_manifest.exists():
        sources = load_manifest_sources(project_root, input_manifest)
    else:
        sources = scan_default_sources(data_root)

    typed: Dict[str, List[Path]] = {score_type: [] for score_type in SCORE_TYPE_TARGET_DISTRIBUTION}
    for source in sources:
        score_type = infer_score_type(source)
        typed.setdefault(score_type, []).append(source)

    if max_scores is None or len(sources) <= max_scores:
        return sources, {name: len(paths) for name, paths in typed.items()}

    rng = random.Random(seed)
    for paths in typed.values():
        rng.shuffle(paths)

    target_counts = _compute_target_counts(max_scores, SCORE_TYPE_TARGET_DISTRIBUTION)
    selected: List[Path] = []
    selected_counts: Counter[str] = Counter()
    leftovers: List[Path] = []

    for score_type, target in target_counts.items():
        bucket = typed.get(score_type, [])
        take = min(target, len(bucket))
        selected.extend(bucket[:take])
        selected_counts[score_type] += take
        leftovers.extend(bucket[take:])

    if len(selected) < max_scores and leftovers:
        rng.shuffle(leftovers)
        needed = max_scores - len(selected)
        extra = leftovers[:needed]
        selected.extend(extra)
        for item in extra:
            selected_counts[infer_score_type(item)] += 1

    selected = sorted(selected[:max_scores])
    return selected, {score_type: int(selected_counts.get(score_type, 0)) for score_type in SCORE_TYPE_TARGET_DISTRIBUTION}


def build_jobs(
    project_root: Path,
    sources: Sequence[Path],
    style_ids: Sequence[str],
) -> List[RenderJob]:
    jobs: List[RenderJob] = []
    for source in sources:
        source_relpath = relpath(project_root, source)
        score_type = infer_score_type(source)
        for style_id in style_ids:
            jobs.append(
                RenderJob(
                    source_path=source,
                    source_relpath=source_relpath,
                    style_id=style_id,
                    score_type=score_type,
                )
            )
    return jobs


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def parse_svg_dimensions(svg_root: ET.Element) -> Tuple[float, float]:
    view_box = svg_root.attrib.get("viewBox")
    if view_box:
        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", view_box)]
        if len(numbers) == 4:
            return numbers[2], numbers[3]

    width = parse_float(svg_root.attrib.get("width"))
    height = parse_float(svg_root.attrib.get("height"))
    if width and height:
        return width, height

    raise ValueError("Could not resolve SVG dimensions from width/height/viewBox.")


def normalize_bbox_to_yolo(
    bbox: Tuple[float, float, float, float], page_width: float, page_height: float
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / page_width
    y_center = (y + h / 2.0) / page_height
    w_norm = w / page_width
    h_norm = h / page_height
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, w_norm)),
        max(0.0, min(1.0, h_norm)),
    )


def estimate_staff_boxes(page_width: float, page_height: float, staff_count: int) -> List[Tuple[float, float, float, float]]:
    count = max(1, staff_count)
    top_margin = page_height * 0.08
    usable_height = page_height * 0.84
    stride = usable_height / count
    box_height = min(90.0, stride * 0.55)
    boxes = []
    for idx in range(count):
        y = top_margin + idx * stride + (stride - box_height) / 2.0
        boxes.append((page_width * 0.05, y, page_width * 0.90, box_height))
    return boxes


def _merge_staff_boxes(staff_boxes: Sequence[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    if not staff_boxes:
        return []

    deduped: List[Tuple[float, float, float, float]] = []
    seen = set()
    for box in staff_boxes:
        key = tuple(round(float(value), 2) for value in box)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)

    rows: List[Dict[str, object]] = []
    for box in sorted(deduped, key=lambda item: (item[1] + (item[3] / 2.0), item[0])):
        y_center = box[1] + (box[3] / 2.0)
        placed = False
        for row in rows:
            row_center = float(row["y_center"])
            row_height = float(row["height"])
            tolerance = max(6.0, min(row_height, box[3]) * 0.35)
            if abs(y_center - row_center) <= tolerance:
                row_boxes = row["boxes"]
                assert isinstance(row_boxes, list)
                row_boxes.append(box)
                count = len(row_boxes)
                row["y_center"] = ((row_center * (count - 1)) + y_center) / count
                row["height"] = max(row_height, box[3])
                placed = True
                break
        if not placed:
            rows.append({"y_center": y_center, "height": box[3], "boxes": [box]})

    merged: List[Tuple[float, float, float, float]] = []
    for row in rows:
        row_boxes = row["boxes"]
        assert isinstance(row_boxes, list)
        x_min = min(item[0] for item in row_boxes)
        y_min = min(item[1] for item in row_boxes)
        x_max = max(item[0] + item[2] for item in row_boxes)
        y_max = max(item[1] + item[3] for item in row_boxes)
        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)
        if width > 0 and height > 0:
            merged.append((x_min, y_min, width, height))

    return sorted(merged, key=lambda item: (item[1], item[0]))


def _sanitize_staff_boxes(
    staff_boxes: Sequence[Tuple[float, float, float, float]],
    *,
    page_width: float,
    page_height: float,
    min_width_ratio: float = 0.20,
) -> List[Tuple[float, float, float, float]]:
    if not staff_boxes:
        return []

    sane: List[Tuple[float, float, float, float]] = []
    for x, y, w, h in staff_boxes:
        w_norm = w / max(page_width, 1e-6)
        h_norm = h / max(page_height, 1e-6)
        # Staff boxes should be wide and relatively short; reject degenerate
        # full-system or microscopic fragments that pollute YOLO labels.
        if w_norm < max(0.05, float(min_width_ratio)):
            continue
        if h_norm < 0.012 or h_norm > 0.20:
            continue
        sane.append((x, y, w, h))

    if not sane:
        return []

    heights = sorted(box[3] for box in sane)
    median_h = heights[len(heights) // 2]
    if median_h <= 0:
        return sane

    min_h = median_h * 0.35
    max_h = median_h * 2.5
    stable = [box for box in sane if min_h <= box[3] <= max_h]
    return stable if stable else sane


def _expand_staff_boxes_vertical(
    staff_boxes: Sequence[Tuple[float, float, float, float]],
    *,
    page_height: float,
    top_ratio: float = 0.22,
    bottom_ratio: float = 0.26,
    min_pad_px: float = 8.0,
) -> List[Tuple[float, float, float, float]]:
    expanded: List[Tuple[float, float, float, float]] = []
    for x, y, w, h in staff_boxes:
        top_pad = max(float(min_pad_px), h * float(top_ratio))
        bottom_pad = max(float(min_pad_px), h * float(bottom_ratio))
        y1 = max(0.0, y - top_pad)
        y2 = min(float(page_height), y + h + bottom_pad)
        new_h = max(0.0, y2 - y1)
        if new_h <= 0:
            continue
        expanded.append((x, y1, w, new_h))
    return expanded


def _extract_bbox_from_attribs(attrs: Dict[str, str]) -> Optional[Tuple[float, float, float, float]]:
    bbox_attrs = (attrs.get("data-bbox"), attrs.get("bbox"))
    for candidate in bbox_attrs:
        if not candidate:
            continue
        values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", candidate)]
        if len(values) >= 4:
            return values[0], values[1], values[2], values[3]

    x = parse_float(attrs.get("x"))
    y = parse_float(attrs.get("y"))
    width = parse_float(attrs.get("width"))
    height = parse_float(attrs.get("height"))
    if None not in {x, y, width, height} and (width or 0.0) > 0 and (height or 0.0) > 0:
        return x or 0.0, y or 0.0, width or 0.0, height or 0.0
    return None


def _extract_staff_lines_bbox(element: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    horizontal_segments: List[Tuple[float, float, float, float]] = []
    for child in element:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag != "path":
            continue
        d_value = child.attrib.get("d", "")
        values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d_value)]
        if len(values) < 4:
            continue
        x1, y1, x2, y2 = values[:4]
        if abs(y1 - y2) > 2.0:
            continue
        if abs(x2 - x1) < 24.0:
            continue
        horizontal_segments.append((x1, y1, x2, y2))

    if len(horizontal_segments) < 5:
        return None

    x_min = min(min(seg[0], seg[2]) for seg in horizontal_segments)
    x_max = max(max(seg[0], seg[2]) for seg in horizontal_segments)
    y_values = sorted(seg[1] for seg in horizontal_segments)
    y_min = min(y_values)
    y_max = max(y_values)
    if x_max <= x_min or y_max <= y_min:
        return None

    y_diffs = [b - a for a, b in zip(y_values[:-1], y_values[1:]) if (b - a) > 0.5]
    spacing = y_diffs[len(y_diffs) // 2] if y_diffs else 0.0
    y_pad = max(6.0, min(36.0, spacing * 0.8 if spacing > 0 else 12.0))
    padded_y_min = max(0.0, y_min - y_pad)
    padded_y_max = y_max + y_pad
    return (x_min, padded_y_min, max(0.0, x_max - x_min), max(0.0, padded_y_max - padded_y_min))


def _extract_staff_bbox_with_source(
    element: ET.Element,
) -> Optional[Tuple[Tuple[float, float, float, float], ET.Element]]:
    for child in element:
        class_text = child.attrib.get("class", "").lower()
        if "staff bounding-box" not in class_text:
            continue
        direct_bbox = _extract_bbox_from_attribs(child.attrib)
        if direct_bbox is not None:
            return direct_bbox, child
        for grandchild in child.iter():
            if grandchild is child:
                continue
            child_bbox = _extract_bbox_from_attribs(grandchild.attrib)
            if child_bbox is not None:
                return child_bbox, grandchild

    line_bbox = _extract_staff_lines_bbox(element)
    if line_bbox is not None:
        return line_bbox, element
    return _extract_bbox_with_source(element)


def _extract_bbox_with_source(
    element: ET.Element,
) -> Optional[Tuple[Tuple[float, float, float, float], ET.Element]]:
    direct_bbox = _extract_bbox_from_attribs(element.attrib)
    if direct_bbox is not None:
        return direct_bbox, element

    # Verovio commonly stores layout boxes inside child <rect> nodes of "* bounding-box" groups.
    for child in element.iter():
        if child is element:
            continue
        child_bbox = _extract_bbox_from_attribs(child.attrib)
        if child_bbox is not None:
            return child_bbox, child

    return None


def _extract_bbox(element: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    extracted = _extract_bbox_with_source(element)
    if extracted is None:
        return None
    bbox, _ = extracted
    return bbox


def _identity_affine() -> Tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _compose_affine(
    lhs: Tuple[float, float, float, float, float, float],
    rhs: Tuple[float, float, float, float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    # SVG affine matrices:
    # [a c e]
    # [b d f]
    # [0 0 1]
    a1, b1, c1, d1, e1, f1 = lhs
    a2, b2, c2, d2, e2, f2 = rhs
    return (
        (a1 * a2) + (c1 * b2),
        (b1 * a2) + (d1 * b2),
        (a1 * c2) + (c1 * d2),
        (b1 * c2) + (d1 * d2),
        (a1 * e2) + (c1 * f2) + e1,
        (b1 * e2) + (d1 * f2) + f1,
    )


def _parse_transform(transform_text: str) -> Tuple[float, float, float, float, float, float]:
    matrix = _identity_affine()
    for fn_name, raw_args in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", transform_text):
        values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw_args)]
        name = fn_name.strip().lower()
        if name == "matrix" and len(values) >= 6:
            op = (values[0], values[1], values[2], values[3], values[4], values[5])
        elif name == "translate" and values:
            tx = values[0]
            ty = values[1] if len(values) > 1 else 0.0
            op = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale" and values:
            sx = values[0]
            sy = values[1] if len(values) > 1 else sx
            op = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        else:
            # Unsupported transform operations are ignored for bbox extraction.
            continue
        matrix = _compose_affine(matrix, op)
    return matrix


def _apply_affine_to_point(
    matrix: Tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> Tuple[float, float]:
    a, b, c, d, e, f = matrix
    return ((a * x) + (c * y) + e, (b * x) + (d * y) + f)


def _apply_affine_to_bbox(
    matrix: Tuple[float, float, float, float, float, float],
    bbox: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    corners = (
        _apply_affine_to_point(matrix, x, y),
        _apply_affine_to_point(matrix, x + w, y),
        _apply_affine_to_point(matrix, x, y + h),
        _apply_affine_to_point(matrix, x + w, y + h),
    )
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return (x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min))


def _build_cumulative_transforms(
    root: ET.Element,
) -> Dict[int, Tuple[float, float, float, float, float, float]]:
    transforms: Dict[int, Tuple[float, float, float, float, float, float]] = {}
    identity = _identity_affine()

    def visit(element: ET.Element, parent_matrix: Tuple[float, float, float, float, float, float]) -> None:
        own_matrix = parent_matrix
        transform_text = element.attrib.get("transform")
        if transform_text:
            own_matrix = _compose_affine(parent_matrix, _parse_transform(transform_text))
        transforms[id(element)] = own_matrix
        for child in element:
            visit(child, own_matrix)

    visit(root, identity)
    return transforms


def _find_definition_scale_svg(svg_root: ET.Element) -> Optional[ET.Element]:
    for child in svg_root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag != "svg":
            continue
        class_text = child.attrib.get("class", "").lower()
        if "definition-scale" in class_text and child.attrib.get("viewBox"):
            return child
    return None


def _classify_element(class_text: str) -> List[int]:
    classes: List[int] = []
    lowered = class_text.lower().strip()
    if not lowered:
        return classes

    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]

    if "staff" in tokens:
        classes.append(YOLO_CLASS_TO_ID["staff"])
    return classes


def _deduplicate_objects(
    objects: Sequence[Tuple[int, Tuple[float, float, float, float]]],
    *,
    precision: int = 2,
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    deduped: List[Tuple[int, Tuple[float, float, float, float]]] = []
    seen = set()
    for class_id, bbox in objects:
        key = (
            int(class_id),
            round(float(bbox[0]), precision),
            round(float(bbox[1]), precision),
            round(float(bbox[2]), precision),
            round(float(bbox[3]), precision),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((int(class_id), (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
    return deduped


def _filter_page_objects(
    objects: Sequence[Tuple[int, Tuple[float, float, float, float]]],
    *,
    page_width: float,
    page_height: float,
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    filtered: List[Tuple[int, Tuple[float, float, float, float]]] = []
    system_bracket_id = YOLO_CLASS_TO_ID.get("system_bracket")
    barline_id = YOLO_CLASS_TO_ID.get("barline_system")
    for class_id, (x, y, w, h) in objects:
        if w <= 0 or h <= 0:
            continue
        w_norm = w / max(page_width, 1e-6)
        h_norm = h / max(page_height, 1e-6)
        x_center_norm = (x + (w / 2.0)) / max(page_width, 1e-6)

        if system_bracket_id is not None and class_id == system_bracket_id:
            if h < (2.0 * w):
                continue
            if h_norm < 0.05:
                continue
            if w_norm > 0.03:
                continue
            if x_center_norm > 0.18:
                continue
        elif barline_id is not None and class_id == barline_id:
            if h < (2.0 * w):
                continue
            if h_norm < 0.03:
                continue
            if h_norm > 0.60:
                continue
            if w_norm > 0.01:
                continue

        filtered.append((class_id, (x, y, w, h)))

    return filtered


def _fallback_page_objects(
    page_width: float,
    page_height: float,
    staff_boxes: Sequence[Tuple[float, float, float, float]],
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    objects: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for bbox in staff_boxes:
        objects.append((YOLO_CLASS_TO_ID["staff"], bbox))
    return objects


def _normalize_detected_bboxes_to_page(
    detected: Sequence[Tuple[int, Tuple[float, float, float, float]]],
    *,
    page_width: float,
    page_height: float,
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    if not detected:
        return []

    min_x = min(bbox[0] for _, bbox in detected)
    min_y = min(bbox[1] for _, bbox in detected)
    max_x = max(bbox[0] + bbox[2] for _, bbox in detected)
    max_y = max(bbox[1] + bbox[3] for _, bbox in detected)
    span_x = max_x - min_x
    span_y = max_y - min_y

    if span_x <= 1e-6 or span_y <= 1e-6:
        return list(detected)

    mostly_in_page_space = (
        min_x >= (-0.25 * page_width)
        and min_y >= (-0.25 * page_height)
        and max_x <= (1.25 * page_width)
        and max_y <= (1.25 * page_height)
    )
    if mostly_in_page_space:
        return list(detected)

    # Some Verovio outputs encode bboxes in a larger page-unit space with origin already near (0,0).
    # In that case, preserve origin and only scale down to page dimensions.
    near_origin = (
        min_x >= (-0.05 * max(max_x, 1.0))
        and min_y >= (-0.05 * max(max_y, 1.0))
    )
    if near_origin and max_x > 0 and max_y > 0:
        scale_x = page_width / max_x
        scale_y = page_height / max_y
        normalized_origin: List[Tuple[int, Tuple[float, float, float, float]]] = []
        for class_id, (x, y, w, h) in detected:
            nx = max(0.0, min(page_width, x * scale_x))
            ny = max(0.0, min(page_height, y * scale_y))
            nw = max(0.0, min(page_width - nx, w * scale_x))
            nh = max(0.0, min(page_height - ny, h * scale_y))
            if nw <= 0 or nh <= 0:
                continue
            normalized_origin.append((class_id, (nx, ny, nw, nh)))
        return normalized_origin

    scale_x = page_width / span_x
    scale_y = page_height / span_y
    normalized: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for class_id, (x, y, w, h) in detected:
        nx = max(0.0, min(page_width, (x - min_x) * scale_x))
        ny = max(0.0, min(page_height, (y - min_y) * scale_y))
        nw = max(0.0, min(page_width - nx, w * scale_x))
        nh = max(0.0, min(page_height - ny, h * scale_y))
        if nw <= 0 or nh <= 0:
            continue
        normalized.append((class_id, (nx, ny, nw, nh)))
    return normalized


def extract_page_objects(
    svg_text: str,
    *,
    allow_fallback_objects: bool = False,
    min_staff_box_width_ratio: float = 0.20,
) -> Tuple[float, float, List[Tuple[int, Tuple[float, float, float, float]]]]:
    root = ET.fromstring(svg_text)
    page_width, page_height = parse_svg_dimensions(root)
    detected: List[Tuple[int, Tuple[float, float, float, float]]] = []
    staff_like_count = 0
    used_definition_scale = False

    definition_scale_svg = _find_definition_scale_svg(root)
    if definition_scale_svg is not None:
        view_box = definition_scale_svg.attrib.get("viewBox", "")
        vb_numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", view_box)]
        if len(vb_numbers) == 4 and vb_numbers[2] > 0 and vb_numbers[3] > 0:
            vb_x, vb_y, vb_width, vb_height = vb_numbers
            scale_x = page_width / vb_width
            scale_y = page_height / vb_height
            transforms = _build_cumulative_transforms(definition_scale_svg)

            for element in definition_scale_svg.iter():
                class_value = (
                    element.attrib.get("class", "")
                    + " "
                    + element.attrib.get("data-class", "")
                )
                class_ids = _classify_element(class_value)
                if not class_ids:
                    continue
                if YOLO_CLASS_TO_ID["staff"] in class_ids:
                    extracted = _extract_staff_bbox_with_source(element)
                else:
                    extracted = _extract_bbox_with_source(element)
                if extracted is None:
                    continue
                raw_bbox, source_element = extracted
                if raw_bbox[2] <= 0 or raw_bbox[3] <= 0:
                    continue

                element_matrix = transforms.get(id(source_element), _identity_affine())
                inner_bbox = _apply_affine_to_bbox(element_matrix, raw_bbox)

                nx = (inner_bbox[0] - vb_x) * scale_x
                ny = (inner_bbox[1] - vb_y) * scale_y
                nw = inner_bbox[2] * scale_x
                nh = inner_bbox[3] * scale_y

                x1 = max(0.0, min(page_width, nx))
                y1 = max(0.0, min(page_height, ny))
                x2 = max(0.0, min(page_width, nx + nw))
                y2 = max(0.0, min(page_height, ny + nh))
                clipped_bbox = (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))
                if clipped_bbox[2] <= 0 or clipped_bbox[3] <= 0:
                    continue

                if YOLO_CLASS_TO_ID["staff"] in class_ids:
                    staff_like_count += 1
                for class_id in class_ids:
                    detected.append((class_id, clipped_bbox))
            used_definition_scale = True

    if not used_definition_scale:
        for element in root.iter():
            class_value = (
                element.attrib.get("class", "")
                + " "
                + element.attrib.get("data-class", "")
            )
            class_ids = _classify_element(class_value)
            if not class_ids:
                continue
            if YOLO_CLASS_TO_ID["staff"] in class_ids:
                extracted = _extract_staff_bbox_with_source(element)
                bbox = extracted[0] if extracted is not None else None
            else:
                bbox = _extract_bbox(element)
            if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            if YOLO_CLASS_TO_ID["staff"] in class_ids:
                staff_like_count += 1
            for class_id in class_ids:
                detected.append((class_id, bbox))

        detected = _deduplicate_objects(detected)
        detected = _normalize_detected_bboxes_to_page(
            detected,
            page_width=page_width,
            page_height=page_height,
        )

    detected = _deduplicate_objects(detected)
    detected = _filter_page_objects(
        detected,
        page_width=page_width,
        page_height=page_height,
    )
    detected = _deduplicate_objects(detected)

    staff_boxes = [bbox for class_id, bbox in detected if class_id == YOLO_CLASS_TO_ID["staff"]]
    if staff_boxes:
        merged_staff_boxes = _merge_staff_boxes(staff_boxes)
        merged_staff_boxes = _sanitize_staff_boxes(
            merged_staff_boxes,
            page_width=page_width,
            page_height=page_height,
            min_width_ratio=min_staff_box_width_ratio,
        )
        if merged_staff_boxes:
            merged_staff_boxes = _expand_staff_boxes_vertical(
                merged_staff_boxes,
                page_height=page_height,
            )
            non_staff_objects = [item for item in detected if item[0] != YOLO_CLASS_TO_ID["staff"]]
            detected = non_staff_objects + [(YOLO_CLASS_TO_ID["staff"], box) for box in merged_staff_boxes]
            staff_boxes = merged_staff_boxes
        else:
            # Drop implausible staff detections; optional fallback synthesis
            # is handled below when explicitly enabled.
            detected = [item for item in detected if item[0] != YOLO_CLASS_TO_ID["staff"]]
            staff_boxes = []
    if not staff_boxes and allow_fallback_objects:
        staff_boxes = estimate_staff_boxes(page_width, page_height, max(1, staff_like_count))
    if allow_fallback_objects:
        if detected:
            existing = {class_id for class_id, _ in detected}
            fallback = _fallback_page_objects(page_width, page_height, staff_boxes)
            for class_id, bbox in fallback:
                if class_id not in existing:
                    detected.append((class_id, bbox))
        else:
            detected = _fallback_page_objects(page_width, page_height, staff_boxes)

    detected = _deduplicate_objects(detected)
    return page_width, page_height, detected


def render_svg_pages(
    source_path: Path,
    style_options: Dict[str, object],
    max_pages_per_score: Optional[int],
    warning_counts: Optional[Counter[str]] = None,
    show_verovio_warnings: bool = False,
) -> Iterator[Tuple[int, str]]:
    try:
        import verovio
    except ImportError as exc:  # pragma: no cover - dependency based
        raise RuntimeError("verovio is required for render mode. Install with: pip install verovio") from exc

    capture = _NativeStderrCapture(enabled=not show_verovio_warnings)
    with capture:
        toolkit = verovio.toolkit()
        toolkit.loadFile(str(source_path))
        toolkit.setOptions(style_options)
        toolkit.redoLayout()
        page_count = int(toolkit.getPageCount())
        if max_pages_per_score is not None:
            page_count = min(page_count, max_pages_per_score)
        for page_no in range(1, page_count + 1):
            yield page_no, toolkit.renderToSVG(page_no)
    if warning_counts is not None:
        _record_verovio_warnings(capture.lines, warning_counts)


def maybe_write_png(svg_text: str, output_png: Path, dpi: int = 300) -> bool:
    from src.data.multi_dpi import rasterize_svg_bytes
    rasterize_svg_bytes(svg_text.encode("utf-8"), output_png, dpi)
    return True


def write_yolo_labels(
    output_label_path: Path,
    objects: Iterable[Tuple[int, Tuple[float, float, float, float]]],
    page_width: float,
    page_height: float,
) -> None:
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, bbox in objects:
        x_center, y_center, w_norm, h_norm = normalize_bbox_to_yolo(bbox, page_width, page_height)
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    output_label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _rasterize_svg_to_array(svg_text: str):
    import numpy as np
    with _rasterize_svg_to_image(svg_text) as image:
        return np.asarray(image, dtype=np.float32)


def _rasterize_svg_to_image(svg_text: str):
    from PIL import Image
    from src.data.multi_dpi import rasterize_svg_bytes_to_png_bytes

    png_bytes = rasterize_svg_bytes_to_png_bytes(svg_text.encode("utf-8"), dpi=96)
    with Image.open(BytesIO(png_bytes)) as image_obj:
        if "A" in image_obj.getbands():
            rgba = image_obj.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(white_bg, rgba).convert("L")
            return composited
        return image_obj.convert("L")


def _split_staff_token_sequences(token_sequence: Sequence[str]) -> List[List[str]]:
    staff_sequences: List[List[str]] = []
    idx = 0
    while idx < len(token_sequence):
        if token_sequence[idx] != "<staff_start>":
            idx += 1
            continue
        end_idx = idx + 1
        while end_idx < len(token_sequence) and token_sequence[end_idx] != "<staff_end>":
            end_idx += 1
        if end_idx >= len(token_sequence):
            break
        staff_sequences.append(["<bos>", *token_sequence[idx : end_idx + 1], "<eos>"])
        idx = end_idx + 1
    return staff_sequences


def _slice_staff_sequence_segment(
    staff_sequence: Sequence[str],
    *,
    segment_index: int,
    segment_count: int,
    segment_weights: Optional[Sequence[int]] = None,
) -> List[str]:
    tokens = list(staff_sequence)
    if segment_count <= 1:
        return tokens

    try:
        staff_start_idx = tokens.index("<staff_start>")
        staff_end_idx = len(tokens) - 1 - tokens[::-1].index("<staff_end>")
    except ValueError:
        return tokens
    if staff_end_idx <= staff_start_idx:
        return tokens

    first_measure_idx = None
    for idx in range(staff_start_idx + 1, staff_end_idx):
        if tokens[idx] == "<measure_start>":
            first_measure_idx = idx
            break
    if first_measure_idx is None:
        return tokens

    prefix = tokens[staff_start_idx + 1 : first_measure_idx]
    measures: List[List[str]] = []
    idx = first_measure_idx
    while idx < staff_end_idx:
        if tokens[idx] != "<measure_start>":
            idx += 1
            continue
        end_idx = idx + 1
        while end_idx < staff_end_idx and tokens[end_idx] != "<measure_end>":
            end_idx += 1
        if end_idx >= staff_end_idx:
            break
        measures.append(tokens[idx : end_idx + 1])
        idx = end_idx + 1

    if not measures:
        return tokens

    total_measures = len(measures)
    index = max(1, min(segment_count, segment_index))
    normalized_weights: Optional[List[int]] = None
    if segment_weights is not None and len(segment_weights) == segment_count:
        normalized = [max(0, int(weight)) for weight in segment_weights]
        if sum(normalized) > 0:
            normalized_weights = normalized

    if normalized_weights is not None:
        weight_cumsum = [0]
        for weight in normalized_weights:
            weight_cumsum.append(weight_cumsum[-1] + weight)
        total_weight = max(1, weight_cumsum[-1])
        start = int((weight_cumsum[index - 1] * total_measures) / total_weight)
        end = int((weight_cumsum[index] * total_measures) / total_weight)
    else:
        start = int(((index - 1) * total_measures) / segment_count)
        end = int((index * total_measures) / segment_count)
    if end <= start:
        if start >= total_measures:
            start = max(0, total_measures - 1)
            end = total_measures
        else:
            end = min(total_measures, start + 1)
    selected_measures = measures[start:end]

    sliced = ["<bos>", "<staff_start>"]
    sliced.extend(prefix)
    for measure_tokens in selected_measures:
        sliced.extend(measure_tokens)
    sliced.append("<staff_end>")
    sliced.append("<eos>")
    return sliced


def _slice_staff_sequence_for_page(
    staff_sequence: Sequence[str],
    *,
    page_number: int,
    page_count: int,
) -> List[str]:
    return _slice_staff_sequence_segment(
        staff_sequence,
        segment_index=page_number,
        segment_count=page_count,
    )


def _slice_staff_sequence_for_fragment(
    staff_sequence: Sequence[str],
    *,
    fragment_index: int,
    fragment_count: int,
    fragment_weights: Optional[Sequence[int]] = None,
) -> List[str]:
    return _slice_staff_sequence_segment(
        staff_sequence,
        segment_index=fragment_index,
        segment_count=fragment_count,
        segment_weights=fragment_weights,
    )


def _sorted_staff_boxes(
    objects: Sequence[Tuple[int, Tuple[float, float, float, float]]]
) -> List[Tuple[float, float, float, float]]:
    boxes = [
        bbox
        for class_id, bbox in objects
        if class_id == YOLO_CLASS_TO_ID["staff"] and bbox[2] > 0 and bbox[3] > 0
    ]
    return sorted(boxes, key=lambda item: (item[1], item[0]))


def _estimate_staff_measure_weights(
    staff_boxes: Sequence[Tuple[float, float, float, float]],
    barline_boxes: Sequence[Tuple[float, float, float, float]],
) -> List[int]:
    if not staff_boxes:
        return []
    if not barline_boxes:
        return [1 for _ in staff_boxes]

    weights: List[int] = []
    for _, staff_y, _, staff_h in staff_boxes:
        staff_top = staff_y
        staff_bottom = staff_y + staff_h
        overlap_count = 0
        for _, bar_y, _, bar_h in barline_boxes:
            bar_top = bar_y
            bar_bottom = bar_y + bar_h
            overlap = min(staff_bottom, bar_bottom) - max(staff_top, bar_top)
            min_height = min(staff_h, bar_h)
            if min_height <= 0:
                continue
            if overlap >= (0.45 * min_height):
                overlap_count += 1
        weights.append(max(1, overlap_count))
    return weights


def _bbox_ink_ratio_from_array(
    image_array,
    bbox: Tuple[float, float, float, float],
    *,
    gray_threshold: int = 245,
    sample_step: int = 6,
) -> float:
    import numpy as np

    if image_array is None:
        return 0.0
    if not isinstance(image_array, np.ndarray) or image_array.size == 0:
        return 0.0
    if image_array.ndim != 2:
        return 0.0

    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image_array.shape[1], int(round(x + w)))
    y2 = min(image_array.shape[0], int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    step = max(1, int(sample_step))
    sampled = image_array[y1:y2:step, x1:x2:step]
    if sampled.size == 0:
        return 0.0
    return float((sampled < gray_threshold).mean())


def _page_ink_ratio_from_array(
    image_array,
    *,
    gray_threshold: int = 245,
    sample_step: int = 12,
) -> float:
    import numpy as np

    if image_array is None:
        return 0.0
    if not isinstance(image_array, np.ndarray) or image_array.size == 0:
        return 0.0
    if image_array.ndim != 2:
        return 0.0

    step = max(1, int(sample_step))
    sampled = image_array[::step, ::step]
    if sampled.size == 0:
        return 0.0
    return float((sampled < gray_threshold).mean())


def _ensure_metadata_regions(
    objects: Sequence[Tuple[int, Tuple[float, float, float, float]]],
    *,
    page_width: float,
    page_height: float,
    page_ink_array,
    min_title_ink_ratio: float = 0.004,
    min_page_number_ink_ratio: float = 0.003,
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    # Metadata classes are intentionally disabled for staff-only YOLO training.
    return list(objects)


def _assess_yolo_label_quality(
    objects: Sequence[Tuple[int, Tuple[float, float, float, float]]],
    *,
    page_width: float,
    page_height: float,
    page_ink_array=None,
    min_page_ink_ratio: float = 0.005,
    min_staff_ink_ratio: float = 0.01,
    expected_min_staff_count: int = 1,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    staff_boxes = _sorted_staff_boxes(objects)
    if not staff_boxes:
        reasons.append("missing_staff")
        return False, reasons
    if len(staff_boxes) < max(1, int(expected_min_staff_count)):
        reasons.append("staff_count_too_low")
        return False, sorted(set(reasons))

    if page_ink_array is not None:
        page_ink = _page_ink_ratio_from_array(page_ink_array)
        if page_ink <= float(min_page_ink_ratio):
            reasons.append("low_page_ink")
            return False, sorted(set(reasons))

        for bbox in staff_boxes:
            staff_ink = _bbox_ink_ratio_from_array(page_ink_array, bbox, sample_step=4)
            if staff_ink < float(min_staff_ink_ratio):
                reasons.append("low_staff_ink")
                break

    for _, _, w, h in staff_boxes:
        w_norm = w / max(page_width, 1e-6)
        h_norm = h / max(page_height, 1e-6)
        if w_norm < 0.18:
            reasons.append("staff_too_narrow")
            break
        if h_norm < 0.010 or h_norm > 0.30:
            reasons.append("staff_height_out_of_range")
            break

    return (len(reasons) == 0), sorted(set(reasons))


@dataclass
class _SvgSystemInfo:
    """Layout information for one system extracted from Verovio SVG."""
    measure_count: int
    staves_per_system: int
    y_top: float
    y_bottom: float
    # x_left is the left edge of the system bounding-box rect, which Verovio
    # places at the bracket position (rect width is just the bracket itself,
    # ~20-30 px). Use as a candidate for the bbox left edge to capture the
    # bracket as a distinctive YOLO-detection anchor. None when no bracket
    # info is available (e.g., older callers, or pages without a system rect).
    x_left: Optional[float] = None


def _extract_system_layout_from_svg(
    svg_text: str,
) -> Optional[List[_SvgSystemInfo]]:
    """Parse Verovio SVG to extract layout info per system.

    For each ``<g class="system">`` in the SVG, extracts:
    - measure count (direct ``<g class="measure">`` children)
    - staves per system (``<g class="staff">`` children of the first measure)
    - vertical bounding box (y_top, y_bottom) from the system bounding-box rect

    Returns a list ordered top-to-bottom, or *None* on parse failure.
    """
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return None

    raw: List[Tuple[float, float, int, int, float]] = []  # (y_top, y_bottom, measures, staves, x_left)
    for elem in root.iter():
        if elem.attrib.get("class", "").strip() != "system":
            continue

        # Count direct measure children
        measures = [
            c for c in elem
            if c.attrib.get("class", "").strip() == "measure"
        ]
        if not measures:
            continue

        # Count staff elements in the first measure (= staves per system)
        staves_in_first = sum(
            1 for c in measures[0]
            if c.attrib.get("class", "").strip() == "staff"
        )

        # Extract bounding box y, height, and x (bracket left edge).
        y_top = 0.0
        y_bottom = 0.0
        x_left = 0.0
        for child in elem.iter():
            if child.attrib.get("class", "").strip() == "system bounding-box":
                for rect in child.iter():
                    tag = rect.tag.split("}")[-1] if "}" in rect.tag else rect.tag
                    if tag == "rect":
                        x_val = parse_float(rect.attrib.get("x"))
                        y_val = parse_float(rect.attrib.get("y"))
                        h_val = parse_float(rect.attrib.get("height"))
                        if y_val is not None:
                            y_top = y_val
                            y_bottom = y_val + (h_val or 0.0)
                        if x_val is not None:
                            x_left = x_val
                        break
                break

        raw.append((y_top, y_bottom, len(measures), staves_in_first, x_left))

    if not raw:
        return None

    raw.sort(key=lambda item: item[0])
    return [
        _SvgSystemInfo(
            measure_count=m,
            staves_per_system=s,
            y_top=yt,
            y_bottom=yb,
            x_left=xl,
        )
        for yt, yb, m, s, xl in raw
    ]


def _assign_staff_boxes_to_systems(
    staff_boxes: Sequence[Tuple[float, float, float, float]],
    systems: Sequence[_SvgSystemInfo],
) -> Dict[int, Tuple[int, int]]:
    """Map each staff box index to ``(system_idx, position_within_system)``.

    Uses the per-system stave counts from the SVG hierarchy to do a
    sequential assignment: the first *S0* boxes (sorted top-to-bottom)
    belong to system 0, the next *S1* to system 1, etc.  Any surplus
    boxes beyond what the SVG expects are left unassigned.
    """
    if not systems or not staff_boxes:
        return {}

    result: Dict[int, Tuple[int, int]] = {}
    box_idx = 0
    for sys_idx, sys_info in enumerate(systems):
        for pos in range(sys_info.staves_per_system):
            if box_idx >= len(staff_boxes):
                break
            result[box_idx] = (sys_idx, pos)
            box_idx += 1
    return result


def _build_system_yolo_objects(
    staff_boxes: Sequence[Tuple[float, float, float, float]],
    svg_layout: Sequence["_SvgSystemInfo"],
) -> Tuple[List[Tuple[int, Tuple[float, float, float, float]]], List[int]]:
    """Group per-staff bboxes into per-system YOLO label objects.

    Uses Verovio's authoritative system y-range from each ``_SvgSystemInfo`` to
    assign staff bboxes to systems by overlap. Falls back to sequential
    assignment when overlap-based fails (defensive — should rarely happen since
    Verovio's y-range is authoritative).

    The overlap-based approach correctly handles pages with extra
    ``<g class="staff">`` elements (ossia/annotation snippets) that aren't
    structurally part of any system: those bboxes won't y-overlap any system
    and get dropped.

    Returns a pair ``(label_objects, staves_in_system)`` where:
      - ``label_objects`` is ``[(0, (x, y, w, h)), ...]`` — class 0, page-pixel coords
      - ``staves_in_system`` is a parallel list of stave counts
    """
    if not staff_boxes or not svg_layout:
        return [], []

    # Step 1: assign each staff to system by y-center containment, or closest
    # system if y-center is outside all ranges (boundary cases).
    candidates: Dict[int, List[Tuple[int, float]]] = {}
    for box_idx, (_sx, sy, sw, sh) in enumerate(staff_boxes):
        s_y_center = sy + sh / 2.0
        best_sys = None
        for i, sys_info in enumerate(svg_layout):
            if sys_info.y_top <= s_y_center <= sys_info.y_bottom:
                best_sys = i
                break
        if best_sys is None:
            best_dist = float("inf")
            for i, sys_info in enumerate(svg_layout):
                sys_center = (sys_info.y_top + sys_info.y_bottom) / 2.0
                d = abs(s_y_center - sys_center)
                if d < best_dist:
                    best_dist = d
                    best_sys = i
        if best_sys is not None:
            candidates.setdefault(best_sys, []).append((box_idx, sw))

    # Step 2: enforce Verovio's authoritative staves_per_system count per
    # system. When more staves match than expected, keep the widest ones —
    # narrow staves are usually ossias / annotation snippets, not structural
    # parts of the system.
    by_system: Dict[int, List[int]] = {}
    for sys_idx, cands in candidates.items():
        max_count = svg_layout[sys_idx].staves_per_system
        cands.sort(key=lambda t: -t[1])  # widest first
        by_system[sys_idx] = [c[0] for c in cands[:max_count]]

    label_objects: List[Tuple[int, Tuple[float, float, float, float]]] = []
    staves_in_system: List[int] = []
    for sys_idx in sorted(by_system.keys()):
        indices = by_system[sys_idx]
        if not indices:
            continue
        x_min = min(staff_boxes[i][0] for i in indices)
        y_min = min(staff_boxes[i][1] for i in indices)
        x_max = max(staff_boxes[i][0] + staff_boxes[i][2] for i in indices)
        y_max = max(staff_boxes[i][1] + staff_boxes[i][3] for i in indices)
        # Pull the bbox left edge out to include the bracket: Verovio's system
        # bounding-box rect's x is at the bracket position. This gives YOLO a
        # distinctive visual anchor for system detection (brackets are unique
        # markers that don't appear elsewhere on the page).
        if 0 <= sys_idx < len(svg_layout):
            sys_x_left = svg_layout[sys_idx].x_left
            if sys_x_left is not None:
                x_min = min(x_min, sys_x_left)
        bbox = (x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min))
        label_objects.append((0, bbox))
        staves_in_system.append(len(indices))

    return label_objects, staves_in_system


def _extract_measure_range_from_sequence(
    staff_sequence: Sequence[str],
    measure_start: int,
    measure_end: int,
) -> List[str]:
    """Extract an absolute measure range ``[measure_start, measure_end)``
    from a full staff token sequence.

    Preserves the staff prefix (clef, key sig, time sig) and wraps in
    ``<bos>/<eos>`` and ``<staff_start>/<staff_end>``.
    """
    tokens = list(staff_sequence)

    try:
        staff_start_idx = tokens.index("<staff_start>")
        staff_end_idx = len(tokens) - 1 - tokens[::-1].index("<staff_end>")
    except ValueError:
        return tokens
    if staff_end_idx <= staff_start_idx:
        return tokens

    # Prefix tokens (clef, key sig, time sig) before the first measure
    first_measure_idx = None
    for idx in range(staff_start_idx + 1, staff_end_idx):
        if tokens[idx] == "<measure_start>":
            first_measure_idx = idx
            break
    if first_measure_idx is None:
        return tokens

    token_prefix = tokens[staff_start_idx + 1 : first_measure_idx]

    # Collect all measures
    measures: List[List[str]] = []
    idx = first_measure_idx
    while idx < staff_end_idx:
        if tokens[idx] != "<measure_start>":
            idx += 1
            continue
        end_idx = idx + 1
        while end_idx < staff_end_idx and tokens[end_idx] != "<measure_end>":
            end_idx += 1
        if end_idx >= staff_end_idx:
            break
        measures.append(tokens[idx : end_idx + 1])
        idx = end_idx + 1

    if not measures:
        return tokens

    # Clamp range
    clamped_start = max(0, min(measure_start, len(measures)))
    clamped_end = max(clamped_start, min(measure_end, len(measures)))
    selected = measures[clamped_start:clamped_end]

    if not selected:
        return ["<bos>", "<staff_start>"] + token_prefix + ["<staff_end>", "<eos>"]

    result: List[str] = ["<bos>", "<staff_start>"]
    result.extend(token_prefix)
    for measure_tokens in selected:
        result.extend(measure_tokens)
    result.append("<staff_end>")
    result.append("<eos>")
    return result


def _write_staff_crops(
    *,
    svg_text: str,
    staff_boxes: Sequence[Tuple[float, float, float, float]],
    source_page_width: float,
    source_page_height: float,
    output_dir: Path,
    page_basename: str,
    vertical_padding_ratio: float = 0.35,
    horizontal_padding_ratio: float = 0.03,
    min_ink_ratio: float = 0.03,
    max_border_ink_ratio: float = 0.25,
    max_crops: Optional[int] = None,
) -> List[Tuple[Path, int]]:
    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_entries: List[Tuple[Path, int]] = []
    if max_crops is not None and max_crops <= 0:
        return crop_entries
    with _rasterize_svg_to_image(svg_text) as page_image:
        image = page_image.copy()
    image_width, image_height = image.size
    scale_x = float(image_width) / float(max(source_page_width, 1e-6))
    scale_y = float(image_height) / float(max(source_page_height, 1e-6))
    scaled_staff_boxes: List[Tuple[float, float, float, float]] = []
    for x, y, w, h in staff_boxes:
        scaled_staff_boxes.append((x * scale_x, y * scale_y, w * scale_x, h * scale_y))
    scaled_staff_boxes = sorted(scaled_staff_boxes, key=lambda item: (item[1], item[0]))
    if max_crops is not None and max_crops > 1 and len(scaled_staff_boxes) > max_crops:
        grouped: List[List[Tuple[float, float, float, float]]] = [[] for _ in range(max_crops)]
        for idx, box in enumerate(scaled_staff_boxes):
            grouped[idx % max_crops].append(box)
        merged_groups: List[Tuple[float, float, float, float]] = []
        for group in grouped:
            if not group:
                continue
            gx_min = min(item[0] for item in group)
            gy_min = min(item[1] for item in group)
            gx_max = max(item[0] + item[2] for item in group)
            gy_max = max(item[1] + item[3] for item in group)
            merged_groups.append((gx_min, gy_min, max(0.0, gx_max - gx_min), max(0.0, gy_max - gy_min)))
        scaled_staff_boxes = sorted(merged_groups, key=lambda item: (item[1], item[0]))

    def _ink_ratio(crop_array: np.ndarray) -> float:
        return float((crop_array < 245).mean()) if crop_array.size else 0.0

    candidate_crops: List[Tuple[int, float, "Image.Image"]] = []
    for idx, (scaled_x, scaled_y, scaled_w, scaled_h) in enumerate(scaled_staff_boxes):
        index = idx + 1
        x_pad = scaled_w * horizontal_padding_ratio
        y_pad = scaled_h * vertical_padding_ratio
        prev_bottom = None
        next_top = None
        if idx > 0:
            prev_box = scaled_staff_boxes[idx - 1]
            prev_bottom = prev_box[1] + prev_box[3]
        if idx + 1 < len(scaled_staff_boxes):
            next_box = scaled_staff_boxes[idx + 1]
            next_top = next_box[1]

        edge_up_allowance = max(8, int(round(min(scaled_h * 0.6, image_height * 0.06))))
        edge_down_allowance = max(8, int(round(min(scaled_h * 0.6, image_height * 0.06))))
        if prev_bottom is not None:
            min_top = max(0, int(round((prev_bottom + scaled_y) / 2.0)))
        else:
            min_top = max(0, int(round(scaled_y - edge_up_allowance)))
        if next_top is not None:
            max_bottom = min(image_height, int(round((scaled_y + scaled_h + next_top) / 2.0)))
        else:
            max_bottom = min(image_height, int(round(scaled_y + scaled_h + edge_down_allowance)))

        left = max(0, int(round(scaled_x - x_pad)))
        top = max(min_top, int(round(scaled_y - y_pad)))
        right = min(image_width, int(round(scaled_x + scaled_w + x_pad)))
        bottom = min(max_bottom, int(round(scaled_y + scaled_h + y_pad)))
        if right <= left or bottom <= top:
            continue

        # Expand adaptively when symbols touch crop borders (prevents half-cut noteheads/stems).
        expansion_step = max(8, int(round(scaled_h * 0.20)))
        for _ in range(5):
            crop = image.crop((left, top, right, bottom))
            crop_array = np.asarray(crop, dtype=np.uint8)
            if crop_array.size == 0:
                break
            border = max(1, int(round(crop_array.shape[0] * 0.08)))
            top_touch = float((crop_array[:border, :] < 245).mean())
            bottom_touch = float((crop_array[-border:, :] < 245).mean())
            changed = False
            if top_touch > 0.015 and top > 0:
                top = max(min_top, top - expansion_step)
                changed = True
            if bottom_touch > 0.015 and bottom < image_height:
                bottom = min(max_bottom, bottom + expansion_step)
                changed = True
            if not changed:
                break
            if right <= left or bottom <= top:
                break

        if right <= left or bottom <= top:
            continue
        crop = image.crop((left, top, right, bottom))
        crop_array = np.asarray(crop, dtype=np.uint8)
        ink_ratio = _ink_ratio(crop_array)
        if ink_ratio < min_ink_ratio:
            continue
        border = max(1, int(round(crop_array.shape[0] * 0.08)))
        top_touch = float((crop_array[:border, :] < 245).mean())
        bottom_touch = float((crop_array[-border:, :] < 245).mean())
        if top_touch > max_border_ink_ratio or bottom_touch > max_border_ink_ratio:
            continue
        quality_score = ink_ratio - (0.5 * (top_touch + bottom_touch))
        candidate_crops.append((index, quality_score, crop))

    if max_crops is not None and len(candidate_crops) > max_crops:
        candidate_crops = sorted(candidate_crops, key=lambda item: item[1], reverse=True)[:max_crops]
    candidate_crops = sorted(candidate_crops, key=lambda item: item[0])
    for saved_index, (source_index, _, crop_image) in enumerate(candidate_crops, start=1):
        crop_path = output_dir / f"{page_basename}__staff{saved_index:02d}.png"
        crop_image.save(crop_path)
        crop_entries.append((crop_path, max(0, source_index - 1)))
    return crop_entries


def validate_musicxml_roundtrip(
    source_path: Path,
    style_options: Dict[str, object],
    page_number: int,
    reference_svg_text: str,
    warning_counts: Optional[Counter[str]] = None,
    show_verovio_warnings: bool = False,
) -> Optional[Dict[str, float]]:
    if source_path.suffix.lower() not in {".mxl", ".musicxml", ".xml"}:
        return None
    try:
        import numpy as np
        import verovio
        from PIL import Image
        from skimage.metrics import structural_similarity
    except ImportError:
        return None

    roundtrip_svg: Optional[str]
    capture = _NativeStderrCapture(enabled=not show_verovio_warnings)
    with capture:
        toolkit = verovio.toolkit()
        toolkit.loadFile(str(source_path))
        toolkit.setOptions(style_options)
        toolkit.redoLayout()
        page_count = int(toolkit.getPageCount())
        if page_number > page_count:
            roundtrip_svg = None
        else:
            roundtrip_svg = toolkit.renderToSVG(page_number)
    if warning_counts is not None:
        _record_verovio_warnings(capture.lines, warning_counts)
    if roundtrip_svg is None:
        return None

    ref = _rasterize_svg_to_array(reference_svg_text)
    pred = _rasterize_svg_to_array(roundtrip_svg)
    if ref.shape != pred.shape:
        pred_image = Image.fromarray(pred.astype("uint8"), mode="L").resize((ref.shape[1], ref.shape[0]))
        pred = pred_image.copy()
        pred = np.asarray(pred, dtype=np.float32)

    mse = float(((ref - pred) ** 2).mean())
    ssim = float(structural_similarity(ref, pred, data_range=255))
    return {"mse": mse, "ssim": ssim}


def convert_source_tokens(source_path: Path) -> List[List[str]]:
    suffix = source_path.suffix.lower()
    if suffix not in {".mxl", ".musicxml", ".xml", ".mscx"}:
        raise ValueError(f"Unsupported symbolic source for token conversion: {source_path}")
    from src.data.convert_tokens import convert_musicxml_file, validate_token_sequence

    score_tokens = convert_musicxml_file(source_path)
    try:
        validate_token_sequence(score_tokens, strict=True)
    except Exception:
        validate_token_sequence(score_tokens, strict=False)
    staff_sequences = _split_staff_token_sequences(score_tokens)
    if not staff_sequences:
        raise ValueError(f"No staff-level token sequences parsed from: {source_path}")
    return staff_sequences


def _render_single_job(
    job: RenderJob,
    *,
    project_root: Path,
    max_pages_per_score: Optional[int],
    write_png: bool,
    dpis: Sequence[int] = (300,),
    roundtrip_validate: bool,
    show_verovio_warnings: bool,
    svg_root: Path,
    png_root: Path,
    staff_crop_root: Path,
    label_root: Path,
    label_systems_root: Path,
    token_cache: Dict[str, Optional[List[List[str]]]],
    allow_fallback_labels: bool,
) -> Dict[str, object]:
    style_options = dict(DEFAULT_STYLE_PRESETS[job.style_id])
    score_id = sanitize_relpath_for_id(job.source_relpath)
    source_key = str(job.source_path.resolve())

    warning_counts: Counter[str] = Counter()
    yolo_counts = {name: 0 for name in YOLO_CLASS_NAMES}
    token_entries_by_dataset: Dict[str, int] = {}
    page_rows: List[Dict[str, object]] = []
    token_rows: List[Dict[str, object]] = []
    roundtrip_results: List[Dict[str, float]] = []
    pages_rendered = 0
    total_staff_boxes = 0
    token_entries_written = 0
    token_pairing_mismatches = 0
    yolo_labels_rejected = 0
    yolo_reject_reasons: Counter[str] = Counter()

    try:
        page_iterator = render_svg_pages(
            source_path=job.source_path,
            style_options=style_options,
            max_pages_per_score=max_pages_per_score,
            warning_counts=warning_counts,
            show_verovio_warnings=show_verovio_warnings,
        )
        rendered_pages_for_job = list(page_iterator)
        page_count_for_job = len(rendered_pages_for_job)
        staff_token_sequences = token_cache.get(source_key)
        part_count = len(staff_token_sequences) if staff_token_sequences else 0
        cumulative_measure_offset = 0

        for page_no, svg_text in rendered_pages_for_job:
            pages_rendered += 1
            page_basename = f"{score_id}__{job.style_id}__p{page_no:03d}"
            output_svg = svg_root / job.style_id / f"{page_basename}.svg"
            output_svg.parent.mkdir(parents=True, exist_ok=True)
            output_svg.write_text(svg_text, encoding="utf-8")

            page_width, page_height, objects = extract_page_objects(
                svg_text,
                allow_fallback_objects=allow_fallback_labels,
            )
            page_ink_array = None
            try:
                page_ink_array = _rasterize_svg_to_array(svg_text)
            except Exception:
                page_ink_array = None
            objects = _ensure_metadata_regions(
                objects,
                page_width=page_width,
                page_height=page_height,
                page_ink_array=page_ink_array,
            )
            output_label = label_root / job.style_id / f"{page_basename}.txt"
            output_label_systems = label_systems_root / job.style_id / f"{page_basename}.txt"
            staff_boxes = _sorted_staff_boxes(objects)
            barline_class_id = YOLO_CLASS_TO_ID.get("barline_system")
            barline_boxes = [
                bbox
                for class_id, bbox in objects
                if barline_class_id is not None and class_id == barline_class_id and bbox[2] > 0 and bbox[3] > 0
            ]
            staff_measure_weights = _estimate_staff_measure_weights(staff_boxes, barline_boxes)

            expected_min_staff_count = 1
            if job.score_type in {"piano", "solo_instrument_with_piano", "chamber", "choral"}:
                expected_min_staff_count = 2
            elif job.score_type == "orchestral":
                expected_min_staff_count = 4

            label_valid, reject_reasons = _assess_yolo_label_quality(
                objects,
                page_width=page_width,
                page_height=page_height,
                page_ink_array=page_ink_array,
                expected_min_staff_count=expected_min_staff_count,
            )
            label_objects = list(objects) if label_valid else []
            label_written = False
            if label_valid:
                write_yolo_labels(output_label, label_objects, page_width=page_width, page_height=page_height)
                label_written = True
            else:
                if output_label.exists():
                    output_label.unlink()
                yolo_labels_rejected += 1
                for reason in reject_reasons:
                    yolo_reject_reasons[reason] += 1

            # System-level labels: parsed once here and reused below for token pairing.
            svg_layout = _extract_system_layout_from_svg(svg_text)
            staves_in_system: List[int] = []
            label_systems_written = False
            if label_valid and svg_layout:
                system_label_objects, staves_in_system = _build_system_yolo_objects(staff_boxes, svg_layout)
                if system_label_objects:
                    write_yolo_labels(
                        output_label_systems,
                        system_label_objects,
                        page_width=page_width,
                        page_height=page_height,
                    )
                    output_label_systems.with_suffix(".staves.json").write_text(
                        json.dumps(staves_in_system), encoding="utf-8",
                    )
                    label_systems_written = True
            if not label_systems_written and output_label_systems.exists():
                output_label_systems.unlink()
                sidecar = output_label_systems.with_suffix(".staves.json")
                if sidecar.exists():
                    sidecar.unlink()

            class_counter = Counter()
            for class_id, _ in label_objects:
                class_name = YOLO_CLASS_NAMES[class_id]
                class_counter[class_name] += 1
                yolo_counts[class_name] += 1
                if class_name == "staff":
                    total_staff_boxes += 1

            output_png: Optional[Path] = None
            if write_png:
                for dpi in dpis:
                    dpi_dir = png_root / f"dpi{dpi}" / job.style_id
                    candidate_png = dpi_dir / f"{page_basename}.png"
                    if maybe_write_png(svg_text, candidate_png, dpi=dpi):
                        if output_png is None:
                            output_png = candidate_png

            roundtrip = None
            if roundtrip_validate:
                roundtrip = validate_musicxml_roundtrip(
                    source_path=job.source_path,
                    style_options=style_options,
                    page_number=page_no,
                    reference_svg_text=svg_text,
                    warning_counts=warning_counts,
                    show_verovio_warnings=show_verovio_warnings,
                )
                if roundtrip is not None:
                    roundtrip_results.append(roundtrip)

            staff_token_pairs = 0
            staff_crop_saved_count = 0
            if staff_token_sequences is not None and part_count > 0:
                crop_output_dir = staff_crop_root / job.style_id
                staff_crop_entries = _write_staff_crops(
                    svg_text=svg_text,
                    staff_boxes=staff_boxes,
                    source_page_width=page_width,
                    source_page_height=page_height,
                    output_dir=crop_output_dir,
                    page_basename=page_basename,
                    max_crops=None,
                )
                staff_crop_saved_count = len(staff_crop_entries)

                if not staff_crop_entries:
                    token_pairing_mismatches += 1
                else:
                    # --- Primary path: SVG-based exact measure mapping ---
                    # svg_layout was hoisted above (alongside system-label emission)
                    use_svg = svg_layout is not None and len(svg_layout) > 0

                    if use_svg:
                        svg_measures = [s.measure_count for s in svg_layout]
                        # Assign staff boxes to systems using SVG bounding-box
                        # y-ranges (robust to extra/fewer detected boxes).
                        staff_to_system = _assign_staff_boxes_to_systems(
                            staff_boxes, svg_layout,
                        )

                        for staff_index, (crop_path, source_staff_index) in enumerate(staff_crop_entries):
                            mapping = staff_to_system.get(source_staff_index)
                            if mapping is None:
                                continue
                            sys_idx, pos_in_system = mapping
                            if sys_idx >= len(svg_measures):
                                continue
                            pi = pos_in_system % part_count
                            if pi >= part_count:
                                continue
                            m_start = cumulative_measure_offset + sum(svg_measures[:sys_idx])
                            m_end = m_start + svg_measures[sys_idx]
                            token_sequence = _extract_measure_range_from_sequence(
                                staff_token_sequences[pi], m_start, m_end
                            )

                            base_sample_id = f"{page_basename}__staff{staff_index + 1:02d}"
                            dataset_variants: List[Tuple[str, str]] = [("synthetic_fullpage", "")]
                            if job.score_type in {"piano", "chamber", "solo_instrument_with_piano"}:
                                dataset_variants.append(("synthetic_polyphonic", "__poly"))
                            for dataset_name, sample_suffix in dataset_variants:
                                sample_id = f"{base_sample_id}{sample_suffix}"
                                token_rows.append(
                                    {
                                        "sample_id": sample_id,
                                        "dataset": dataset_name,
                                        "split": "train",
                                        "image_path": relpath(project_root, crop_path),
                                        "page_id": page_basename,
                                        "source_path": job.source_relpath,
                                        "style_id": job.style_id,
                                        "page_number": page_no,
                                        "staff_index": staff_index,
                                        "source_format": "musicxml",
                                        "score_type": job.score_type,
                                        "token_sequence": token_sequence,
                                        "token_count": len(token_sequence),
                                    }
                                )
                                token_entries_written += 1
                                token_entries_by_dataset[dataset_name] = token_entries_by_dataset.get(dataset_name, 0) + 1
                            staff_token_pairs += 1

                        cumulative_measure_offset += sum(svg_measures)
                    else:
                        # --- Fallback: proportional slicing (old behaviour) ---
                        page_staff_sequences = [
                            _slice_staff_sequence_for_page(
                                sequence,
                                page_number=page_no,
                                page_count=page_count_for_job,
                            )
                            for sequence in staff_token_sequences
                        ]
                        fragment_counts = [0 for _ in range(part_count)]
                        fragment_weights_fb: List[List[int]] = [[] for _ in range(part_count)]
                        for _, source_staff_index in staff_crop_entries:
                            pi_fb = source_staff_index % part_count
                            fragment_counts[pi_fb] += 1
                            weight = 1
                            if 0 <= source_staff_index < len(staff_measure_weights):
                                weight = max(1, int(staff_measure_weights[source_staff_index]))
                            fragment_weights_fb[pi_fb].append(weight)

                        if min(fragment_counts) <= 0:
                            token_pairing_mismatches += 1
                            pair_count = min(len(staff_crop_entries), part_count)
                            for staff_index in range(pair_count):
                                crop_path, _ = staff_crop_entries[staff_index]
                                token_sequence = page_staff_sequences[staff_index]
                                base_sample_id = f"{page_basename}__staff{staff_index + 1:02d}"
                                dataset_variants: List[Tuple[str, str]] = [("synthetic_fullpage", "")]
                                if job.score_type in {"piano", "chamber", "solo_instrument_with_piano"}:
                                    dataset_variants.append(("synthetic_polyphonic", "__poly"))
                                for dataset_name, sample_suffix in dataset_variants:
                                    sample_id = f"{base_sample_id}{sample_suffix}"
                                    token_rows.append(
                                        {
                                            "sample_id": sample_id,
                                            "dataset": dataset_name,
                                            "split": "train",
                                            "image_path": relpath(project_root, crop_path),
                                            "page_id": page_basename,
                                            "source_path": job.source_relpath,
                                            "style_id": job.style_id,
                                            "page_number": page_no,
                                            "staff_index": staff_index,
                                            "source_format": "musicxml",
                                            "score_type": job.score_type,
                                            "token_sequence": token_sequence,
                                            "token_count": len(token_sequence),
                                        }
                                    )
                                    token_entries_written += 1
                                    token_entries_by_dataset[dataset_name] = token_entries_by_dataset.get(dataset_name, 0) + 1
                                staff_token_pairs += 1
                        else:
                            part_fragment_seen = [0 for _ in range(part_count)]
                            for staff_index, (crop_path, source_staff_index) in enumerate(staff_crop_entries):
                                pi_fb = source_staff_index % part_count
                                part_fragment_seen[pi_fb] += 1
                                fragment_index = part_fragment_seen[pi_fb]
                                part_weights = fragment_weights_fb[pi_fb] if pi_fb < len(fragment_weights_fb) else None
                                token_sequence = _slice_staff_sequence_for_fragment(
                                    page_staff_sequences[pi_fb],
                                    fragment_index=fragment_index,
                                    fragment_count=fragment_counts[pi_fb],
                                    fragment_weights=part_weights,
                                )
                                base_sample_id = f"{page_basename}__staff{staff_index + 1:02d}"
                                dataset_variants: List[Tuple[str, str]] = [("synthetic_fullpage", "")]
                                if job.score_type in {"piano", "chamber", "solo_instrument_with_piano"}:
                                    dataset_variants.append(("synthetic_polyphonic", "__poly"))
                                for dataset_name, sample_suffix in dataset_variants:
                                    sample_id = f"{base_sample_id}{sample_suffix}"
                                    token_rows.append(
                                        {
                                            "sample_id": sample_id,
                                            "dataset": dataset_name,
                                            "split": "train",
                                            "image_path": relpath(project_root, crop_path),
                                            "page_id": page_basename,
                                            "source_path": job.source_relpath,
                                            "style_id": job.style_id,
                                            "page_number": page_no,
                                            "staff_index": staff_index,
                                            "source_format": "musicxml",
                                            "score_type": job.score_type,
                                            "token_sequence": token_sequence,
                                            "token_count": len(token_sequence),
                                        }
                                    )
                                    token_entries_written += 1
                                    token_entries_by_dataset[dataset_name] = token_entries_by_dataset.get(dataset_name, 0) + 1
                                staff_token_pairs += 1

                        # Estimate cumulative offset for fallback path
                        if page_count_for_job > 0:
                            max_measures = max(
                                (sum(1 for t in seq if t == "<measure_start>") for seq in staff_token_sequences),
                                default=0,
                            )
                            cumulative_measure_offset += max(1, max_measures // max(1, page_count_for_job))

            page_rows.append(
                {
                    "page_id": page_basename,
                    "source_path": job.source_relpath,
                    "style_id": job.style_id,
                    "score_type": job.score_type,
                    "page_number": page_no,
                    "svg_path": relpath(project_root, output_svg),
                    "png_path": relpath(project_root, output_png) if output_png else None,
                    "label_path": relpath(project_root, output_label) if label_written else None,
                    "label_systems_path": relpath(project_root, output_label_systems) if label_systems_written else None,
                    "staves_in_system": staves_in_system,
                    "yolo_label_valid": label_valid,
                    "yolo_reject_reasons": reject_reasons,
                    "page_width": page_width,
                    "page_height": page_height,
                    "class_counts": dict(class_counter),
                    "token_count": staff_token_pairs,
                    "has_token_sequence": staff_token_pairs > 0,
                    "staff_crop_count": staff_crop_saved_count,
                    "paired_staff_tokens": staff_token_pairs,
                    "render_options": style_options,
                    "roundtrip_validation": roundtrip,
                }
            )
    except Exception as exc:
        return {
            "failed": True,
            "source_path": job.source_relpath,
            "style_id": job.style_id,
            "error": str(exc),
            "warning_counts": dict(warning_counts),
        }

    return {
        "failed": False,
        "source_path": job.source_relpath,
        "style_id": job.style_id,
        "pages_rendered": pages_rendered,
        "total_staff_boxes": total_staff_boxes,
        "yolo_class_counts": yolo_counts,
        "roundtrip_results": roundtrip_results,
        "token_entries_written": token_entries_written,
        "token_entries_by_dataset": token_entries_by_dataset,
        "token_pairing_mismatches": token_pairing_mismatches,
        "yolo_labels_rejected": yolo_labels_rejected,
        "yolo_reject_reasons": dict(yolo_reject_reasons),
        "page_rows": page_rows,
        "token_rows": token_rows,
        "warning_counts": dict(warning_counts),
    }


def _render_source_task(
    task: RenderSourceTask,
    *,
    style_ids: Sequence[str],
    project_root: Path,
    max_pages_per_score: Optional[int],
    write_png: bool,
    dpis: Sequence[int] = (300,),
    roundtrip_validate: bool,
    show_verovio_warnings: bool,
    svg_root: Path,
    png_root: Path,
    staff_crop_root: Path,
    label_root: Path,
    label_systems_root: Path,
    allow_fallback_labels: bool,
) -> Dict[str, object]:
    source_key = str(task.source_path.resolve())
    token_cache: Dict[str, Optional[List[List[str]]]] = {}
    token_sources_converted = 0
    token_sources_failed = 0
    token_failures: List[Dict[str, str]] = []

    try:
        token_cache[source_key] = convert_source_tokens(task.source_path)
        token_sources_converted = 1
    except Exception as exc:
        token_cache[source_key] = None
        token_sources_failed = 1
        token_failures.append(
            {
                "source_path": task.source_relpath,
                "error": str(exc),
            }
        )

    job_results: List[Dict[str, object]] = []
    for style_id in style_ids:
        job = RenderJob(
            source_path=task.source_path,
            source_relpath=task.source_relpath,
            style_id=style_id,
            score_type=task.score_type,
        )
        result = _render_single_job(
            job,
            project_root=project_root,
            max_pages_per_score=max_pages_per_score,
            write_png=write_png,
            dpis=dpis,
            roundtrip_validate=roundtrip_validate,
            show_verovio_warnings=show_verovio_warnings,
            svg_root=svg_root,
            png_root=png_root,
            staff_crop_root=staff_crop_root,
            label_root=label_root,
            label_systems_root=label_systems_root,
            token_cache=token_cache,
            allow_fallback_labels=allow_fallback_labels,
        )
        job_results.append(result)

    return {
        "token_sources_converted": token_sources_converted,
        "token_sources_failed": token_sources_failed,
        "token_failures": token_failures,
        "job_results": job_results,
    }


def run(
    project_root: Path,
    data_root: Path,
    input_manifest: Optional[Path],
    output_dir: Path,
    style_ids: Sequence[str],
    max_scores: Optional[int],
    max_pages_per_score: Optional[int],
    seed: int,
    render: bool,
    write_png: bool,
    dpis: Sequence[int] = (300,),
    roundtrip_validate: bool = False,
    show_verovio_warnings: bool = False,
    workers: int = 1,
    allow_fallback_labels: bool = False,
) -> Dict[str, object]:
    unknown_styles = [style_id for style_id in style_ids if style_id not in DEFAULT_STYLE_PRESETS]
    if unknown_styles:
        raise ValueError(f"Unknown style id(s): {unknown_styles}")

    sources, source_type_counts = select_sources(
        project_root=project_root,
        data_root=data_root,
        input_manifest=input_manifest,
        max_scores=max_scores,
        seed=seed,
    )
    jobs = build_jobs(project_root=project_root, sources=sources, style_ids=style_ids)
    source_tasks = [
        RenderSourceTask(
            source_path=source,
            source_relpath=relpath(project_root, source),
            score_type=infer_score_type(source),
        )
        for source in sources
    ]

    synthetic_root = output_dir.resolve()
    svg_root = synthetic_root / "pages"
    png_root = synthetic_root / "images"
    staff_crop_root = synthetic_root / "staff_crops"
    label_root = synthetic_root / "labels"
    label_systems_root = synthetic_root / "labels_systems"
    manifest_root = synthetic_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    page_manifest_path = manifest_root / "synthetic_pages.jsonl"
    token_manifest_path = manifest_root / "synthetic_token_manifest.jsonl"
    summary_path = manifest_root / "synthetic_summary.json"
    job_plan_path = manifest_root / "render_jobs.jsonl"

    total_staff_boxes = 0
    rendered_pages = 0
    planned_jobs = 0
    jobs_skipped = 0
    style_page_counts: Dict[str, int] = {style_id: 0 for style_id in style_ids}
    yolo_class_counts = {name: 0 for name in YOLO_CLASS_NAMES}
    roundtrip_results: List[Dict[str, float]] = []
    token_sources_converted = 0
    token_sources_failed = 0
    token_entries_written = 0
    token_entries_by_dataset: Dict[str, int] = {}
    token_pairing_mismatches = 0
    yolo_labels_rejected = 0
    yolo_reject_reason_counts: Counter[str] = Counter()
    token_failures: List[Dict[str, str]] = []
    render_failures: List[Dict[str, str]] = []
    verovio_warning_counts: Counter[str] = Counter()

    with job_plan_path.open("w", encoding="utf-8") as plan_file, page_manifest_path.open(
        "w", encoding="utf-8"
    ) as page_manifest_file, token_manifest_path.open("w", encoding="utf-8") as token_manifest_file:
        for job in jobs:
            planned_jobs += 1
            score_id = sanitize_relpath_for_id(job.source_relpath)
            plan_file.write(
                json.dumps(
                    {
                        "source_path": job.source_relpath,
                        "style_id": job.style_id,
                        "score_id": score_id,
                        "score_type": job.score_type,
                    }
                )
                + "\n"
            )
        if render:
            worker_count = max(1, int(workers))
            worker_fn = partial(
                _render_source_task,
                style_ids=tuple(style_ids),
                project_root=project_root,
                max_pages_per_score=max_pages_per_score,
                write_png=write_png,
                dpis=tuple(dpis),
                roundtrip_validate=roundtrip_validate,
                show_verovio_warnings=show_verovio_warnings,
                svg_root=svg_root,
                png_root=png_root,
                staff_crop_root=staff_crop_root,
                label_root=label_root,
                label_systems_root=label_systems_root,
                allow_fallback_labels=allow_fallback_labels,
            )

            def _consume_result(result: Dict[str, object]) -> None:
                nonlocal jobs_skipped, rendered_pages, total_staff_boxes, token_entries_written, token_pairing_mismatches
                nonlocal yolo_labels_rejected

                for warning, count in dict(result.get("warning_counts", {})).items():
                    verovio_warning_counts[str(warning)] += int(count)

                if bool(result.get("failed", False)):
                    jobs_skipped += 1
                    render_failures.append(
                        {
                            "source_path": str(result.get("source_path", "")),
                            "style_id": str(result.get("style_id", "")),
                            "error": str(result.get("error", "unknown render failure")),
                        }
                    )
                    return

                style_id = str(result.get("style_id", ""))
                pages_for_style = int(result.get("pages_rendered", 0))
                style_page_counts[style_id] = style_page_counts.get(style_id, 0) + pages_for_style
                rendered_pages += pages_for_style
                total_staff_boxes += int(result.get("total_staff_boxes", 0))
                token_entries_written += int(result.get("token_entries_written", 0))
                token_pairing_mismatches += int(result.get("token_pairing_mismatches", 0))
                yolo_labels_rejected += int(result.get("yolo_labels_rejected", 0))
                for reason, count in dict(result.get("yolo_reject_reasons", {})).items():
                    yolo_reject_reason_counts[str(reason)] += int(count)

                for class_name, count in dict(result.get("yolo_class_counts", {})).items():
                    if class_name in yolo_class_counts:
                        yolo_class_counts[class_name] += int(count)

                for item in list(result.get("roundtrip_results", [])):
                    if isinstance(item, dict):
                        roundtrip_results.append(item)

                for dataset_name, count in dict(result.get("token_entries_by_dataset", {})).items():
                    token_entries_by_dataset[str(dataset_name)] = token_entries_by_dataset.get(str(dataset_name), 0) + int(
                        count
                    )

                for row in list(result.get("page_rows", [])):
                    page_manifest_file.write(json.dumps(row) + "\n")
                for row in list(result.get("token_rows", [])):
                    token_manifest_file.write(json.dumps(row) + "\n")

            def _consume_source_result(source_result: Dict[str, object]) -> None:
                nonlocal token_sources_converted, token_sources_failed

                token_sources_converted += int(source_result.get("token_sources_converted", 0))
                token_sources_failed += int(source_result.get("token_sources_failed", 0))
                for failure in list(source_result.get("token_failures", [])):
                    if isinstance(failure, dict):
                        token_failures.append(
                            {
                                "source_path": str(failure.get("source_path", "")),
                                "error": str(failure.get("error", "")),
                            }
                        )
                for result in list(source_result.get("job_results", [])):
                    if isinstance(result, dict):
                        _consume_result(result)

            if worker_count == 1:
                for task in source_tasks:
                    _consume_source_result(worker_fn(task))
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    for source_result in executor.map(worker_fn, source_tasks):
                        _consume_source_result(source_result)

    summary = {
        "render_mode": "render" if render else "dry-run",
        "scores_selected": len(sources),
        "styles_selected": list(style_ids),
        "workers": max(1, int(workers)),
        "jobs_planned": planned_jobs,
        "jobs_skipped": jobs_skipped,
        "pages_rendered": rendered_pages,
        "total_staff_boxes": total_staff_boxes,
        "pages_per_style": style_page_counts,
        "yolo_classes": YOLO_CLASS_NAMES,
        "yolo_class_counts": yolo_class_counts,
        "score_type_target_distribution": SCORE_TYPE_TARGET_DISTRIBUTION,
        "score_type_selected_counts": source_type_counts,
        "roundtrip_validation_enabled": roundtrip_validate,
        "roundtrip_samples": len(roundtrip_results),
        "roundtrip_mean_mse": (
            float(sum(item["mse"] for item in roundtrip_results) / len(roundtrip_results))
            if roundtrip_results
            else None
        ),
        "roundtrip_mean_ssim": (
            float(sum(item["ssim"] for item in roundtrip_results) / len(roundtrip_results))
            if roundtrip_results
            else None
        ),
        "job_plan_path": relpath(project_root, job_plan_path),
        "page_manifest_path": relpath(project_root, page_manifest_path),
        "token_manifest_path": relpath(project_root, token_manifest_path),
        "staff_crop_root": relpath(project_root, staff_crop_root),
        "token_entries_written": token_entries_written,
        "token_entries_by_dataset": token_entries_by_dataset,
        "token_sources_converted": token_sources_converted,
        "token_sources_failed": token_sources_failed,
        "token_pairing_mismatches": token_pairing_mismatches,
        "yolo_allow_fallback_labels": bool(allow_fallback_labels),
        "yolo_labels_rejected": yolo_labels_rejected,
        "yolo_reject_reason_counts": dict(yolo_reject_reason_counts),
        "token_failures_preview": token_failures[:20],
        "render_failures_preview": render_failures[:20],
        "show_verovio_warnings": show_verovio_warnings,
        "verovio_warnings_captured": int(sum(verovio_warning_counts.values())),
        "verovio_warning_counts": dict(verovio_warning_counts),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Synthetic full-page dataset generation pipeline.")
    parser.add_argument("--project-root", type=Path, default=project_root, help="Repository root path.")
    parser.add_argument("--data-root", type=Path, default=project_root / "data", help="Data root directory.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=project_root / "src" / "data" / "manifests" / "master_manifest.jsonl",
        help="Optional manifest for source discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed" / "synthetic",
        help="Output root directory for synthetic assets.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default="leipzig-default,bravura-compact,gootville-wide",
        help="Comma-separated style preset ids.",
    )
    parser.add_argument("--max-scores", type=int, default=None, help="Optional max number of scores.")
    parser.add_argument(
        "--max-pages-per-score",
        type=int,
        default=None,
        help="Optional max pages rendered per source score.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for render mode job processing (1 disables parallelism).",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Seed for deterministic sampling.")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "render"),
        default="dry-run",
        help="dry-run creates job plans only; render runs Verovio rendering.",
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="When rendering, rasterize SVG to PNG at a single DPI (legacy; use --dpis instead).",
    )
    parser.add_argument(
        "--dpis",
        type=int,
        nargs="+",
        default=None,
        help="One or more DPIs to rasterize each page at. Each DPI produces "
             "a parallel set of PNGs in dpi-suffixed subdirectories. Labels "
             "are normalized YOLO format and shared across DPIs. "
             "When provided, implies --write-png and takes precedence over it.",
    )
    parser.add_argument(
        "--roundtrip-validate",
        action="store_true",
        help="When rendering MusicXML inputs, run Verovio re-render SSIM/MSE validation.",
    )
    parser.add_argument(
        "--show-verovio-warnings",
        action="store_true",
        help="Print raw Verovio warnings instead of aggregating them in summary.",
    )
    parser.add_argument(
        "--allow-fallback-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow heuristic fallback boxes when SVG metadata is missing. Keep disabled for clean training labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style_ids = [style_id.strip() for style_id in args.styles.split(",") if style_id.strip()]
    input_manifest = args.input_manifest if args.input_manifest and args.input_manifest.exists() else None

    # --dpis takes precedence; fall back to [300] when --write-png was used without --dpis
    if args.dpis is not None:
        dpis = args.dpis
        write_png = True
    elif args.write_png:
        dpis = [300]
        write_png = True
    else:
        dpis = [300]
        write_png = False

    summary = run(
        project_root=args.project_root,
        data_root=args.data_root,
        input_manifest=input_manifest,
        output_dir=args.output_dir,
        style_ids=style_ids,
        max_scores=args.max_scores,
        max_pages_per_score=args.max_pages_per_score,
        seed=args.seed,
        render=args.mode == "render",
        write_png=write_png,
        dpis=dpis,
        roundtrip_validate=args.roundtrip_validate,
        show_verovio_warnings=args.show_verovio_warnings,
        workers=max(1, args.workers),
        allow_fallback_labels=bool(args.allow_fallback_labels),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

