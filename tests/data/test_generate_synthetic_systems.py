"""Unit tests for system-bbox helper in generate_synthetic.py.

Covers _build_system_yolo_objects, which groups already-pixel-space staff_boxes
by system (using Verovio's authoritative staves_per_system layout) and emits
one union bbox per system as YOLO label objects (class 0).
"""
from __future__ import annotations

import pytest

from src.data.generate_synthetic import (
    _SvgSystemInfo,
    _build_system_yolo_objects,
)


def _layout(*staves_per_system: int) -> list[_SvgSystemInfo]:
    """Build a list of _SvgSystemInfo with synthetic y-bounds (top-to-bottom)."""
    out: list[_SvgSystemInfo] = []
    y = 100.0
    for s in staves_per_system:
        h = 200.0 * s
        out.append(_SvgSystemInfo(
            measure_count=4,
            staves_per_system=s,
            y_top=y,
            y_bottom=y + h,
        ))
        y += h + 100.0  # gap between systems
    return out


def test_two_three_staff_systems_grouped():
    """6 staff bboxes split 3+3 into two systems → two union bboxes."""
    staff_boxes = [
        # System 0 (top), 3 staves spanning x=100..2100
        (100.0, 200.0, 2000.0, 80.0),
        (100.0, 320.0, 2000.0, 80.0),
        (100.0, 440.0, 2000.0, 80.0),
        # System 1 (bottom), 3 staves
        (100.0, 800.0, 2000.0, 80.0),
        (100.0, 920.0, 2000.0, 80.0),
        (100.0, 1040.0, 2000.0, 80.0),
    ]
    svg_layout = _layout(3, 3)

    label_objects, staves_per = _build_system_yolo_objects(staff_boxes, svg_layout)

    assert len(label_objects) == 2
    assert staves_per == [3, 3]

    # System 0: union of top three staves (x=100..2100, y=200..520)
    cls0, (x0, y0, w0, h0) = label_objects[0]
    assert cls0 == 0
    assert (x0, y0) == pytest.approx((100.0, 200.0))
    assert (w0, h0) == pytest.approx((2000.0, 320.0))  # 520 - 200

    # System 1: union of bottom three staves
    cls1, (x1, y1, w1, h1) = label_objects[1]
    assert cls1 == 0
    assert (x1, y1) == pytest.approx((100.0, 800.0))
    assert (w1, h1) == pytest.approx((2000.0, 320.0))


def test_single_staff_system():
    """One vocal-only system with one staff."""
    staff_boxes = [(100.0, 200.0, 1800.0, 80.0)]
    svg_layout = _layout(1)

    label_objects, staves_per = _build_system_yolo_objects(staff_boxes, svg_layout)

    assert len(label_objects) == 1
    assert staves_per == [1]
    assert label_objects[0] == (0, (100.0, 200.0, 1800.0, 80.0))


def test_empty_inputs_return_empty():
    """Either no staff boxes or no svg layout → no system labels."""
    assert _build_system_yolo_objects([], _layout(3)) == ([], [])
    assert _build_system_yolo_objects([(0.0, 0.0, 100.0, 100.0)], []) == ([], [])
    assert _build_system_yolo_objects([], []) == ([], [])


def test_unequal_staff_widths_unioned_to_widest_extent():
    """Staves of varying widths → system bbox spans widest extent."""
    staff_boxes = [
        # System with 2 staves: one shorter (vocal line), one longer (piano grand staff)
        (200.0, 300.0, 1500.0, 80.0),  # x: 200..1700
        (100.0, 420.0, 2000.0, 80.0),  # x: 100..2100 (wider)
    ]
    svg_layout = _layout(2)

    label_objects, staves_per = _build_system_yolo_objects(staff_boxes, svg_layout)

    assert len(label_objects) == 1
    assert staves_per == [2]
    cls, (x, y, w, h) = label_objects[0]
    assert cls == 0
    # Union x: 100..2100 (widest), y: 300..500
    assert (x, y) == pytest.approx((100.0, 300.0))
    assert (w, h) == pytest.approx((2000.0, 200.0))


def test_extra_staff_boxes_beyond_layout_are_dropped():
    """If detection produces more staff bboxes than svg layout expects, the
    surplus is unassigned (matching _assign_staff_boxes_to_systems behavior)
    and excluded from system unions."""
    staff_boxes = [
        # System 0 expects 2 staves
        (100.0, 200.0, 2000.0, 80.0),
        (100.0, 320.0, 2000.0, 80.0),
        # Surplus staff that doesn't belong to any layout-declared system
        (100.0, 440.0, 2000.0, 80.0),
    ]
    svg_layout = _layout(2)

    label_objects, staves_per = _build_system_yolo_objects(staff_boxes, svg_layout)

    assert len(label_objects) == 1
    assert staves_per == [2]
    # Union should NOT include the surplus third staff (y_max should be 400, not 520)
    _, (_x, _y, _w, h) = label_objects[0]
    assert h == pytest.approx(200.0)  # 400 - 200, not 520 - 200


def test_top_to_bottom_ordering_preserved():
    """Output order matches svg_layout order (top-to-bottom)."""
    staff_boxes = [
        # Two systems, each one staff
        (100.0, 200.0, 2000.0, 80.0),
        (100.0, 600.0, 2000.0, 80.0),
    ]
    svg_layout = _layout(1, 1)

    label_objects, _ = _build_system_yolo_objects(staff_boxes, svg_layout)

    assert len(label_objects) == 2
    # First (top) system has smaller y
    assert label_objects[0][1][1] < label_objects[1][1][1]
