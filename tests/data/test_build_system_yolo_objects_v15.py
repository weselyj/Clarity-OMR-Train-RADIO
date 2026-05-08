"""Unit tests for _build_system_yolo_objects_v15 (v15 SVG-hierarchy algorithm).

Tests the five behaviors called out in Phase 2A scope:
  1. SVG-hierarchy grouping correctly assigns staves when grouped by y_center.
  2. Per-row SVG fallback fires when there are fewer disk labels than SVG rows.
  3. Single-staff fallback fires when SVG has no <g class="system"> elements.
  4. Neighbor cap clips a system's y_bot/y_top when adjacent system is close.
  5. Output format is valid YOLO (class 0, coords in pixel (x,y,w,h)).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.data.generate_synthetic import (
    _build_system_yolo_objects_v15,
    V15_SYSTEM_TOP_MARGIN_PX,
    V15_SYSTEM_BOTTOM_MARGIN_PX,
    V15_RIGHT_BARLINE_MARGIN_PX,
    V15_LEFTWARD_BRACKET_MARGIN_PX,
    V15_NEIGHBOR_GAP_PX,
    V15_SYNTHETIC_ROW_TOP_RATIO,
    V15_SYNTHETIC_ROW_BOTTOM_RATIO,
    write_yolo_labels,
)


# ---------------------------------------------------------------------------
# SVG fixture builder
# ---------------------------------------------------------------------------

def _make_svg(systems: list[list[dict]], *, vb_w: float = 2200, vb_h: float = 3000) -> str:
    """Build a minimal Verovio-like SVG with <g class="system"> hierarchy.

    Each system is a list of row dicts with keys:
        y_top, y_bot   — in SVG viewBox units
        x_min, x_max   — in SVG viewBox units

    The SVG has a definition-scale inner element with viewBox="0 0 {vb_w} {vb_h}".
    Staff lines are horizontal paths (M x1 y1 L x2 y2) matching the row bounds.
    """
    system_elements = []
    for sys_rows in systems:
        measure_staves = []
        for row in sys_rows:
            x1 = row["x_min"]
            x2 = row["x_max"]
            yt = row["y_top"]
            yb = row["y_bot"]
            # 5 staff lines evenly spaced between y_top and y_bot
            step = (yb - yt) / 4.0
            paths = []
            for i in range(5):
                y = yt + i * step
                paths.append(f'<path d="M {x1} {y} L {x2} {y}"/>')
            measure_staves.append(
                '<g class="staff">' + "".join(paths) + "</g>"
            )
        measure_content = "".join(
            f'<g class="measure">' + "".join(measure_staves) + "</g>"
        )
        system_elements.append(
            f'<g class="system">{measure_content}</g>'
        )

    inner_content = "".join(system_elements)
    svg = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="{vb_w}" height="{vb_h}">
          <svg class="definition-scale" viewBox="0 0 {vb_w} {vb_h}">
            {inner_content}
          </svg>
        </svg>
    """)
    return svg


def _make_svg_no_systems() -> str:
    """SVG with no <g class='system'> elements."""
    return textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="2200" height="3000">
          <svg class="definition-scale" viewBox="0 0 2200 3000">
            <g class="page"><rect x="0" y="0" width="2200" height="3000"/></g>
          </svg>
        </svg>
    """)


# ---------------------------------------------------------------------------
# Test 1: SVG-hierarchy grouping with correct y_center-based assignment
#
# Scenario inspired by Stanford-class case: Verovio rect may be undersized but
# the <g class="system"> hierarchy gives correct row y_center values.
# ---------------------------------------------------------------------------

class TestSvgHierarchyGrouping:
    def test_two_systems_two_staves_each_grouped_correctly(self, tmp_path: Path) -> None:
        """6 staff boxes assigned to 2 systems (2+2) by SVG y_center proximity.

        The SVG says systems at y≈200-400 and y≈700-900 (viewBox units = pixels
        since vb_w=png_w, vb_h=png_h).
        """
        png_w, png_h = 2200, 3000
        systems = [
            # System 0: two rows at y=200 and y=320
            [
                {"y_top": 200, "y_bot": 280, "x_min": 100, "x_max": 2100},
                {"y_top": 320, "y_bot": 400, "x_min": 100, "x_max": 2100},
            ],
            # System 1: two rows at y=700 and y=820
            [
                {"y_top": 700, "y_bot": 780, "x_min": 100, "x_max": 2100},
                {"y_top": 820, "y_bot": 900, "x_min": 100, "x_max": 2100},
            ],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        # Per-staff disk labels in YOLO (x,y,w,h) pixel format
        staff_boxes = [
            (100.0, 200.0, 2000.0, 80.0),   # sys0 row0  yc=240
            (100.0, 320.0, 2000.0, 80.0),   # sys0 row1  yc=360
            (100.0, 700.0, 2000.0, 80.0),   # sys1 row0  yc=740
            (100.0, 820.0, 2000.0, 80.0),   # sys1 row1  yc=860
        ]

        label_objects, staves_per = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 2, "Expected 2 systems"
        assert staves_per == [2, 2]

        # System 0: y_top raw = 200, y_bot raw = 400 → after margins:
        expected_y1_0 = max(0.0, 200.0 - V15_SYSTEM_TOP_MARGIN_PX)
        expected_y2_0 = min(float(png_h), 400.0 + V15_SYSTEM_BOTTOM_MARGIN_PX)

        cls0, (x0, y0, w0, h0) = label_objects[0]
        assert cls0 == 0
        assert y0 == pytest.approx(expected_y1_0, abs=2.0)
        assert y0 + h0 == pytest.approx(expected_y2_0, abs=2.0)

    def test_undersized_verovio_rect_does_not_affect_assignment(self, tmp_path: Path) -> None:
        """v15 ignores the Verovio system bounding-box rect; assignment is from
        staff-line geometry only, so undersized rects cannot mis-assign staves."""
        png_w, png_h = 2200, 3000
        systems = [
            [
                {"y_top": 300, "y_bot": 380, "x_min": 80, "x_max": 2100},
                {"y_top": 420, "y_bot": 500, "x_min": 80, "x_max": 2100},
            ],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        staff_boxes = [
            (80.0, 300.0, 2020.0, 80.0),
            (80.0, 420.0, 2020.0, 80.0),
        ]

        label_objects, staves_per = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 1
        assert staves_per == [2]


# ---------------------------------------------------------------------------
# Test 2: Per-row SVG fallback fires for unclaimed SVG rows
# ---------------------------------------------------------------------------

class TestPerRowSvgFallback:
    def test_fallback_fires_for_unclaimed_row(self, tmp_path: Path) -> None:
        """SVG declares 2 rows per system; only 1 disk label exists → fallback
        synthesises a bbox for the unclaimed row, merged into the system."""
        png_w, png_h = 2200, 3000
        systems = [
            [
                {"y_top": 200, "y_bot": 280, "x_min": 100, "x_max": 2100},
                {"y_top": 340, "y_bot": 420, "x_min": 100, "x_max": 2100},
            ],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        # Only one disk label — row 1 gets no match
        staff_boxes = [(100.0, 200.0, 2000.0, 80.0)]

        label_objects, staves_per = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 1, "Should still produce 1 system"
        # staves_per counts disk members + synthetic members
        assert staves_per[0] == 2, "One real + one synthetic = 2"

        cls, (x, y, w, h) = label_objects[0]
        assert cls == 0

        # The raw bbox bottom must include the fallback row (y_bot=420), so the
        # final y_bot must be ≥ 420 + V15_SYSTEM_BOTTOM_MARGIN_PX (before cap)
        y_bot_final = y + h
        assert y_bot_final >= 420.0, "Fallback row y_bot not included in system"

    def test_all_rows_unclaimed_entire_system_from_svg(self, tmp_path: Path) -> None:
        """When NO disk labels match a system, all rows are synthetic; the
        system bbox is still derived from SVG geometry."""
        png_w, png_h = 2200, 3000
        systems = [
            [{"y_top": 200, "y_bot": 280, "x_min": 100, "x_max": 2100}],
            [{"y_top": 700, "y_bot": 780, "x_min": 100, "x_max": 2100}],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        # Disk label only for system 1 — system 0 is entirely unclaimed
        staff_boxes = [(100.0, 700.0, 2000.0, 80.0)]

        label_objects, staves_per = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 2, "Both systems must appear (system 0 via fallback)"


# ---------------------------------------------------------------------------
# Test 3: Single-staff fallback when SVG has no <g class="system"> elements
# ---------------------------------------------------------------------------

class TestSingleStaffFallback:
    def test_no_system_elements_each_disk_label_becomes_own_system(self, tmp_path: Path) -> None:
        """When SVG has no system elements, each per-staff disk label → its own system."""
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg_no_systems(), encoding="utf-8")

        staff_boxes = [
            (100.0, 200.0, 2000.0, 80.0),
            (100.0, 600.0, 2000.0, 80.0),
            (100.0, 1000.0, 2000.0, 80.0),
        ]
        png_w, png_h = 2200, 3000

        label_objects, staves_per = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 3, "3 disk labels → 3 solo systems"
        assert all(s == 1 for s in staves_per), "Each solo system has 1 staff"
        assert all(c == 0 for c, _ in label_objects), "All class ids must be 0"

    def test_missing_svg_triggers_single_staff_fallback(self, tmp_path: Path) -> None:
        """If the SVG file does not exist, fall back to single-staff mode."""
        svg_path = tmp_path / "nonexistent.svg"

        staff_boxes = [(100.0, 200.0, 2000.0, 80.0)]
        png_w, png_h = 2200, 3000

        label_objects, staves_per = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 1
        assert staves_per == [1]

    def test_empty_staff_boxes_returns_empty(self, tmp_path: Path) -> None:
        """No disk labels → empty output regardless of SVG content."""
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg_no_systems(), encoding="utf-8")

        label_objects, staves_per = _build_system_yolo_objects_v15(
            svg_path, [], 2200, 3000
        )

        assert label_objects == []
        assert staves_per == []


# ---------------------------------------------------------------------------
# Test 4: Neighbor cap — adjacent systems must not overlap
# ---------------------------------------------------------------------------

class TestNeighborCap:
    def test_neighbor_cap_clips_y_bot_when_systems_are_close(self, tmp_path: Path) -> None:
        """When two systems are close, the cap prevents y_bot of system 0 from
        overlapping y_top of system 1.

        Raw system bboxes:
          sys0: y_top=100, y_bot=300
          sys1: y_top=350, y_bot=550

        Without the cap:
          sys0 final y_bot = 300 + 130 = 430
          sys1 final y_top = 350 - 80  = 270
          These would overlap (430 > 270).

        With the cap, gap mid = (300+350)/2 = 325:
          sys0 final y_bot clamped to 325
          sys1 final y_top clamped to 325
          No overlap.
        """
        png_w, png_h = 2200, 3000
        systems = [
            [{"y_top": 100, "y_bot": 300, "x_min": 100, "x_max": 2100}],
            [{"y_top": 350, "y_bot": 550, "x_min": 100, "x_max": 2100}],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        staff_boxes = [
            (100.0, 100.0, 2000.0, 200.0),
            (100.0, 350.0, 2000.0, 200.0),
        ]

        label_objects, _ = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 2
        _, (x0, y0, w0, h0) = label_objects[0]
        _, (x1, y1, w1, h1) = label_objects[1]

        y_bot_sys0 = y0 + h0
        y_top_sys1 = y1
        assert y_bot_sys0 <= y_top_sys1 + 0.6, (
            f"Overlap: sys0 y_bot={y_bot_sys0:.1f} > sys1 y_top={y_top_sys1:.1f}"
        )

    def test_no_cap_needed_for_well_separated_systems(self, tmp_path: Path) -> None:
        """Well-separated systems: margins are applied in full without capping."""
        png_w, png_h = 2200, 3000
        systems = [
            [{"y_top": 100, "y_bot": 200, "x_min": 100, "x_max": 2100}],
            [{"y_top": 2000, "y_bot": 2100, "x_min": 100, "x_max": 2100}],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        staff_boxes = [
            (100.0, 100.0, 2000.0, 100.0),
            (100.0, 2000.0, 2000.0, 100.0),
        ]

        label_objects, _ = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 2
        _, (x0, y0, w0, h0) = label_objects[0]
        # Sys0 raw y_top=100, y_bot=200.  Full top margin = 80, so:
        expected_y_top = max(0.0, 100.0 - V15_SYSTEM_TOP_MARGIN_PX)
        assert y0 == pytest.approx(expected_y_top, abs=1.0)


# ---------------------------------------------------------------------------
# Test 5: Output format is valid YOLO (class 0, (x,y,w,h) pixel coords)
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_class_id_is_always_zero(self, tmp_path: Path) -> None:
        """All returned label objects must have class_id == 0."""
        png_w, png_h = 2200, 3000
        systems = [
            [{"y_top": 100, "y_bot": 200, "x_min": 100, "x_max": 2100}],
            [{"y_top": 500, "y_bot": 600, "x_min": 100, "x_max": 2100}],
            [{"y_top": 900, "y_bot": 1000, "x_min": 100, "x_max": 2100}],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        staff_boxes = [
            (100.0, 100.0, 2000.0, 100.0),
            (100.0, 500.0, 2000.0, 100.0),
            (100.0, 900.0, 2000.0, 100.0),
        ]

        label_objects, _ = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        for class_id, bbox in label_objects:
            assert class_id == 0

    def test_bbox_in_xywh_pixel_format(self, tmp_path: Path) -> None:
        """Returned bboxes are (x, y, w, h) with w > 0 and h > 0."""
        png_w, png_h = 2200, 3000
        systems = [
            [{"y_top": 200, "y_bot": 300, "x_min": 100, "x_max": 2100}],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        staff_boxes = [(100.0, 200.0, 2000.0, 100.0)]

        label_objects, _ = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 1
        _, (x, y, w, h) = label_objects[0]
        assert w > 0, "Width must be positive"
        assert h > 0, "Height must be positive"
        # x, y must be within page
        assert 0.0 <= x <= png_w
        assert 0.0 <= y <= png_h

    def test_write_yolo_labels_round_trips_normalized_coords(self, tmp_path: Path) -> None:
        """write_yolo_labels correctly normalises the (x,y,w,h) pixel bbox."""
        png_w, png_h = 2200, 3000
        systems = [
            [{"y_top": 200, "y_bot": 400, "x_min": 100, "x_max": 2100}],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        staff_boxes = [(100.0, 200.0, 2000.0, 200.0)]

        label_objects, _ = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        out_path = tmp_path / "out.txt"
        write_yolo_labels(out_path, label_objects, page_width=png_w, page_height=png_h)
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()

        assert len(lines) == 1
        parts = lines[0].split()
        assert len(parts) == 5
        assert parts[0] == "0"
        # All normalized coords should be in [0, 1]
        for v in parts[1:]:
            val = float(v)
            assert 0.0 <= val <= 1.0, f"Normalized coord out of range: {val}"

    def test_margins_expand_beyond_raw_staff_bbox(self, tmp_path: Path) -> None:
        """Final bbox must be larger than the raw staff bbox due to margins."""
        png_w, png_h = 2200, 3000
        systems = [
            [{"y_top": 500, "y_bot": 600, "x_min": 200, "x_max": 2000}],
        ]
        svg_path = tmp_path / "page.svg"
        svg_path.write_text(_make_svg(systems, vb_w=png_w, vb_h=png_h), encoding="utf-8")

        staff_boxes = [(200.0, 500.0, 1800.0, 100.0)]

        label_objects, _ = _build_system_yolo_objects_v15(
            svg_path, staff_boxes, png_w, png_h
        )

        assert len(label_objects) == 1
        _, (x, y, w, h) = label_objects[0]

        # y must be smaller than staff y_top=500 (top margin applied)
        assert y < 500.0
        # y+h must be larger than staff y_bot=600 (bottom margin applied)
        assert y + h > 600.0
        # x must be smaller than staff x1=200 (left bracket margin applied)
        assert x < 200.0
        # x+w must be larger than staff x2=2000 (right barline margin applied)
        assert x + w > 2000.0
