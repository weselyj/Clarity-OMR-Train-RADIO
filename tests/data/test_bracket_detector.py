"""Unit tests for bracket-based staff grouping.

The visual detection (detect_brackets_on_page) is smoke-tested on real images
elsewhere; here we test the pure grouping function.
"""
import pytest

from src.data.bracket_detector import group_staves_by_brackets


class TestGroupStavesByBrackets:
    def test_three_brackets_three_staves_each(self):
        """Standard piano-vocal lieder: 3 brackets each grouping vocal+treble+bass."""
        staves = [
            {"bbox": (100, 100, 2100, 130)},   # bracket 1 vocal
            {"bbox": (100, 180, 2100, 220)},   # bracket 1 treble
            {"bbox": (100, 240, 2100, 280)},   # bracket 1 bass
            {"bbox": (100, 400, 2100, 430)},   # bracket 2 vocal
            {"bbox": (100, 480, 2100, 520)},   # bracket 2 treble
            {"bbox": (100, 540, 2100, 580)},   # bracket 2 bass
            {"bbox": (100, 700, 2100, 730)},   # bracket 3 vocal
            {"bbox": (100, 780, 2100, 820)},   # bracket 3 treble
            {"bbox": (100, 840, 2100, 880)},   # bracket 3 bass
        ]
        brackets = [
            {"x": 50, "y_top": 95, "y_bottom": 285, "height": 190},
            {"x": 50, "y_top": 395, "y_bottom": 585, "height": 190},
            {"x": 50, "y_top": 695, "y_bottom": 885, "height": 190},
        ]
        systems = group_staves_by_brackets(staves, brackets)
        assert len(systems) == 3
        for s in systems:
            assert s["staves_in_system"] == 3
        # Bracket position pulled into bbox left edge
        assert systems[0]["bbox"][0] == 50

    def test_vocal_to_piano_gap_larger_than_inter_system(self):
        """Spatial heuristic fails this case; bracket detector handles it.

        Per-staff y positions where intra-bracket vocal-to-treble gap is huge
        but inter-bracket gap is small. Bracket detection makes this trivial.
        """
        staves = [
            {"bbox": (100, 100, 2100, 130)},    # br1 vocal (gap to next: 70)
            {"bbox": (100, 200, 2100, 240)},    # br1 treble
            {"bbox": (100, 250, 2100, 290)},    # br1 bass
            {"bbox": (100, 320, 2100, 350)},    # br2 vocal (only 30px gap from br1 bass!)
            {"bbox": (100, 420, 2100, 460)},    # br2 treble
            {"bbox": (100, 470, 2100, 510)},    # br2 bass
        ]
        brackets = [
            {"x": 50, "y_top": 95, "y_bottom": 295, "height": 200},
            {"x": 50, "y_top": 315, "y_bottom": 515, "height": 200},
        ]
        systems = group_staves_by_brackets(staves, brackets)
        assert len(systems) == 2
        assert all(s["staves_in_system"] == 3 for s in systems)

    def test_no_brackets_each_staff_is_own_system(self):
        """No brackets detected → each staff becomes its own 1-staff 'system'."""
        staves = [
            {"bbox": (100, 100, 2100, 140)},
            {"bbox": (100, 300, 2100, 340)},
        ]
        systems = group_staves_by_brackets(staves, [])
        assert len(systems) == 2
        assert systems[0]["staves_in_system"] == 1
        assert systems[1]["staves_in_system"] == 1

    def test_unbracketed_staff_becomes_solo_system(self):
        """A staff outside any bracket's y-range (e.g., vocal-only line above
        a bracketed grand staff) becomes its own 1-staff system."""
        staves = [
            {"bbox": (100, 50, 2100, 90)},     # solo vocal (no bracket)
            {"bbox": (100, 200, 2100, 240)},   # bracketed treble
            {"bbox": (100, 260, 2100, 300)},   # bracketed bass
        ]
        brackets = [
            {"x": 50, "y_top": 195, "y_bottom": 305, "height": 110},
        ]
        systems = group_staves_by_brackets(staves, brackets)
        assert len(systems) == 2
        # First system is the solo vocal (1 staff, sorted by y)
        assert systems[0]["staves_in_system"] == 1
        assert systems[0]["bbox"][1] == 50
        # Second is the bracketed pair
        assert systems[1]["staves_in_system"] == 2

    def test_bracket_extends_bbox_left_and_vertical(self):
        """Bracket position pulls bbox left and bracket y-range pulls vertical extent."""
        staves = [
            {"bbox": (300, 200, 2000, 240)},   # staff starts at x=300
        ]
        brackets = [
            {"x": 80, "y_top": 180, "y_bottom": 260, "height": 80},  # bracket extends past staff y
        ]
        systems = group_staves_by_brackets(staves, brackets)
        assert len(systems) == 1
        x1, y1, x2, y2 = systems[0]["bbox"]
        assert x1 == 80   # bracket pulled left edge out
        assert y1 == 180  # bracket extended top
        assert y2 == 260  # bracket extended bottom

    def test_sorted_top_to_bottom_with_unassigned(self):
        """Output is sorted by y_top regardless of bracket vs unassigned mix."""
        staves = [
            {"bbox": (100, 600, 2100, 640)},   # bottom solo
            {"bbox": (100, 100, 2100, 140)},   # top bracketed
            {"bbox": (100, 200, 2100, 240)},   # top bracketed
        ]
        brackets = [
            {"x": 50, "y_top": 95, "y_bottom": 245, "height": 150},
        ]
        systems = group_staves_by_brackets(staves, brackets)
        assert len(systems) == 2
        assert systems[0]["bbox"][1] < systems[1]["bbox"][1]
