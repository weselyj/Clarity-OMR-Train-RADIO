"""Unit tests for spatial-grouping heuristic that derives system bboxes from per-staff bboxes."""
import pytest

from src.data.derive_systems_from_staves import group_staves_into_systems


class TestGroupStavesIntoSystems:
    def test_two_clear_systems(self):
        """Two groups of 3 staves each, well-separated vertically."""
        staves = [
            # System 0
            {"bbox": (100, 200, 2100, 250)},
            {"bbox": (100, 280, 2100, 330)},
            {"bbox": (100, 360, 2100, 410)},
            # Big gap
            # System 1
            {"bbox": (100, 1000, 2100, 1050)},
            {"bbox": (100, 1080, 2100, 1130)},
            {"bbox": (100, 1160, 2100, 1210)},
        ]
        systems = group_staves_into_systems(staves)
        assert len(systems) == 2
        assert systems[0]["staves_in_system"] == 3
        assert systems[0]["bbox"][1] == pytest.approx(200)  # y1 of topmost staff
        assert systems[0]["bbox"][3] == pytest.approx(410)  # y2 of bottommost staff
        assert systems[1]["staves_in_system"] == 3

    def test_single_staff_system(self):
        """One isolated staff = 1-staff system (vocal-only case)."""
        staves = [{"bbox": (100, 200, 2100, 280)}]
        systems = group_staves_into_systems(staves)
        assert len(systems) == 1
        assert systems[0]["staves_in_system"] == 1

    def test_close_staves_grouped(self):
        """Two staves with gap < 1.5x their height are in the same system."""
        staves = [
            {"bbox": (100, 200, 2100, 280)},  # 80 tall
            {"bbox": (100, 350, 2100, 430)},  # 70-pixel gap, 80 tall -> gap < 1.5 * 80 = 120
        ]
        systems = group_staves_into_systems(staves)
        assert len(systems) == 1
        assert systems[0]["staves_in_system"] == 2

    def test_far_staves_split(self):
        """Two staves with gap > 1.5x their height are in separate systems."""
        staves = [
            {"bbox": (100, 200, 2100, 280)},  # 80 tall
            {"bbox": (100, 600, 2100, 680)},  # 320-pixel gap, 80 tall -> gap > 1.5 * 80 = 120
        ]
        systems = group_staves_into_systems(staves)
        assert len(systems) == 2

    def test_x_misaligned_staves_split(self):
        """Staves whose x-extents overlap < 80% are in separate systems (e.g. partial-width inset)."""
        staves = [
            {"bbox": (100, 200, 2100, 280)},   # spans full page width
            {"bbox": (1500, 320, 2100, 400)},  # only right-side ~30% overlap; not in same system
        ]
        systems = group_staves_into_systems(staves)
        assert len(systems) == 2

    def test_unsorted_input_sorts_by_y(self):
        """Input given in non-spatial order — should still produce correct system grouping."""
        staves = [
            {"bbox": (100, 1000, 2100, 1050)},  # bottom system, listed first
            {"bbox": (100, 200, 2100, 250)},    # top system, listed second
            {"bbox": (100, 1080, 2100, 1130)},
            {"bbox": (100, 280, 2100, 330)},
        ]
        systems = group_staves_into_systems(staves)
        assert len(systems) == 2
        # Top system should come first (sorted by y1)
        assert systems[0]["bbox"][1] == pytest.approx(200)
        assert systems[1]["bbox"][1] == pytest.approx(1000)
