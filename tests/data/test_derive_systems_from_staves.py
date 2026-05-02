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

    def test_piano_vocal_lieder_5_systems_3_staves_each(self):
        """Schubert lieder D911-04 page 004 layout: 5 systems × 3 staves each.

        Real-world failure mode that motivated bimodal threshold detection:
        intra-system gaps (~30-40 px between vocal/treble/bass) and
        inter-system gaps (~50 px) are too close for any single fixed factor
        to distinguish. The bimodal-detection heuristic finds the natural
        split per-page.
        """
        staves = []
        y = 100.0
        for sys_idx in range(5):
            for staff_idx in range(3):
                staves.append({"bbox": (100.0, y, 2100.0, y + 40.0)})
                y += 40.0 + 30.0  # staff_h=40, intra-gap=30
            y += 20.0  # extra inter-system gap (intra=30, inter=30+20=50)
        systems = group_staves_into_systems(staves)
        assert len(systems) == 5, f"Expected 5 systems, got {len(systems)}"
        for sys_dict in systems:
            assert sys_dict["staves_in_system"] == 3

    def test_satb_choral_with_lyric_gaps_grouped(self):
        """SATB choral page: 4 vocal staves with ~100 px lyric gaps between them.

        Real-world failure mode: a Chorissimo choral page has 4 staves of height
        ~50 px separated by ~100 px gaps (lyrics sit between staves). The old
        heuristic with vertical_gap_factor=1.5 split them into 4 systems; the
        loosened factor=2.5 default groups them into one system (matching the
        bracket grouping a human reader would see).
        """
        staves = [
            # staff_h ≈ 50 px, inter-staff gap ≈ 100 px (within 2.5x staff_h)
            {"bbox": (100, 200, 2100, 250)},
            {"bbox": (100, 350, 2100, 400)},
            {"bbox": (100, 500, 2100, 550)},
            {"bbox": (100, 650, 2100, 700)},
        ]
        systems = group_staves_into_systems(staves)
        assert len(systems) == 1
        assert systems[0]["staves_in_system"] == 4

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
