"""Unit tests for assemble_score_from_system_predictions."""
from __future__ import annotations


def _make_grandstaff_system_tokens():
    """Two-staff (piano grand-staff) system with minimal note content."""
    return [
        "<bos>",
        "<staff_start>", "<staff_idx_0>",
        "<measure_start>", "note-C4-quarter", "<measure_end>", "<staff_end>",
        "<staff_start>", "<staff_idx_1>",
        "<measure_start>", "note-C3-quarter", "<measure_end>", "<staff_end>",
        "<eos>",
    ]


def test_single_system_two_staves_yields_two_staves_in_assembled_score():
    from src.pipeline.assemble_score import assemble_score_from_system_predictions

    sys_tokens = _make_grandstaff_system_tokens()
    sys_loc = {
        "system_index": 0,
        "bbox": (50.0, 100.0, 950.0, 300.0),
        "page_index": 0,
        "conf": 0.95,
    }

    score = assemble_score_from_system_predictions([sys_tokens], [sys_loc])

    assert len(score.systems) == 1
    assert len(score.systems[0].staves) == 2


def test_even_split_y_coords_within_a_system():
    """Each staff's StaffLocation y-range is the system bbox split by N=2."""
    from src.pipeline.assemble_score import assemble_score_from_system_predictions

    sys_tokens = _make_grandstaff_system_tokens()
    sys_loc = {
        "system_index": 3,
        "bbox": (0.0, 100.0, 1000.0, 300.0),
        "page_index": 0,
        "conf": 0.9,
    }
    score = assemble_score_from_system_predictions([sys_tokens], [sys_loc])

    staves = score.systems[0].staves
    assert staves[0].location.y_top == 100.0
    assert staves[0].location.y_bottom == 200.0
    assert staves[1].location.y_top == 200.0
    assert staves[1].location.y_bottom == 300.0
    assert staves[0].location.x_left == 0.0
    assert staves[0].location.x_right == 1000.0


def test_system_index_hint_groups_correctly_and_preserves_page_index():
    """The system_index_hint flows into group_staves_into_systems for grouping;
    the hint itself is consumed there and the resulting AssembledSystem appears
    on the correct page."""
    from src.pipeline.assemble_score import assemble_score_from_system_predictions

    sys_tokens = _make_grandstaff_system_tokens()
    sys_loc = {
        "system_index": 7,
        "bbox": (0.0, 0.0, 100.0, 100.0),
        "page_index": 2,
        "conf": 0.5,
    }
    score = assemble_score_from_system_predictions([sys_tokens], [sys_loc])

    assert len(score.systems) == 1
    assert score.systems[0].page_index == 2
    for staff in score.systems[0].staves:
        assert staff.location.page_index == 2


def test_empty_system_token_list_yields_empty_score():
    from src.pipeline.assemble_score import assemble_score_from_system_predictions

    score = assemble_score_from_system_predictions([], [])
    assert len(score.systems) == 0
