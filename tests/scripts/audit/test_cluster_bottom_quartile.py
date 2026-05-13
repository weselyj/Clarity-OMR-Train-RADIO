"""Test bottom-quartile lieder cluster analysis script.

CPU-only. Asserts cluster-tagging logic on synthetic decoder-output fixtures.
"""
from pathlib import Path
import json
import pytest

from scripts.audit.cluster_bottom_quartile_lieder import (
    classify_piece,
    _extract_staves_from_token_stream,
    _pitch_to_octave,
)


def test_bass_clef_misread_detected():
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2", "median_octave_pred": 4},
                    {"clef_pred": "clef-G2", "median_octave_pred": 4},  # bottom predicted G2
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
    }
    tags = classify_piece(piece_tokens)
    assert "bass-clef-misread" in tags


def test_phantom_staff_residual_detected():
    """3 predicted staves on a 2-staff GT piece should fire phantom-staff-residual."""
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2"}, {"clef_pred": "clef-G2"}, {"clef_pred": "clef-F4"}
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
        "ground_truth_staff_count": 2,
    }
    tags = classify_piece(piece_tokens)
    assert "phantom-staff-residual" in tags


def test_phantom_does_not_fire_on_legitimate_3_staff_lieder():
    """Vocal+piano lieder has 3 legitimate staves; should NOT flag phantom."""
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2", "median_octave_pred": 5},  # vocal
                    {"clef_pred": "clef-G2", "median_octave_pred": 5},  # piano RH
                    {"clef_pred": "clef-F4", "median_octave_pred": 3},  # piano LH
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-G2", "clef-F4"]],
        "ground_truth_staff_count": 3,
    }
    tags = classify_piece(piece_tokens)
    assert "phantom-staff-residual" not in tags


def test_phantom_fires_when_model_exceeds_gt_staff_count():
    """Even on a 3-staff lieder, 4 predicted staves should flag phantom."""
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2"}, {"clef_pred": "clef-G2"},
                    {"clef_pred": "clef-F4"}, {"clef_pred": "clef-F4"},
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-G2", "clef-F4"]],
        "ground_truth_staff_count": 3,
    }
    tags = classify_piece(piece_tokens)
    assert "phantom-staff-residual" in tags


def test_clean_piece_has_other_tag_only():
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2", "median_octave_pred": 5},
                    {"clef_pred": "clef-F4", "median_octave_pred": 2},
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
        "ground_truth_staff_count": 2,
    }
    tags = classify_piece(piece_tokens)
    assert tags == ["other"]


def test_key_time_signature_residual_detected_for_non_4_4():
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2", "median_octave_pred": 5},
                    {"clef_pred": "clef-F4", "median_octave_pred": 2},
                ],
                "time_sig_pred": "timeSignature-4/4",
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
        "ground_truth_staff_count": 2,
        "ground_truth_time_sig": "timeSignature-3/4",
    }
    tags = classify_piece(piece_tokens)
    assert "key-time-sig-residual" in tags


def test_key_time_signature_residual_via_per_staff_pred():
    """When time_sig_pred is on a staff (as _extract_staves_from_token_stream emits)
    rather than at the system level, classify_piece should still detect mismatch."""
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {
                        "clef_pred": "clef-G2",
                        "median_octave_pred": 5,
                        "time_sig_pred": "timeSignature-4/4",
                    },
                    {
                        "clef_pred": "clef-F4",
                        "median_octave_pred": 2,
                        "time_sig_pred": None,
                    },
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
        "ground_truth_staff_count": 2,
        "ground_truth_time_sig": "timeSignature-3/4",
    }
    tags = classify_piece(piece_tokens)
    assert "key-time-sig-residual" in tags


def test_bass_clef_misread_detected_on_multi_system_piece():
    """Multi-system piece: GT clefs at sys 0 should apply to sys 1+ via fallback."""
    piece_tokens = {
        "systems": [
            # System 0 — model gets it right
            {"staves": [
                {"clef_pred": "clef-G2", "median_octave_pred": 5},
                {"clef_pred": "clef-F4", "median_octave_pred": 2},
            ]},
            # System 1 — model misreads bass as treble
            {"staves": [
                {"clef_pred": "clef-G2", "median_octave_pred": 4},
                {"clef_pred": "clef-G2", "median_octave_pred": 4},
            ]},
        ],
        # GT only has system 0's clefs; system 1 must fall back to system 0
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
        "ground_truth_staff_count": 2,
    }
    tags = classify_piece(piece_tokens)
    assert "bass-clef-misread" in tags


def test_empty_ground_truth_does_not_crash():
    """If GT extraction returned nothing, classify_piece should not raise."""
    piece_tokens = {
        "systems": [
            {"staves": [
                {"clef_pred": "clef-G2", "median_octave_pred": 4},
                {"clef_pred": "clef-G2", "median_octave_pred": 4},
            ]},
        ],
        "ground_truth_clefs_by_system": [],
    }
    tags = classify_piece(piece_tokens)
    # Without GT we can't classify bass-clef-misread; should fall through to "other"
    assert tags == ["other"]


def test_extract_staves_single_staff_no_idx_markers():
    """Single-staff system: decoder emits <staff_start>/<staff_end> only, no idx marker."""
    tokens = [
        "<bos>",
        "<staff_start>",
        "clef-G2",
        "keySignature-CM",
        "timeSignature-4/4",
        "<measure_start>",
        "note-C4", "_quarter",
        "note-E4", "_quarter",
        "<measure_end>",
        "<staff_end>",
        "<eos>",
    ]
    staves = _extract_staves_from_token_stream(tokens)
    assert len(staves) == 1
    assert staves[0]["clef_pred"] == "clef-G2"
    assert staves[0]["time_sig_pred"] == "timeSignature-4/4"
    assert staves[0]["key_sig_pred"] == "keySignature-CM"
    assert staves[0]["median_octave_pred"] == 4


def test_extract_staves_multi_staff_with_idx_markers():
    """Multi-staff system: each <staff_start> followed by <staff_idx_N>."""
    tokens = [
        "<bos>",
        "<staff_start>", "<staff_idx_0>",
        "clef-G2", "timeSignature-3/4",
        "note-C5", "_quarter", "note-E5", "_quarter",
        "<staff_end>",
        "<staff_start>", "<staff_idx_1>",
        "clef-F4", "timeSignature-3/4",
        "note-C3", "_quarter", "note-G2", "_quarter",
        "<staff_end>",
        "<eos>",
    ]
    staves = _extract_staves_from_token_stream(tokens)
    assert len(staves) == 2
    assert staves[0]["clef_pred"] == "clef-G2"
    assert staves[0]["time_sig_pred"] == "timeSignature-3/4"
    assert staves[0]["median_octave_pred"] == 5
    assert staves[1]["clef_pred"] == "clef-F4"
    # median of [3, 2] (sorted: [2,3], len//2=1) -> 3
    assert staves[1]["median_octave_pred"] == 3


def test_extract_staves_handles_bos_prefix_and_trailing_tokens():
    """<bos> before the first <staff_start> and stray tokens after should be ignored."""
    tokens = [
        "<bos>",
        "<staff_start>",
        "clef-G2",
        "note-C4",
        "<staff_end>",
        "<eos>",
        "<pad>",
    ]
    staves = _extract_staves_from_token_stream(tokens)
    assert len(staves) == 1
    assert staves[0]["clef_pred"] == "clef-G2"


def test_extract_staves_three_staff_lieder():
    """Vocal + piano (RH + LH) — three staves in one system."""
    tokens = [
        "<bos>",
        "<staff_start>", "<staff_idx_0>", "clef-G2", "note-G4", "<staff_end>",
        "<staff_start>", "<staff_idx_1>", "clef-G2", "note-E5", "<staff_end>",
        "<staff_start>", "<staff_idx_2>", "clef-F4", "note-C3", "<staff_end>",
        "<eos>",
    ]
    staves = _extract_staves_from_token_stream(tokens)
    assert len(staves) == 3
    assert [s["clef_pred"] for s in staves] == ["clef-G2", "clef-G2", "clef-F4"]


def test_pitch_to_octave_handles_double_accidentals():
    """note-Bbb3 (double-flat) and note-F##5 (double-sharp) should parse."""
    assert _pitch_to_octave("note-C4") == 4
    assert _pitch_to_octave("note-Eb5") == 5
    assert _pitch_to_octave("note-Bbb3") == 3
    assert _pitch_to_octave("note-F##5") == 5
    assert _pitch_to_octave("note-C#6") == 6
    assert _pitch_to_octave("not-a-pitch-token") is None
