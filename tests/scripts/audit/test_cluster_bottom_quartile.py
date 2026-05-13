"""Test bottom-quartile lieder cluster analysis script.

CPU-only. Asserts cluster-tagging logic on synthetic decoder-output fixtures.
"""
from pathlib import Path
import json
import pytest

from scripts.audit.cluster_bottom_quartile_lieder import classify_piece


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
    piece_tokens = {
        "systems": [
            {
                "staves": [
                    {"clef_pred": "clef-G2"}, {"clef_pred": "clef-G2"}, {"clef_pred": "clef-F4"}
                ],
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
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
                "time_sig_pred": "time-4/4",
            }
        ],
        "ground_truth_clefs_by_system": [["clef-G2", "clef-F4"]],
        "ground_truth_time_sig": "time-3/4",
    }
    tags = classify_piece(piece_tokens)
    assert "key-time-sig-residual" in tags
