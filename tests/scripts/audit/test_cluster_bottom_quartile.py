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
