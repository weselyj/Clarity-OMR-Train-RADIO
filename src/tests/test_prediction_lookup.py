"""Tests for prediction-key matching (Bug 2: _load_prediction_lookup + run_pipeline).

TDD: these tests were written before the implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


TOKENS_A = ["<bos>", "<staff_start>", "clef-G2", "<staff_end>", "<eos>"]
TOKENS_B = ["<bos>", "<staff_start>", "clef-F4", "<staff_end>", "<eos>"]


# ---------------------------------------------------------------------------
# Test 1: lookup keyed by crop_path filename resolves via crop filename
# ---------------------------------------------------------------------------

def test_lookup_by_crop_filename(tmp_path):
    """_load_prediction_lookup with crop_path keys must resolve by crop filename."""
    from src.cli import _load_prediction_lookup

    pred_file = tmp_path / "preds.jsonl"
    _write_jsonl(pred_file, [
        {"crop_path": "/some/deep/dir/staff_001.png", "tokens": TOKENS_A},
        {"crop_path": "/other/path/staff_002.png", "tokens": TOKENS_B},
    ])

    lookup = _load_prediction_lookup(pred_file)

    # Must resolve by filename
    assert "staff_001.png" in lookup
    assert "staff_002.png" in lookup
    assert lookup["staff_001.png"] == TOKENS_A
    assert lookup["staff_002.png"] == TOKENS_B


# ---------------------------------------------------------------------------
# Test 2: lookup keyed by sample_id resolves via sample_id
# ---------------------------------------------------------------------------

def test_lookup_by_sample_id(tmp_path):
    """_load_prediction_lookup with sample_id-only rows must resolve by sample_id."""
    from src.cli import _load_prediction_lookup

    pred_file = tmp_path / "preds_sid.jsonl"
    _write_jsonl(pred_file, [
        {"sample_id": "staff_001", "tokens": TOKENS_A},
        {"sample_id": "staff_002", "tokens": TOKENS_B},
    ])

    lookup = _load_prediction_lookup(pred_file)

    assert "staff_001" in lookup
    assert "staff_002" in lookup
    assert lookup["staff_001"] == TOKENS_A


# ---------------------------------------------------------------------------
# Test 3: run_pipeline lookup resolves even when predictions use sample_id keys
# ---------------------------------------------------------------------------

def test_run_pipeline_lookup_resolves_sample_id_keys(tmp_path):
    """When the prediction file has sample_id-only keys, run_pipeline must not
    raise 'No token prediction found' when matching via crop filename stem."""
    from src.cli import _load_prediction_lookup

    pred_file = tmp_path / "preds_sid2.jsonl"
    # sample_id matches the stem of the crop file name
    _write_jsonl(pred_file, [
        {"sample_id": "staff_001", "tokens": TOKENS_A},
    ])

    lookup = _load_prediction_lookup(pred_file)

    # run_pipeline tries: crop_filename → crop_stem → sample_id
    # crop filename = "staff_001.png", stem = "staff_001"
    crop_name = "staff_001.png"
    crop_stem = Path(crop_name).stem  # "staff_001"

    tokens = (
        lookup.get(crop_name)
        or lookup.get(crop_stem)
        or lookup.get(crop_stem.replace("_", "-"))
    )
    assert tokens == TOKENS_A, (
        f"Expected lookup to resolve 'staff_001.png' via stem 'staff_001', "
        f"but got: {tokens!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: lookup with crop_path keys resolves via both filename AND stem
# ---------------------------------------------------------------------------

def test_lookup_resolves_via_stem_for_crop_path_keys(tmp_path):
    """When predictions are keyed by crop_path filename, the lookup must also
    expose the stem so that lookups from both directions work."""
    from src.cli import _load_prediction_lookup

    pred_file = tmp_path / "preds_stem.jsonl"
    _write_jsonl(pred_file, [
        {"crop_path": "/deep/path/staff_042.png", "tokens": TOKENS_A},
    ])

    lookup = _load_prediction_lookup(pred_file)

    # Both filename and stem should resolve
    assert lookup.get("staff_042.png") == TOKENS_A
    assert lookup.get("staff_042") == TOKENS_A


# ---------------------------------------------------------------------------
# Test 5: mixed manifest (some rows crop_path, some sample_id)
# ---------------------------------------------------------------------------

def test_lookup_mixed_keys(tmp_path):
    """Mixed manifests with both crop_path rows and sample_id-only rows work."""
    from src.cli import _load_prediction_lookup

    pred_file = tmp_path / "preds_mixed.jsonl"
    _write_jsonl(pred_file, [
        {"crop_path": "/a/b/staff_010.png", "tokens": TOKENS_A},
        {"sample_id": "staff_020", "tokens": TOKENS_B},
    ])

    lookup = _load_prediction_lookup(pred_file)

    assert lookup.get("staff_010.png") == TOKENS_A
    assert lookup.get("staff_010") == TOKENS_A
    assert lookup.get("staff_020") == TOKENS_B
