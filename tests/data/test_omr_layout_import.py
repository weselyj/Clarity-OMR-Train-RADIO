"""Unit test for src/data/omr_layout_import.py."""
import json
from pathlib import Path

import pytest


def _write_annotation(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_extracts_only_staves(tmp_path: Path):
    from src.data.omr_layout_import import convert_annotation_to_yolo

    src = tmp_path / "page_001.json"
    _write_annotation(src, {
        "width": 2000,
        "height": 2800,
        "staves": [{"left": 100, "top": 200, "width": 1800, "height": 80}],
        "stave_measures": [{"left": 100, "top": 200, "width": 400, "height": 80}],
        "systems": [{"left": 100, "top": 200, "width": 1800, "height": 200}],
    })

    out = tmp_path / "labels" / "page_001.txt"
    n = convert_annotation_to_yolo(src, out)

    assert n == 1
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0", "Class ID should be 0 (single-class)"
    # x_center = (100 + 1800/2) / 2000 = 0.5
    # y_center = (200 + 80/2) / 2800 = 0.0857
    assert abs(float(parts[1]) - 0.5) < 1e-3
    assert abs(float(parts[2]) - 0.0857) < 1e-3
    assert abs(float(parts[3]) - 0.9) < 1e-3   # 1800/2000
    assert abs(float(parts[4]) - 0.02857) < 1e-3  # 80/2800


def test_skips_pages_with_no_staves(tmp_path: Path):
    """Pages with only systems/measures (no staves) must produce no label file."""
    from src.data.omr_layout_import import convert_annotation_to_yolo

    src = tmp_path / "no_staves.json"
    _write_annotation(src, {
        "width": 2000,
        "height": 2800,
        "staves": [],
        "systems": [{"left": 100, "top": 200, "width": 1800, "height": 200}],
    })

    out = tmp_path / "labels" / "no_staves.txt"
    n = convert_annotation_to_yolo(src, out)

    assert n == 0
    assert not out.exists()


def test_handles_multiple_staves(tmp_path: Path):
    from src.data.omr_layout_import import convert_annotation_to_yolo

    src = tmp_path / "multi.json"
    _write_annotation(src, {
        "width": 1000,
        "height": 1000,
        "staves": [
            {"left": 100, "top": 100, "width": 800, "height": 50},
            {"left": 100, "top": 300, "width": 800, "height": 50},
            {"left": 100, "top": 500, "width": 800, "height": 50},
        ],
    })

    out = tmp_path / "labels" / "multi.txt"
    n = convert_annotation_to_yolo(src, out)

    assert n == 3
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert line.startswith("0 "), "Every line must use class 0"


def test_missing_optional_arrays_ok(tmp_path: Path):
    """Annotation files from sources that omit `systems`/`grand_staff` must work."""
    from src.data.omr_layout_import import convert_annotation_to_yolo

    src = tmp_path / "minimal.json"
    _write_annotation(src, {
        "width": 1000,
        "height": 1000,
        "staves": [{"left": 0, "top": 0, "width": 1000, "height": 50}],
        # No system_measures, stave_measures, systems, or grand_staff arrays at all
    })

    out = tmp_path / "labels" / "minimal.txt"
    n = convert_annotation_to_yolo(src, out)
    assert n == 1
    assert out.exists()


def test_zero_size_bbox_skipped(tmp_path: Path):
    """A pathological zero-size bbox should be skipped, not produce NaN labels."""
    from src.data.omr_layout_import import convert_annotation_to_yolo

    src = tmp_path / "bad.json"
    _write_annotation(src, {
        "width": 1000,
        "height": 1000,
        "staves": [
            {"left": 0, "top": 0, "width": 0, "height": 0},   # bad
            {"left": 100, "top": 100, "width": 800, "height": 50},  # good
        ],
    })

    out = tmp_path / "labels" / "bad.txt"
    n = convert_annotation_to_yolo(src, out)
    assert n == 1, "Only the well-formed bbox should be written"
