# tests/data/test_yolo_aligned_systems.py
from pathlib import Path
import json
import pytest

from src.data.yolo_aligned_systems import (
    load_oracle_system_bboxes,
)


def _write_label_files(tmp_path: Path, txt_lines: list[str], staves_list: list[int]) -> tuple[Path, Path]:
    txt_path = tmp_path / "page.txt"
    json_path = tmp_path / "page.staves.json"
    txt_path.write_text("\n".join(txt_lines) + "\n")
    json_path.write_text(json.dumps(staves_list))
    return txt_path, json_path


def test_load_oracle_system_bboxes_three_systems(tmp_path: Path):
    txt, _ = _write_label_files(
        tmp_path,
        [
            "0 0.5 0.15 0.9 0.25",  # top system
            "0 0.5 0.4 0.99 0.25",  # middle
            "0 0.5 0.65 0.99 0.22",  # bottom
        ],
        [3, 3, 3],
    )
    out = load_oracle_system_bboxes(txt, page_width=1000, page_height=1500)
    assert len(out) == 3
    # Returned in y-sorted order (top→bottom). Check y_centers are ascending.
    y_centers = [(b["bbox"][1] + b["bbox"][3]) / 2 for b in out]
    assert y_centers == sorted(y_centers)
    # system_index assigned by sort order
    assert [b["system_index"] for b in out] == [0, 1, 2]
    # Bbox values are in pixel space
    top_bbox = out[0]["bbox"]
    assert 0 < top_bbox[0] < 1000  # x1
    assert 0 < top_bbox[1] < 1500  # y1
    assert top_bbox[0] < top_bbox[2]  # x1 < x2
    assert top_bbox[1] < top_bbox[3]  # y1 < y2


def test_load_oracle_system_bboxes_empty_label(tmp_path: Path):
    txt = tmp_path / "page.txt"
    txt.write_text("")
    out = load_oracle_system_bboxes(txt, page_width=1000, page_height=1500)
    assert out == []
