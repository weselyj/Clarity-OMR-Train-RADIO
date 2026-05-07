# tests/data/test_yolo_aligned_systems.py
from pathlib import Path
import json
import pytest

from src.data.yolo_aligned_systems import (
    load_oracle_system_bboxes,
    load_staves_per_system,
    match_yolo_to_oracle_systems,
    staff_indices_for_system,
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


def test_load_staves_per_system(tmp_path: Path):
    p = tmp_path / "page.staves.json"
    p.write_text("[3, 3, 3]")
    assert load_staves_per_system(p) == [3, 3, 3]


def test_load_staves_per_system_missing(tmp_path: Path):
    # Missing file → empty list (the system bbox file is the source of truth)
    p = tmp_path / "missing.staves.json"
    assert load_staves_per_system(p) == []


def test_staff_indices_for_system_uniform():
    # 3 systems with 3 staves each
    assert staff_indices_for_system(0, [3, 3, 3]) == [0, 1, 2]
    assert staff_indices_for_system(1, [3, 3, 3]) == [3, 4, 5]
    assert staff_indices_for_system(2, [3, 3, 3]) == [6, 7, 8]


def test_staff_indices_for_system_varied():
    # Systems with different staff counts: vocal (1) + piano (2) + piano (2)
    assert staff_indices_for_system(0, [1, 2, 2]) == [0]
    assert staff_indices_for_system(1, [1, 2, 2]) == [1, 2]
    assert staff_indices_for_system(2, [1, 2, 2]) == [3, 4]


def test_match_yolo_to_oracle_systems_basic():
    yolo_boxes = [
        {"yolo_idx": 0, "bbox": (10, 10, 100, 50), "conf": 0.99},
        {"yolo_idx": 1, "bbox": (10, 60, 100, 100), "conf": 0.95},
    ]
    oracle = [
        {"system_index": 0, "bbox": (10, 10, 100, 50)},  # exact match for yolo 0
        {"system_index": 1, "bbox": (10, 60, 100, 100)},  # exact match for yolo 1
    ]
    matches = match_yolo_to_oracle_systems(yolo_boxes, oracle, iou_threshold=0.5)
    assert len(matches) == 2
    assert {m["system_index"] for m in matches} == {0, 1}
    assert all(m["iou"] > 0.99 for m in matches)


def test_match_yolo_to_oracle_systems_drops_below_threshold():
    yolo_boxes = [{"yolo_idx": 0, "bbox": (0, 0, 10, 10), "conf": 0.99}]
    oracle = [{"system_index": 0, "bbox": (50, 50, 100, 100)}]  # disjoint
    assert match_yolo_to_oracle_systems(yolo_boxes, oracle, iou_threshold=0.5) == []


def test_match_yolo_to_oracle_systems_keeps_highest_conf_on_dup():
    yolo_boxes = [
        {"yolo_idx": 0, "bbox": (10, 10, 100, 50), "conf": 0.80},
        {"yolo_idx": 1, "bbox": (12, 12, 102, 52), "conf": 0.99},  # same oracle, higher conf
    ]
    oracle = [{"system_index": 0, "bbox": (10, 10, 100, 50)}]
    matches = match_yolo_to_oracle_systems(yolo_boxes, oracle, iou_threshold=0.5)
    assert len(matches) == 1
    assert matches[0]["yolo_idx"] == 1  # higher-conf YOLO box won
