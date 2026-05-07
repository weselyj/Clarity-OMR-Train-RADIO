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


from src.data.yolo_aligned_systems import assemble_multi_staff_tokens


def test_assemble_multi_staff_tokens_two_staves():
    staff_0 = ["<bos>", "<staff_start>", "clef-G2", "note-C4", "<staff_end>", "<eos>"]
    staff_1 = ["<bos>", "<staff_start>", "clef-F3", "note-C2", "<staff_end>", "<eos>"]
    out = assemble_multi_staff_tokens([staff_0, staff_1])
    assert out == [
        "<bos>",
        "<staff_start>", "<staff_idx_0>", "clef-G2", "note-C4", "<staff_end>",
        "<staff_start>", "<staff_idx_1>", "clef-F3", "note-C2", "<staff_end>",
        "<eos>",
    ]


def test_assemble_multi_staff_tokens_three_staves():
    s0 = ["<bos>", "<staff_start>", "a", "<staff_end>", "<eos>"]
    s1 = ["<bos>", "<staff_start>", "b", "<staff_end>", "<eos>"]
    s2 = ["<bos>", "<staff_start>", "c", "<staff_end>", "<eos>"]
    out = assemble_multi_staff_tokens([s0, s1, s2])
    assert out == [
        "<bos>",
        "<staff_start>", "<staff_idx_0>", "a", "<staff_end>",
        "<staff_start>", "<staff_idx_1>", "b", "<staff_end>",
        "<staff_start>", "<staff_idx_2>", "c", "<staff_end>",
        "<eos>",
    ]


def test_assemble_multi_staff_tokens_single_staff():
    s = ["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"]
    out = assemble_multi_staff_tokens([s])
    assert out == ["<bos>", "<staff_start>", "<staff_idx_0>", "x", "<staff_end>", "<eos>"]


def test_assemble_multi_staff_tokens_rejects_malformed():
    # Per-staff sequence missing wrapper tokens → assertion error (defensive)
    with pytest.raises(AssertionError):
        assemble_multi_staff_tokens([["clef-G2", "note-C4"]])  # no <bos>/<eos>


def test_assemble_multi_staff_tokens_rejects_too_many_staves():
    # vocab has 8 marker tokens (<staff_idx_0> through <staff_idx_7>); 9th raises
    staves = [["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"]] * 9
    with pytest.raises(ValueError):
        assemble_multi_staff_tokens(staves)


from PIL import Image
from src.data.yolo_aligned_systems import process_page_systems


class _FakeYoloResult:
    """Minimal Ultralytics-result shape that yolo_aligned_crops._yolo_predict_to_boxes accepts."""
    def __init__(self, xyxy_list, conf_list):
        class _Boxes:
            pass
        self.boxes = _Boxes()
        self.boxes.xyxy = xyxy_list
        self.boxes.conf = conf_list


class _FakeYoloModel:
    def __init__(self, predictions: list[tuple[tuple[float, float, float, float], float]]):
        self._predictions = predictions

    def predict(self, image, imgsz=1920, conf=0.25, verbose=False):
        xyxy = [list(box) for box, _ in self._predictions]
        confs = [c for _, c in self._predictions]
        return [_FakeYoloResult(xyxy, confs)]


def test_process_page_systems_end_to_end(tmp_path: Path):
    page = Image.new("RGB", (1000, 1500), color=(255, 255, 255))
    page_path = tmp_path / "page.png"
    page.save(page_path)

    # Write oracle: 2 systems
    txt = tmp_path / "page.txt"
    txt.write_text(
        "0 0.5 0.2 0.9 0.3\n"   # system 0: y_center=0.2, height=0.3 → pixel y in [75, 525], x in [50, 950]
        "0 0.5 0.7 0.9 0.3\n"   # system 1: y_center=0.7
    )
    json_path = tmp_path / "page.staves.json"
    json_path.write_text("[2, 2]")

    # Token lookup: 4 staves total
    def _staff_seq(label):
        return ["<bos>", "<staff_start>", label, "<staff_end>", "<eos>"]
    token_lookup = {
        ("p001", 0): {"page_id": "p001", "staff_index": 0, "style_id": "x", "page_number": 1, "split": "train", "source_path": "src", "source_format": "musicxml", "score_type": "piano", "token_sequence": _staff_seq("a"), "token_count": 5, "dataset": "synthetic_fullpage"},
        ("p001", 1): {"page_id": "p001", "staff_index": 1, "style_id": "x", "page_number": 1, "split": "train", "source_path": "src", "source_format": "musicxml", "score_type": "piano", "token_sequence": _staff_seq("b"), "token_count": 5, "dataset": "synthetic_fullpage"},
        ("p001", 2): {"page_id": "p001", "staff_index": 2, "style_id": "x", "page_number": 1, "split": "train", "source_path": "src", "source_format": "musicxml", "score_type": "piano", "token_sequence": _staff_seq("c"), "token_count": 5, "dataset": "synthetic_fullpage"},
        ("p001", 3): {"page_id": "p001", "staff_index": 3, "style_id": "x", "page_number": 1, "split": "train", "source_path": "src", "source_format": "musicxml", "score_type": "piano", "token_sequence": _staff_seq("d"), "token_count": 5, "dataset": "synthetic_fullpage"},
    }

    # YOLO predictions matching both oracle systems (high IoU)
    yolo_model = _FakeYoloModel([
        ((50, 75, 950, 525), 0.99),
        ((50, 825, 950, 1275), 0.97),
    ])

    crops_dir = tmp_path / "crops"
    entries, report = process_page_systems(
        page_id="p001",
        page_image_path=page_path,
        oracle_label_path=txt,
        oracle_staves_json_path=json_path,
        yolo_model=yolo_model,
        token_lookup=token_lookup,
        out_crops_dir=crops_dir,
        crop_path_template="crops/{filename}",
        iou_threshold=0.5,
    )

    assert report["yolo_boxes"] == 2
    assert report["oracle_systems"] == 2
    assert report["matches"] == 2
    assert len(entries) == 2

    # First entry: system 0, staves 0+1
    e0 = entries[0]
    assert e0["staves_in_system"] == 2
    assert e0["dataset"] == "synthetic_systems"
    assert "<staff_idx_0>" in e0["token_sequence"]
    assert "<staff_idx_1>" in e0["token_sequence"]
    assert e0["token_sequence"][0] == "<bos>"
    assert e0["token_sequence"][-1] == "<eos>"
    # Crop file actually written
    assert (crops_dir / Path(e0["image_path"]).name).exists()


def test_process_page_systems_handles_recall_gap(tmp_path: Path):
    """If YOLO misses a system, that system is dropped from output."""
    page = Image.new("RGB", (1000, 1500))
    page_path = tmp_path / "page.png"
    page.save(page_path)

    txt = tmp_path / "page.txt"
    txt.write_text("0 0.5 0.2 0.9 0.3\n0 0.5 0.7 0.9 0.3\n")
    (tmp_path / "page.staves.json").write_text("[1, 1]")

    yolo_model = _FakeYoloModel([
        ((50, 75, 950, 525), 0.99),  # only system 0
    ])
    token_lookup = {
        ("p", i): {"page_id": "p", "staff_index": i, "style_id": "x", "page_number": 1, "split": "train", "source_path": "s", "source_format": "musicxml", "score_type": "piano", "token_sequence": ["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"], "token_count": 5, "dataset": "synthetic_fullpage"}
        for i in (0, 1)
    }
    entries, report = process_page_systems(
        page_id="p", page_image_path=page_path, oracle_label_path=txt,
        oracle_staves_json_path=tmp_path / "page.staves.json",
        yolo_model=yolo_model, token_lookup=token_lookup,
        out_crops_dir=tmp_path / "crops", crop_path_template="crops/{filename}",
    )
    assert len(entries) == 1
    assert report["dropped_oracle_recall_gap"] == 1


def test_process_page_systems_handles_token_miss(tmp_path: Path):
    """If a matched system has no per-staff tokens for one of its staves, drop the system."""
    page = Image.new("RGB", (1000, 1500))
    page_path = tmp_path / "page.png"
    page.save(page_path)

    (tmp_path / "page.txt").write_text("0 0.5 0.5 0.9 0.5\n")
    (tmp_path / "page.staves.json").write_text("[3]")

    yolo_model = _FakeYoloModel([((50, 375, 950, 1125), 0.99)])
    # Only 2 of 3 staves have token entries
    token_lookup = {
        ("p", i): {"page_id": "p", "staff_index": i, "style_id": "x", "page_number": 1, "split": "train", "source_path": "s", "source_format": "musicxml", "score_type": "piano", "token_sequence": ["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"], "token_count": 5, "dataset": "synthetic_fullpage"}
        for i in (0, 1)  # missing staff 2
    }
    entries, report = process_page_systems(
        page_id="p", page_image_path=page_path, oracle_label_path=tmp_path / "page.txt",
        oracle_staves_json_path=tmp_path / "page.staves.json",
        yolo_model=yolo_model, token_lookup=token_lookup,
        out_crops_dir=tmp_path / "crops", crop_path_template="crops/{filename}",
    )
    assert len(entries) == 0
    assert report["dropped_token_miss"] == 1
