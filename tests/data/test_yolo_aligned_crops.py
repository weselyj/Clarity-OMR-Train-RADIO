"""Unit tests for src.data.yolo_aligned_crops."""
import pytest

from src.data.yolo_aligned_crops import iou_xyxy, match_yolo_to_oracle


class TestIoU:
    def test_identical_boxes_iou_1(self):
        assert iou_xyxy((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_0(self):
        assert iou_xyxy((0, 0, 100, 100), (200, 200, 300, 300)) == 0.0

    def test_half_overlap_iou(self):
        # box B is x:[50,150] y:[0,100]; intersection 50x100=5000; union 15000
        assert iou_xyxy((0, 0, 100, 100), (50, 0, 150, 100)) == pytest.approx(5000 / 15000)

    def test_zero_area_box_iou_0(self):
        assert iou_xyxy((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0


class TestMatchYoloToOracle:
    def test_one_to_one_match(self):
        # 2 oracle staves vertically stacked, 2 YOLO bboxes near them
        oracles = [
            {"staff_index": 0, "bbox": (10, 10, 200, 50)},
            {"staff_index": 1, "bbox": (10, 100, 200, 140)},
        ]
        yolo = [
            {"yolo_idx": 0, "bbox": (12, 11, 202, 49), "conf": 0.95},
            {"yolo_idx": 1, "bbox": (10, 101, 199, 140), "conf": 0.90},
        ]
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 2
        assert matches[0]["yolo_idx"] == 0 and matches[0]["staff_index"] == 0
        assert matches[1]["yolo_idx"] == 1 and matches[1]["staff_index"] == 1

    def test_yolo_false_positive_dropped(self):
        oracles = [{"staff_index": 0, "bbox": (10, 10, 200, 50)}]
        yolo = [
            {"yolo_idx": 0, "bbox": (10, 10, 200, 50), "conf": 0.95},  # match
            {"yolo_idx": 1, "bbox": (500, 500, 600, 600), "conf": 0.80},  # no overlap
        ]
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0]["yolo_idx"] == 0

    def test_oracle_with_no_yolo_match_dropped(self):
        oracles = [
            {"staff_index": 0, "bbox": (10, 10, 200, 50)},
            {"staff_index": 1, "bbox": (10, 100, 200, 140)},  # YOLO misses this
        ]
        yolo = [{"yolo_idx": 0, "bbox": (10, 10, 200, 50), "conf": 0.95}]
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0]["staff_index"] == 0
        # Oracle index 1 silently dropped per α-policy

    def test_two_yolo_hit_one_oracle_keep_highest_conf(self):
        oracles = [{"staff_index": 0, "bbox": (10, 10, 200, 50)}]
        yolo = [
            {"yolo_idx": 0, "bbox": (10, 10, 200, 50), "conf": 0.85},
            {"yolo_idx": 1, "bbox": (12, 11, 198, 49), "conf": 0.95},  # higher conf wins
        ]
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0]["yolo_idx"] == 1
        assert matches[0]["staff_index"] == 0

    def test_below_threshold_dropped(self):
        oracles = [{"staff_index": 0, "bbox": (0, 0, 100, 100)}]
        yolo = [{"yolo_idx": 0, "bbox": (50, 50, 150, 150), "conf": 0.95}]  # IoU = 1/7
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 0
