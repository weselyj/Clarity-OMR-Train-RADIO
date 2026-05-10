"""Unit tests for src.models.yolo_stage_a_systems."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


def _fake_yolo_results(boxes_xyxy, confs):
    """Build a minimal Ultralytics-result shape that _yolo_predict_to_boxes accepts."""
    class _Boxes:
        pass
    boxes = _Boxes()
    boxes.xyxy = np.array(boxes_xyxy, dtype=np.float32)
    boxes.conf = np.array(confs, dtype=np.float32)
    result = MagicMock()
    result.boxes = boxes
    return [result]


def test_detect_systems_sorts_top_to_bottom_and_extends_left():
    from src.models.yolo_stage_a_systems import YoloStageASystems
    from src.data.generate_synthetic import V15_LEFTWARD_BRACKET_MARGIN_PX

    page = Image.new("RGB", (1000, 800), color="white")
    fake_model = MagicMock()
    # YOLO returns boxes out of order: bottom system first, top system second.
    fake_model.predict.return_value = _fake_yolo_results(
        boxes_xyxy=[[200, 500, 800, 700], [200, 100, 800, 300]],
        confs=[0.9, 0.85],
    )

    with patch("src.models.yolo_stage_a_systems.YOLO", return_value=fake_model):
        wrapper = YoloStageASystems(weights_path="dummy.pt")

    systems = wrapper.detect_systems(page)

    assert len(systems) == 2
    # Sorted top-to-bottom by y_center.
    assert systems[0]["system_index"] == 0
    assert systems[1]["system_index"] == 1
    # Top system bbox came from the second YOLO entry.
    top_x1, top_y1, _, _ = systems[0]["bbox_extended"]
    bot_y1 = systems[1]["bbox_extended"][1]
    assert top_y1 < bot_y1
    # Brace margin extension applied (x1 reduced by margin, clamped to 0).
    expected_top_x1 = max(0, 200 - V15_LEFTWARD_BRACKET_MARGIN_PX)
    assert top_x1 == expected_top_x1


def test_detect_systems_clamps_to_page_bounds():
    from src.models.yolo_stage_a_systems import YoloStageASystems

    page = Image.new("RGB", (500, 400), color="white")
    fake_model = MagicMock()
    # YOLO returns a box whose x1 is already at the left edge.
    fake_model.predict.return_value = _fake_yolo_results(
        boxes_xyxy=[[5, 100, 400, 300]],
        confs=[0.95],
    )

    with patch("src.models.yolo_stage_a_systems.YOLO", return_value=fake_model):
        wrapper = YoloStageASystems(weights_path="dummy.pt")

    systems = wrapper.detect_systems(page)
    assert len(systems) == 1
    x1, _, _, _ = systems[0]["bbox_extended"]
    assert x1 >= 0


def test_detect_systems_empty_page():
    from src.models.yolo_stage_a_systems import YoloStageASystems

    page = Image.new("RGB", (500, 400), color="white")
    fake_model = MagicMock()
    fake_model.predict.return_value = _fake_yolo_results(boxes_xyxy=[], confs=[])

    with patch("src.models.yolo_stage_a_systems.YOLO", return_value=fake_model):
        wrapper = YoloStageASystems(weights_path="dummy.pt")

    assert wrapper.detect_systems(page) == []
