"""Tests for src.models.system_postprocess."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.generate_synthetic import V15_LEFTWARD_BRACKET_MARGIN_PX
from src.models.system_postprocess import extend_left_for_brace


class TestExtendLeftForBrace:
    def test_default_margin_matches_v15_label_constant(self):
        """Inference-time extension MUST use the same constant as label-time."""
        boxes = np.array([[100.0, 50.0, 200.0, 150.0]])
        out = extend_left_for_brace(boxes, page_w=300)
        # Default margin == V15 label constant (currently 40).
        assert out[0, 0] == pytest.approx(100.0 - V15_LEFTWARD_BRACKET_MARGIN_PX)

    def test_y_and_right_edge_unchanged(self):
        boxes = np.array([[100.0, 50.0, 200.0, 150.0]])
        out = extend_left_for_brace(boxes, page_w=300)
        assert out[0, 1] == 50.0
        assert out[0, 2] == 200.0
        assert out[0, 3] == 150.0

    def test_x_left_clamped_to_zero(self):
        """Boxes already near the left edge can't go negative."""
        boxes = np.array([[10.0, 50.0, 200.0, 150.0]])  # x1=10, margin=40 -> -30 -> 0
        out = extend_left_for_brace(boxes, page_w=300)
        assert out[0, 0] == 0.0

    def test_explicit_margin_override(self):
        boxes = np.array([[100.0, 50.0, 200.0, 150.0]])
        out = extend_left_for_brace(boxes, page_w=300, margin_px=15)
        assert out[0, 0] == pytest.approx(85.0)

    def test_zero_margin_is_identity(self):
        boxes = np.array([[100.0, 50.0, 200.0, 150.0]])
        out = extend_left_for_brace(boxes, page_w=300, margin_px=0)
        np.testing.assert_array_equal(out, boxes)

    def test_multiple_boxes_independently_extended(self):
        boxes = np.array(
            [
                [100.0, 50.0, 200.0, 150.0],
                [500.0, 300.0, 800.0, 400.0],
                [20.0, 600.0, 250.0, 700.0],  # near edge -> floor at 0
            ]
        )
        out = extend_left_for_brace(boxes, page_w=900)
        assert out[0, 0] == pytest.approx(60.0)
        assert out[1, 0] == pytest.approx(460.0)
        assert out[2, 0] == 0.0

    def test_input_not_mutated(self):
        """Caller's array must be untouched."""
        boxes = np.array([[100.0, 50.0, 200.0, 150.0]])
        original = boxes.copy()
        _ = extend_left_for_brace(boxes, page_w=300)
        np.testing.assert_array_equal(boxes, original)

    def test_accepts_list_of_tuples(self):
        boxes = [(100.0, 50.0, 200.0, 150.0), (300.0, 100.0, 400.0, 200.0)]
        out = extend_left_for_brace(boxes, page_w=500)
        assert out.shape == (2, 4)
        assert out[0, 0] == pytest.approx(60.0)
        assert out[1, 0] == pytest.approx(260.0)

    def test_empty_input_returns_empty_2d(self):
        out = extend_left_for_brace([], page_w=300)
        assert out.shape == (0, 4)

    def test_empty_ndarray_returns_empty_2d(self):
        out = extend_left_for_brace(np.array([]), page_w=300)
        assert out.shape == (0, 4)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="must be shape"):
            extend_left_for_brace(np.array([[1.0, 2.0, 3.0]]), page_w=300)

    def test_page_w_optional_skips_right_clamp(self):
        """If page_w is None, x_right is not clamped."""
        boxes = np.array([[100.0, 50.0, 9999.0, 150.0]])
        out = extend_left_for_brace(boxes, page_w=None)
        assert out[0, 2] == 9999.0

    def test_page_w_clamps_x_right(self):
        """x_right is clamped to page_w to prevent out-of-bounds boxes."""
        boxes = np.array([[100.0, 50.0, 9999.0, 150.0]])
        out = extend_left_for_brace(boxes, page_w=2750)
        assert out[0, 2] == 2750.0

    def test_realistic_lieder_page(self):
        """End-to-end check on a realistic 3-system piano-vocal page."""
        # 3 systems on a 2750x3750 page (production DPI), each ~2700 wide.
        # These are 'predicted' boxes hugging the staves; brace extension
        # should pull x_left from ~175 to ~135.
        page_w, page_h = 2750, 3750
        predicted = np.array(
            [
                [175.0, 200.0, 2625.0, 900.0],
                [175.0, 1100.0, 2625.0, 1850.0],
                [175.0, 2150.0, 2625.0, 3000.0],
            ]
        )
        out = extend_left_for_brace(predicted, page_w=page_w)
        assert out.shape == (3, 4)
        for i in range(3):
            # x_left extended by 40 -> 135
            assert out[i, 0] == pytest.approx(135.0)
            # other coords unchanged
            assert out[i, 1] == predicted[i, 1]
            assert out[i, 2] == predicted[i, 2]
            assert out[i, 3] == predicted[i, 3]
