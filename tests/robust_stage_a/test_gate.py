"""CPU unit tests for the pure Stage-A strict-gate scoring logic."""
import pytest

from eval.robust_stage_a.gate import Pred, iou, contains, match_predictions


def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # a=100 area, b=100 area, intersection 50 -> 50/150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_contains_true_within_tol():
    assert contains((0, 0, 100, 100), (1, 1, 99, 99), tol=2.0) is True
    # inner pokes 1px outside but within 2px tol
    assert contains((0, 0, 100, 100), (-1, 0, 100, 100), tol=2.0) is True


def test_contains_false_outside_tol():
    assert contains((0, 0, 100, 100), (-5, 0, 100, 100), tol=2.0) is False


def test_match_greedy_by_confidence():
    gt = [(0, 0, 10, 10), (100, 100, 110, 110)]
    preds = [
        Pred(box=(0, 0, 10, 10), conf=0.9),       # matches gt0
        Pred(box=(100, 100, 110, 110), conf=0.8),  # matches gt1
        Pred(box=(200, 200, 210, 210), conf=0.7),  # false
    ]
    m = match_predictions(gt, preds, match_iou=0.5)
    assert m.matched == [(0, 0), (1, 1)]  # (gt_idx, pred_idx)
    assert m.missed_gt == []
    assert m.false_pred == [2]


def test_match_missed_when_below_iou():
    gt = [(0, 0, 10, 10)]
    preds = [Pred(box=(7, 0, 17, 10), conf=0.9)]  # iou 30/170 < 0.5
    m = match_predictions(gt, preds, match_iou=0.5)
    assert m.matched == []
    assert m.missed_gt == [0]
    assert m.false_pred == [0]


def test_match_one_pred_cannot_take_two_gt():
    gt = [(0, 0, 10, 10), (0, 0, 10, 10)]
    preds = [Pred(box=(0, 0, 10, 10), conf=0.9)]
    m = match_predictions(gt, preds, match_iou=0.5)
    assert len(m.matched) == 1
    assert m.missed_gt == [1]
    assert m.false_pred == []
