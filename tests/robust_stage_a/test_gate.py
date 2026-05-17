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


from eval.robust_stage_a.gate import score_scenario  # noqa: E402
from eval.robust_stage_a.manifest import GtSystem, Scenario  # noqa: E402


def _sc(gt_systems, is_non_music=False):
    return Scenario("s", "arch", "img.png", is_non_music, gt_systems)


def test_scenario_perfect_passes():
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    preds = [Pred((0, 0, 100, 50), 0.9)]
    r = score_scenario(sc, preds)
    assert (r.false, r.missed, r.geometry_fail, r.lyric_clip) == (0, 0, 0, 0)
    assert r.passed is True


def test_scenario_false_system_fails():
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    preds = [Pred((0, 0, 100, 50), 0.9), Pred((300, 300, 400, 350), 0.8)]
    r = score_scenario(sc, preds)
    assert r.false == 1 and r.passed is False


def test_scenario_missed_system_fails():
    sc = _sc([GtSystem((0, 0, 100, 50), False, []),
              GtSystem((0, 100, 100, 150), False, [])])
    preds = [Pred((0, 0, 100, 50), 0.9)]
    r = score_scenario(sc, preds)
    assert r.missed == 1 and r.passed is False


def test_scenario_geometry_fail_when_pred_does_not_cover_gt():
    # pred matches by IoU>=0.5 but does not contain the full GT system
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    preds = [Pred((0, 0, 100, 40), 0.9)]  # iou 4000/5000=0.8, but clips bottom
    r = score_scenario(sc, preds)
    assert r.geometry_fail == 1 and r.lyric_clip == 0 and r.passed is False


def test_scenario_lyric_clip_counts_as_geometry_and_lyricclip():
    sc = _sc([GtSystem((0, 0, 100, 60), True, [(5, 50, 95, 60)])])
    # pred covers the staff but cuts off the lyric band (y stops at 45)
    preds = [Pred((0, 0, 100, 45), 0.95)]
    r = score_scenario(sc, preds)
    assert r.geometry_fail == 1 and r.lyric_clip == 1 and r.passed is False


def test_non_music_scenario_passes_with_zero_preds():
    sc = _sc([], is_non_music=True)
    assert score_scenario(sc, []).passed is True


def test_non_music_scenario_any_pred_is_false():
    sc = _sc([], is_non_music=True)
    r = score_scenario(sc, [Pred((0, 0, 10, 10), 0.99)])
    assert r.false == 1 and r.passed is False
