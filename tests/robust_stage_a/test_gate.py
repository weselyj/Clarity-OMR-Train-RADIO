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


from eval.robust_stage_a.gate import (  # noqa: E402
    lyric_system_recall,
    combined_gate,
)


def test_lyric_system_recall_counts_only_lyric_systems():
    sc = _sc([
        GtSystem((0, 0, 100, 60), True, [(5, 50, 95, 60)]),    # lyric, detected
        GtSystem((0, 100, 100, 160), True, [(5, 150, 95, 160)]),  # lyric, clipped
        GtSystem((0, 200, 100, 250), False, []),                # non-lyric, ignored
    ])
    preds = [
        Pred((0, 0, 100, 60), 0.9),       # covers sys0 + its lyric band
        Pred((0, 100, 100, 145), 0.9),    # clips sys1 lyric band
        Pred((0, 200, 100, 250), 0.9),
    ]
    # 2 lyric systems, 1 detected cleanly -> 0.5
    assert lyric_system_recall([sc], {"s": preds}) == pytest.approx(0.5)


def test_lyric_system_recall_is_one_when_no_lyric_systems():
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    assert lyric_system_recall([sc], {"s": [Pred((0, 0, 100, 50), 0.9)]}) == 1.0


def test_combined_gate_pass():
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    results = [score_scenario(sc, [Pred((0, 0, 100, 50), 0.9)])]
    v = combined_gate(results, lyric_recall=1.0, lyric_recall_baseline=1.0,
                       lieder_recall=0.94, lieder_baseline=0.93)
    assert v.passed is True
    assert v.n_failed_scenarios == 0


def test_combined_gate_fails_on_one_bad_scenario():
    sc_ok = Scenario("ok", "a", "i", False, [GtSystem((0, 0, 10, 10), False, [])])
    sc_bad = Scenario("bad", "a", "j", True, [])
    results = [
        score_scenario(sc_ok, [Pred((0, 0, 10, 10), 0.9)]),
        score_scenario(sc_bad, [Pred((0, 0, 5, 5), 0.9)]),  # pred on non-music
    ]
    v = combined_gate(results, 1.0, 1.0, 0.94, 0.93)
    assert v.passed is False
    assert v.failed_scenario_ids == ["bad"]


def test_combined_gate_fails_on_lieder_regression():
    sc = _sc([GtSystem((0, 0, 10, 10), False, [])])
    results = [score_scenario(sc, [Pred((0, 0, 10, 10), 0.9)])]
    v = combined_gate(results, 1.0, 1.0, lieder_recall=0.92, lieder_baseline=0.93)
    assert v.passed is False


def test_combined_gate_fails_on_lyric_recall_regression():
    sc = _sc([GtSystem((0, 0, 10, 10), False, [])])
    results = [score_scenario(sc, [Pred((0, 0, 10, 10), 0.9)])]
    v = combined_gate(results, lyric_recall=0.80, lyric_recall_baseline=0.95,
                      lieder_recall=0.94, lieder_baseline=0.93)
    assert v.passed is False


from eval.robust_stage_a.gate import recall_from_stagea_csv  # noqa: E402


def test_recall_from_stagea_csv(tmp_path):
    # mirrors eval/score_stage_a_only.py output schema
    csv = tmp_path / "r.csv"
    csv.write_text(
        "piece,expected_p1_staves,detected_p1_staves,missing_count\n"
        "a,3,3,0\n"
        "b,4,2,2\n",
        encoding="utf-8",
    )
    # total expected 7, detected 5 -> 5/7
    assert recall_from_stagea_csv(csv) == pytest.approx(5 / 7)


def test_recall_from_stagea_csv_empty_is_zero(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("piece,expected_p1_staves,detected_p1_staves,missing_count\n",
                   encoding="utf-8")
    assert recall_from_stagea_csv(csv) == 0.0


from eval.robust_stage_a.gate import verdict_to_report  # noqa: E402


def test_verdict_to_report_pass_and_fail():
    sc = _sc([GtSystem((0, 0, 10, 10), False, [])])
    ok = score_scenario(sc, [Pred((0, 0, 10, 10), 0.9)])
    v = combined_gate([ok], 1.0, 1.0, 0.94, 0.93)
    txt = verdict_to_report(v, [ok])
    assert "GATE: PASS" in txt
    assert "scenarios: 1/1 passed" in txt

    bad = score_scenario(Scenario("bad", "deed", "j", True, []),
                          [Pred((0, 0, 5, 5), 0.9)])
    v2 = combined_gate([bad], 1.0, 1.0, 0.94, 0.93)
    txt2 = verdict_to_report(v2, [bad])
    assert "GATE: FAIL" in txt2
    assert "bad" in txt2 and "deed" in txt2
