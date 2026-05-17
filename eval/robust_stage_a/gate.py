"""Pure Stage-A strict-gate scoring. No torch / no YOLO — CPU, deterministic.

Coordinate convention: Box = (x1, y1, x2, y2) in pixels, x2>x1, y2>y1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Pred:
    box: Box
    conf: float


@dataclass(frozen=True)
class MatchResult:
    matched: list[tuple[int, int]]  # (gt_idx, pred_idx)
    missed_gt: list[int]            # gt indices with no matching pred
    false_pred: list[int]          # pred indices matching no gt


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def contains(outer: Box, inner: Box, tol: float = 2.0) -> bool:
    """True if `inner` lies inside `outer`, allowing `tol` px slack per side."""
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] <= outer[2] + tol
        and inner[3] <= outer[3] + tol
    )


def match_predictions(
    gt_boxes: list[Box], preds: list[Pred], match_iou: float = 0.5
) -> MatchResult:
    """Greedy match: highest-confidence preds first, each to the best
    still-unmatched GT with IoU >= match_iou. One pred ↔ at most one GT."""
    order = sorted(range(len(preds)), key=lambda i: preds[i].conf, reverse=True)
    used_gt: set[int] = set()
    matched: list[tuple[int, int]] = []
    false_pred: list[int] = []
    for pi in order:
        best_gi, best_iou = -1, match_iou - 1e-9
        for gi, gb in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            v = iou(gb, preds[pi].box)
            if v >= match_iou and v > best_iou:
                best_gi, best_iou = gi, v
        if best_gi >= 0:
            used_gt.add(best_gi)
            matched.append((best_gi, pi))
        else:
            false_pred.append(pi)
    matched.sort()
    missed_gt = [gi for gi in range(len(gt_boxes)) if gi not in used_gt]
    return MatchResult(matched=matched, missed_gt=missed_gt,
                       false_pred=sorted(false_pred))


from eval.robust_stage_a.manifest import Scenario  # noqa: E402


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    archetype: str
    false: int
    missed: int
    geometry_fail: int
    lyric_clip: int
    passed: bool


def _geometry_ok(gt, pred_box: Box, tol: float) -> tuple[bool, bool]:
    """Returns (geometry_ok, lyric_clipped). geometry_ok requires the pred to
    contain the GT system box and every lyric band; a missed lyric band sets
    lyric_clipped True (and geometry_ok False)."""
    sys_ok = contains(pred_box, gt.box, tol)
    lyric_clipped = any(
        not contains(pred_box, lb, tol) for lb in gt.lyric_bands
    )
    return (sys_ok and not lyric_clipped), lyric_clipped


def score_scenario(
    scenario: Scenario,
    preds: list[Pred],
    match_iou: float = 0.5,
    contain_tol: float = 2.0,
) -> ScenarioResult:
    if scenario.is_non_music:
        n_false = len(preds)
        return ScenarioResult(
            scenario.scenario_id, scenario.archetype,
            false=n_false, missed=0, geometry_fail=0, lyric_clip=0,
            passed=(n_false == 0),
        )

    gt_boxes = [g.box for g in scenario.gt_systems]
    m = match_predictions(gt_boxes, preds, match_iou=match_iou)
    geometry_fail = 0
    lyric_clip = 0
    for gi, pi in m.matched:
        ok, clipped = _geometry_ok(
            scenario.gt_systems[gi], preds[pi].box, contain_tol)
        if not ok:
            geometry_fail += 1
        if clipped:
            lyric_clip += 1
    false = len(m.false_pred)
    missed = len(m.missed_gt)
    passed = (false == 0 and missed == 0 and geometry_fail == 0)
    return ScenarioResult(
        scenario.scenario_id, scenario.archetype,
        false=false, missed=missed, geometry_fail=geometry_fail,
        lyric_clip=lyric_clip, passed=passed,
    )


NO_REGRESSION_EPS = 0.0  # strict: new must be >= baseline (per spec)


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    n_scenarios: int
    n_failed_scenarios: int
    failed_scenario_ids: list[str]
    lyric_recall: float
    lyric_recall_baseline: float
    lieder_recall: float
    lieder_baseline: float


def lyric_system_recall(
    scenarios: list[Scenario],
    preds_by_scenario: dict[str, list[Pred]],
    match_iou: float = 0.5,
    contain_tol: float = 2.0,
) -> float:
    """Fraction of GT systems that carry lyrics which are detected with a
    matching pred whose box also fully contains the lyric band(s).
    1.0 when there are no lyric-bearing GT systems."""
    total = 0
    detected = 0
    for sc in scenarios:
        if sc.is_non_music:
            continue
        preds = preds_by_scenario.get(sc.scenario_id, [])
        m = match_predictions([g.box for g in sc.gt_systems], preds,
                              match_iou=match_iou)
        matched_pred_for_gt = {gi: pi for gi, pi in m.matched}
        for gi, g in enumerate(sc.gt_systems):
            if not g.has_lyrics:
                continue
            total += 1
            pi = matched_pred_for_gt.get(gi)
            if pi is None:
                continue
            ok, _ = _geometry_ok(g, preds[pi].box, contain_tol)
            if ok:
                detected += 1
    return 1.0 if total == 0 else detected / total


def combined_gate(
    scenario_results: list[ScenarioResult],
    lyric_recall: float,
    lyric_recall_baseline: float,
    lieder_recall: float,
    lieder_baseline: float,
    eps: float = NO_REGRESSION_EPS,
) -> GateVerdict:
    failed = [r.scenario_id for r in scenario_results if not r.passed]
    all_scenarios_pass = not failed
    no_lieder_regression = lieder_recall >= lieder_baseline - eps
    no_lyric_regression = lyric_recall >= lyric_recall_baseline - eps
    return GateVerdict(
        passed=(all_scenarios_pass and no_lieder_regression
                and no_lyric_regression),
        n_scenarios=len(scenario_results),
        n_failed_scenarios=len(failed),
        failed_scenario_ids=failed,
        lyric_recall=lyric_recall,
        lyric_recall_baseline=lyric_recall_baseline,
        lieder_recall=lieder_recall,
        lieder_baseline=lieder_baseline,
    )


import csv as _csv  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def recall_from_stagea_csv(path: str | _Path) -> float:
    """Aggregate recall from an eval/score_stage_a_only.py CSV:
    sum(detected_p1_staves) / sum(expected_p1_staves). 0.0 if no rows."""
    total_expected = 0
    total_detected = 0
    with _Path(path).open(newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            total_expected += int(row["expected_p1_staves"])
            total_detected += int(row["detected_p1_staves"])
    return (total_detected / total_expected) if total_expected else 0.0


def verdict_to_report(
    verdict: GateVerdict, scenario_results: list[ScenarioResult]
) -> str:
    lines = []
    lines.append(f"GATE: {'PASS' if verdict.passed else 'FAIL'}")
    lines.append(
        f"scenarios: {verdict.n_scenarios - verdict.n_failed_scenarios}"
        f"/{verdict.n_scenarios} passed"
    )
    lines.append(
        f"lieder_recall: {verdict.lieder_recall:.4f} "
        f"(baseline {verdict.lieder_baseline:.4f}, "
        f"{'OK' if verdict.lieder_recall >= verdict.lieder_baseline else 'REGRESSION'})"
    )
    lines.append(
        f"lyric_system_recall: {verdict.lyric_recall:.4f} "
        f"(baseline {verdict.lyric_recall_baseline:.4f}, "
        f"{'OK' if verdict.lyric_recall >= verdict.lyric_recall_baseline else 'REGRESSION'})"
    )
    for r in scenario_results:
        flag = "PASS" if r.passed else "FAIL"
        lines.append(
            f"  [{flag}] {r.scenario_id} ({r.archetype}) "
            f"false={r.false} missed={r.missed} "
            f"geom={r.geometry_fail} lyric_clip={r.lyric_clip}"
        )
    return "\n".join(lines)
