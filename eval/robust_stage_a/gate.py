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
