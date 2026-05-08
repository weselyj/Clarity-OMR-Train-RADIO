"""Inference-time post-processing for Stage A system-level YOLO predictions.

The Stage A model (`yolo26m_systems/best.pt`) was trained on v15 labels that
include a +40 px leftward bracket margin to capture brace/bracket glyphs.
The trained model does not learn this margin (the empty-paper region carries
no visual signal for the loss to grip on), so predicted boxes consistently
clip the brace.

This is cosmetic for downstream Stage B (no musical notation lives in the
brace), but the brace IS musically meaningful as a visual grouping cue and
the v15 design intent was to include it. This module restores the design
intent at inference time without retraining: extend each predicted box's
left edge by the same V15_LEFTWARD_BRACKET_MARGIN_PX used at label time.

Use:

    from src.models.system_postprocess import extend_left_for_brace

    # After YOLO predict(), before passing crops to Stage B:
    boxes_xyxy = extend_left_for_brace(boxes_xyxy, page_w=image.width)

See ``docs/stage_a_brace_margin_known_gap.md`` for the full analysis and
retrain mitigations if/when the model is retrained.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from src.data.generate_synthetic import V15_LEFTWARD_BRACKET_MARGIN_PX

BoxesArrayLike = Union[
    np.ndarray,
    Sequence[Sequence[float]],
    Sequence[Tuple[float, float, float, float]],
]


def extend_left_for_brace(
    boxes_xyxy: BoxesArrayLike,
    page_w: float | int | None = None,
    margin_px: float = V15_LEFTWARD_BRACKET_MARGIN_PX,
) -> np.ndarray:
    """Extend each box's x_left by ``margin_px``, clamped to ``[0, page_w]``.

    Args:
        boxes_xyxy: shape (N, 4) array-like of (x1, y1, x2, y2) in pixel coords.
            Empty input is allowed and returns an empty (0, 4) array.
        page_w: image width in pixels for right-side clamping. If None, no
            upper clamp is applied (boxes are only floored at 0). Right-clamp
            still applies to x2 so a degenerate box can't extend past page_w.
        margin_px: pixels to extend leftward. Defaults to the v15 label
            constant so the prediction matches the design intent.

    Returns:
        ``np.ndarray`` of shape (N, 4) with x1 reduced by ``margin_px`` and
        floored at 0. y1, x2, y2 are unchanged. Always a fresh array — the
        caller's input is not mutated.

    Notes:
        - Function is purely defensive math; no model state, no I/O.
        - Doesn't enforce x1 < x2 — if a caller passes degenerate boxes, they
          stay degenerate. Validation belongs upstream of this helper.
    """
    arr = np.asarray(boxes_xyxy, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"boxes_xyxy must be shape (N, 4); got {arr.shape}"
        )
    out = arr.copy()
    out[:, 0] = np.maximum(0.0, out[:, 0] - float(margin_px))
    if page_w is not None:
        out[:, 2] = np.minimum(float(page_w), out[:, 2])
    return out
