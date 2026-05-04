# Stage A: predicted boxes do not learn the v15 +40 px leftward bracket margin

**Status:** known gap as of `feat/system-level-rebuild` after Stage A best.pt (epoch 39, 2026-05-04). Cosmetic only for the current pipeline; flagged for any retrain that needs precise brace coverage.

## Symptom

The v15 system-bbox label algorithm (`src/data/generate_synthetic.py::_build_system_yolo_objects_v15`) applies `LEFTWARD_BRACKET_MARGIN_PX = 40` so the labeled box extends 40 px to the left of the leftmost staff to capture the brace/bracket glyph.

The trained model **does not reproduce this margin**. Predicted `x_left` consistently sits roughly at the leftmost staff rather than 40 px outside it. Visible in `runs/detect/runs/yolo26m_systems/val_batch*_labels.jpg` vs `..._pred.jpg` — labels' left edges sit visibly outside the music; predictions hug the staves.

## Verdict (from Task 13 lieder spot-check, 2026-05-04)

**Cosmetic only.** Across 1,812 detections on 145 lieder PDFs:

- ~70% of detections have the brace's outermost tip outside the predicted box.
- **0 cases** had clipped musical content (notes, clefs, key/time signatures, lyrics, dynamics, vocal staves above a brace).
- Stage B does not need the brace to function — it carries no musical notation.
- mAP50 = 0.99494, mAP50-95 = 0.94304 on val despite the consistent shift; IoU is dominated by the wide-and-tall body of the system.

## Why the model didn't learn the margin

- `imgsz=1920` downsamples from 2750 px source. 40 px source ≈ 28 px model space.
- The 40 px region between brace and leftmost staff is mostly blank paper — no visual signal, no gradient pushing the model to extend the box leftward.
- mAP50 / mAP50-95 are too coarse to penalize a consistent ~1.5%-of-width left-edge shift.

## What to do if a retrain is triggered

Pick whichever fits the retrain context best:

1. **Strong positive signal (recommended for any future retrain):** generate a synthetic mark or watermark inside the leftward bracket margin (a faint corner indicator, dotted border, or anchor glyph at the label's `x_left`) so the model has something to localize on. This is the cleanest single-purpose fix.

2. **Loss-side fix:** add a directional loss term that penalizes asymmetric IoU shrinkage — weight left-edge errors higher than right-edge errors.

3. **Crop augmentation:** randomly crop training images so the brace is sometimes the leftmost visible content. Forces the model to learn that the bbox extends to the brace.

4. **Resolution increase:** train at higher `imgsz` (2880+). Keeps 40 px source > 30 px in model space — more gradient signal.

## Inference-time mitigation (no retrain needed) — IMPLEMENTED

A purpose-built helper restores the design intent at inference time without
modifying model weights:

```python
from src.models.system_postprocess import extend_left_for_brace

# After YOLO predict(), before passing crops to Stage B:
boxes_xyxy = extend_left_for_brace(predictions_xyxy, page_w=image.width)
```

This is what every downstream consumer (eval scripts, end-to-end pipeline,
future Stage B crop builder) should call after running `best.pt`. The helper:

- Defaults the margin to `V15_LEFTWARD_BRACKET_MARGIN_PX` (the same constant
  used at label time), keeping label/inference symmetric.
- Floors the resulting `x_left` at 0 (no negative coords).
- Optionally clamps `x_right` to `page_w` for safety.
- Doesn't mutate the caller's array.

Tests in `tests/models/test_system_postprocess.py` (14 cases). The labels
are correct, the model just under-predicts; this restores the design intent
in 1 line at every inference call site.

## Do NOT change

- The v15 label algorithm. The +40 px margin in labels is correct and approved by visual review (15 iterations in Phase 1).
- The trained `best.pt` weights. They're at the metric gate (mAP50 ≥ 0.95) and validated against lieder eval distribution.

The gap is between what the labels specify and what the model can learn from those labels at the current training setup. Retrain only if a downstream stage emerges that genuinely needs the brace.

## References

- v15 label derivation: `src/data/generate_synthetic.py::_build_system_yolo_objects_v15`
- Margin constant: `V15_LEFTWARD_BRACKET_MARGIN_PX = 40.0`
- Trained weights: `runs/detect/runs/yolo26m_systems/weights/best.pt` (epoch 39)
- Val_batch evidence: `runs/detect/runs/yolo26m_systems/val_batch{0,1,2}_{labels,pred}.jpg`
- Task 13 spot-check overlays: `/tmp/lieder_spotcheck/` (local archive, 2026-05-04)
