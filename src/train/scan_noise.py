"""Scan-noise augmentation pipeline for YOLO training.

Patches ultralytics' default Albumentations transform list with a custom
scan-noise pipeline (JPEG compression, sensor noise, blur, brightness/contrast,
small rotation, grid + elastic distortion). Used by ``scripts/train_yolo.py``
behind the ``--noise`` flag.

Supports an optional linear warmup ramp on the augmentation intensity:
``intensity_for_step(step, warmup_steps)`` ramps from 0 (no augmentation)
to 1 (full strength) over the first ``warmup_steps`` training batches.
This avoids the early-training failure mode where ``cls_loss`` blows up
on a fresh detection head fed heavily-augmented inputs (the failure that
motivated PR #38's two-phase clean→noise curriculum).
"""
from __future__ import annotations

import random
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Image-only erosion transform (safe for detection pipelines with bboxes)
# ---------------------------------------------------------------------------
# A.Morphological(operation="erosion") in Albumentations 2.0.8 applies the
# morphological op to bboxes via cv2.erode, which aborts with:
#   cv2.error (-215): dims <= 2 && step[0] > 0 in cv2.Mat.locateROI
# when bounding-box coordinates are passed to it.
#
# Fix: subclass A.ImageOnlyTransform, which Albumentations guarantees is
# NEVER applied to bboxes, masks, or keypoints — only to the image array.
# This preserves the stroke-thinning ("faint/broken ink") intent of the
# original transform while being inherently safe in a YOLO detection pipeline.
#
# The class is defined via a soft import so the module stays importable on
# hosts where albumentations is not installed (e.g. pure-Python CI runners).
# A.ImageOnlyTransform is the documented base for image-only operations and
# is available in all Albumentations versions >= 1.x.

try:
    import albumentations as _A
    import cv2 as _cv2

    class ImageOnlyErosion(_A.ImageOnlyTransform):
        """Apply cv2.erode to the image only; never touches bboxes/masks/keypoints.

        Args:
            scale: Tuple ``(min_px, max_px)`` for the square erosion kernel
                   side-length.  A random value is drawn each call so the
                   transform introduces per-sample variation matching the
                   original ``A.Morphological(scale=(2, 3))`` intent.
            p:     Probability of applying this transform (passed to base class).
        """

        def __init__(self, scale: tuple[int, int] = (2, 3), p: float = 0.3, **kwargs: Any) -> None:
            super().__init__(p=p, **kwargs)
            self.scale = scale

        def apply(self, img: Any, **params: Any) -> Any:  # noqa: ANN401
            """Erode ``img`` with a randomly-sized rectangular kernel."""
            import cv2  # noqa: PLC0415 — keep cv2 import scoped
            min_k, max_k = self.scale
            k = random.randint(min_k, max_k)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            return cv2.erode(img, kernel, iterations=1)

        def get_transform_init_args_names(self) -> tuple[str, ...]:
            return ("scale",)

except ImportError:
    # albumentations not installed — define a placeholder so the module is
    # still importable. Tests that need this class use pytest.importorskip.
    ImageOnlyErosion = None  # type: ignore[assignment,misc]


# Canonical scan-noise pipeline probabilities at full strength. Keys identify
# the transform group; values are the probability used in the Albumentations
# Compose. Centralized so warmup can scale them uniformly.
BASE_NOISE_PROBABILITIES: Dict[str, float] = {
    "image_compression": 0.4,
    "noise_oneof": 0.3,
    "blur_oneof": 0.15,
    "brightness_contrast": 0.4,
    "faint_ink": 0.25,
    "rotate": 0.3,
    "grid_distortion": 0.2,
    "elastic_transform": 0.10,
}


def intensity_for_step(step: int, warmup_steps: int) -> float:
    """Linear ramp from 0 to 1 over ``warmup_steps``; 1.0 if warmup disabled.

    - ``warmup_steps <= 0`` -> always 1.0 (warmup disabled)
    - ``step <= 0`` -> 0.0 (clamp; negative steps shouldn't occur)
    - ``step >= warmup_steps`` -> 1.0 (warmup complete)
    - otherwise: linear interpolation ``step / warmup_steps``
    """
    if warmup_steps <= 0:
        return 1.0
    if step <= 0:
        return 0.0
    return min(1.0, step / warmup_steps)


def scaled_probabilities(intensity: float) -> Dict[str, float]:
    """Return base probabilities scaled uniformly by ``intensity`` (clamped to [0, 1])."""
    intensity = max(0.0, min(1.0, intensity))
    return {key: value * intensity for key, value in BASE_NOISE_PROBABILITIES.items()}


def patch_albumentations_for_scan_noise(warmup_steps: int = 0) -> None:
    """Replace ultralytics' default Albumentations transform list with the scan-noise pipeline.

    If ``warmup_steps > 0``, the augmentation intensity ramps linearly from 0
    to 1 over the first ``warmup_steps`` calls to the augmenter. Each call
    increments an internal step counter; transforms are rebuilt per-call when
    warmup is active so probability values reflect the current step.
    """
    import cv2
    import ultralytics.data.augment as ua
    import albumentations as A

    state = {"current_step": 0}

    def _build_transforms(p_overrides: Dict[str, float]):
        return [
            A.ImageCompression(quality_range=(70, 95), p=p_overrides["image_compression"]),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.11), p=0.6),
                A.ISONoise(intensity=(0.05, 0.2), p=0.4),
            ], p=p_overrides["noise_oneof"]),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=p_overrides["blur_oneof"]),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15,
                p=p_overrides["brightness_contrast"],
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(0.2, 0.5),   # asymmetric: wash toward white
                    contrast_limit=(-0.4, -0.1),   # reduce contrast (faint ink)
                    p=0.7,
                ),
                ImageOnlyErosion(scale=(2, 3), p=0.3),
            ], p=p_overrides["faint_ink"]),
            A.Rotate(
                limit=2, border_mode=cv2.BORDER_REPLICATE, fill=255,
                p=p_overrides["rotate"],
            ),
            A.GridDistortion(
                num_steps=5, distort_limit=0.05, border_mode=cv2.BORDER_REPLICATE,
                p=p_overrides["grid_distortion"],
            ),
            A.ElasticTransform(
                alpha=15, sigma=8, border_mode=cv2.BORDER_REPLICATE,
                p=p_overrides["elastic_transform"],
            ),
        ]

    original_init = ua.Albumentations.__init__
    original_call = ua.Albumentations.__call__

    def patched_init(self, p: float = 1.0, transforms=None) -> None:  # noqa: ANN001
        # Build at full strength; per-call rebuild updates probabilities under warmup.
        full_strength = scaled_probabilities(1.0)
        original_init(self, p=p, transforms=_build_transforms(full_strength))
        self._scan_noise_active = True

    def patched_call(self, *args, **kwargs):
        if getattr(self, "_scan_noise_active", False) and warmup_steps > 0:
            intensity = intensity_for_step(state["current_step"], warmup_steps)
            scaled = scaled_probabilities(intensity)
            self.transform = A.Compose(_build_transforms(scaled), p=1.0)
            state["current_step"] += 1
            if state["current_step"] % 500 == 0:
                print(f"[scan-noise] step={state['current_step']} intensity={intensity:.3f}")
        return original_call(self, *args, **kwargs)

    ua.Albumentations.__init__ = patched_init
    ua.Albumentations.__call__ = patched_call
