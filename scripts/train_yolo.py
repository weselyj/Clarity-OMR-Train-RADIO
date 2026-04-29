"""Train a YOLO model on the mixed Stage A dataset.

Small ultralytics wrapper. Hyperparameters mirror the recovered config from
the original Stage A YOLO checkpoint (rect=True, cos_lr=True, augmentation
disabled), with the mixed dataset and chosen base model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model (e.g. yolov8m.pt or yolo26m.pt)")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml path")
    parser.add_argument("--name", required=True, help="Run name (becomes runs/<name>)")
    parser.add_argument("--project", default="runs", help="Top-level run dir")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1920)
    # batch=8 matches the original Stage A training config. batch=16 OOMs on
    # the 5090 at imgsz=1920 (verified empirically with cu132 nightly torch).
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    # ultralytics defaults workers=8. On Windows each worker spawns ~3 sub-processes
    # so 8 workers = ~25 python.exe processes consuming ~66 GB system RAM. Reduce
    # to 6 for ~50 GB; 4 for ~35 GB. GPU utilization barely changes between 4-8.
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile for ~30%% speedup. Requires triton; OOMs at "
             "batch=16 imgsz=1920. Default off.",
    )
    parser.add_argument(
        "--noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable scan-noise augmentation pipeline. Simulates JPEG compression, "
            "sensor noise, slight blur, brightness/contrast variation, scan skew, "
            "and gentle page curvature (non-flat scans). Intended for YOLO26m and "
            "later runs; leave off for the YOLOv8m clean-data baseline."
        ),
    )
    return parser.parse_args()


def _patch_albumentations_for_scan_noise() -> None:
    """Replace ultralytics' default Albumentations transform list with a scan-noise pipeline.

    Real-PDF eval inputs have: JPEG compression artifacts, faint sensor noise,
    slight blur from out-of-focus scans, brightness/contrast variation, scan
    skew (small rotation), and gentle page curvature when a book wasn't pressed
    flat against the scanner glass. This pipeline simulates that distribution
    so the model generalizes to noisy real-world inputs at eval time.

    All transforms are bbox-aware via ultralytics' built-in BboxParams handling.
    """
    import cv2
    import ultralytics.data.augment as ua
    import albumentations as A

    scan_noise_transforms = [
        # JPEG compression artifacts (most lieder eval PDFs are JPEG-compressed)
        A.ImageCompression(quality_range=(70, 95), p=0.4),
        # Faint scan noise (sensor / pixel-level)
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.11), p=0.6),
            A.ISONoise(intensity=(0.05, 0.2), p=0.4),
        ], p=0.3),
        # Slight blur from scanner motion or out-of-focus pages
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=0.15),
        # Brightness/contrast variation across different scanners
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        # Small rotation simulates scan skew; fill=255 keeps margins white
        A.Rotate(limit=2, border_mode=cv2.BORDER_REPLICATE, fill=255, p=0.3),
        # GENTLE page curvature - simulates books that aren't pressed flat against
        # the scanner glass (spine bulge, page lift). num_steps=5 keeps the warp
        # mesh coarse so distortion is smooth rather than locally jagged.
        # distort_limit=0.05 = up to 5% displacement at mesh nodes.
        A.GridDistortion(num_steps=5, distort_limit=0.05, border_mode=cv2.BORDER_REPLICATE, p=0.2),
        # Local paper bumps / surface unevenness (very mild).
        # alpha low + sigma high = smooth global wave rather than sharp local elastic
        # deformation; appropriate for music scores where staff lines must remain
        # locally straight enough that the model can still detect them.
        A.ElasticTransform(alpha=15, sigma=8, border_mode=cv2.BORDER_REPLICATE, p=0.10),
    ]

    original_init = ua.Albumentations.__init__

    def patched_init(self, p: float = 1.0, transforms=None) -> None:  # noqa: ANN001
        # Always inject our scan-noise transforms; ignore caller-supplied list
        # (ultralytics' build_transforms passes None here in the default code path).
        original_init(self, p=p, transforms=scan_noise_transforms)

    ua.Albumentations.__init__ = patched_init


def main() -> None:
    args = parse_args()
    if args.noise:
        _patch_albumentations_for_scan_noise()
    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        rect=True,
        cos_lr=True,
        name=args.name,
        project=args.project,
        hsv_h=0, hsv_s=0,
        flipud=0, fliplr=0,
        mosaic=0, mixup=0,
        save=True,
        patience=args.patience,
        workers=args.workers,
    )
    if args.compile:
        train_kwargs["compile"] = True
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
