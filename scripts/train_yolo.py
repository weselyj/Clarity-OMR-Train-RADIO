"""Train a YOLO model on the mixed Stage A dataset.

This is a small ultralytics wrapper. The training hyperparameters mirror the
recovered config from the original Stage A YOLO checkpoint (rect=True, cos_lr=True,
augmentation disabled — sheet music is grayscale, non-flippable, and benefits
from non-square aspect ratios), but with the mixed dataset and the chosen base
model.
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
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable scan-noise augmentation pipeline. Simulates JPEG compression, "
            "sensor noise, slight blur, brightness/contrast variation, and scan skew "
            "that appear in real-PDF eval inputs. Intended for YOLO26m and later runs; "
            "leave off for the YOLOv8m clean-data baseline."
        ),
    )
    return parser.parse_args()


def _patch_albumentations_for_scan_noise() -> None:
    """Replace ultralytics' default Albumentations transform list with a scan-noise pipeline.

    Music scores at eval time come from real PDFs with JPEG compression artifacts,
    slight scan rotation, faint scan banding, and modest brightness variation.
    This pipeline simulates that distribution to improve eval-time generalization.

    Implementation note: ultralytics 8.4.41's Albumentations.__init__ accepts a
    ``transforms`` keyword argument (list of albumentations transforms). When provided,
    ultralytics wraps them in A.Compose with the correct BboxParams automatically —
    no need to construct A.Compose here.  The class also auto-detects spatial
    transforms and enables bbox_params only when needed.

    albumentations 2.x API changes applied here vs the reference outline:
    - GaussNoise: var_limit → std_range  (v2 uses std, not variance)
    - ImageCompression: quality_lower/quality_upper → quality_range=(low, high)
    - Rotate: border_mode default changed; we pass border_mode=cv2.BORDER_REPLICATE
      and fill=255 (white page fill for sheet music)
    """
    import cv2
    import ultralytics.data.augment as ua
    import albumentations as A

    scan_noise_transforms = [
        # JPEG compression artifacts (most lieder eval PDFs are JPEG-compressed)
        A.ImageCompression(quality_range=(70, 95), p=0.4),
        # Faint scan noise (sensor / pixel-level).
        # GaussNoise std_range: 0.02–0.11 ≈ var_limit 5–30 mapped to [0, 255] uint8.
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.11), p=0.6),
            A.ISONoise(intensity=(0.05, 0.2), p=0.4),
        ], p=0.3),
        # Slight blur from scanner motion or out-of-focus pages.
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=0.15),
        # Brightness/contrast variation across different scanners.
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        # Small rotation to simulate scan skew; fill=255 keeps margins white.
        A.Rotate(limit=2, border_mode=cv2.BORDER_REPLICATE, fill=255, p=0.3),
    ]

    # Patch the class so every dataloader instantiation uses our pipeline.
    # ultralytics instantiates Albumentations once per dataset in
    # build_transforms(); patching the class before model.train() ensures
    # both train and val dataloaders pick up the new transforms.
    original_init = ua.Albumentations.__init__

    def patched_init(self, p: float = 1.0, transforms=None) -> None:  # noqa: ANN001
        original_init(self, p=p, transforms=scan_noise_transforms)

    ua.Albumentations.__init__ = patched_init


def main() -> None:
    args = parse_args()
    if args.noise:
        _patch_albumentations_for_scan_noise()
    model = YOLO(args.model)
    model.train(
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
    )


if __name__ == "__main__":
    main()
