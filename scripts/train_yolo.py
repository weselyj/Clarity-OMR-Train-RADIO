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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
