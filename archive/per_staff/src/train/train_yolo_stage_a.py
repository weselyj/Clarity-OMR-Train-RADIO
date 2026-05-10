#!/usr/bin/env python3
"""Train Stage-A YOLOv8 model from synthetic page manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.yolo_stage_a import YoloStageA, YoloStageAConfig


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train YOLOv8 Stage-A detector.")
    parser.add_argument("--project-root", type=Path, default=project_root, help="Repository root path.")
    parser.add_argument(
        "--page-manifest",
        type=Path,
        default=project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_pages.jsonl",
        help="Synthetic page manifest JSONL.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=project_root / "data" / "processed" / "synthetic" / "yolo_splits",
        help="Output directory for data.yaml and train/val/test split files.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Optional prebuilt YOLO data.yaml. If set, split generation from page manifest is skipped.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.90, help="YOLO training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="YOLO validation split ratio.")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="yolov8m.pt",
        help="Ultralytics checkpoint or model name.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=2048, help="Training image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic seed.")
    parser.add_argument("--device", type=str, default=None, help="Training device (e.g. 0 or cpu).")
    parser.add_argument(
        "--run-project",
        type=Path,
        default=project_root / "runs" / "stage_a",
        help="Ultralytics project output directory.",
    )
    parser.add_argument("--run-name", type=str, default="yolov8m-stage-a", help="Ultralytics run name.")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic augmentation probability.")
    parser.add_argument("--close-mosaic", type=int, default=0, help="Disable mosaic in final N epochs (0 keeps it off).")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Horizontal flip probability.")
    parser.add_argument("--erasing", type=float, default=0.0, help="Random erasing probability.")
    parser.add_argument("--hsv-h", type=float, default=0.0, help="HSV hue augmentation fraction.")
    parser.add_argument("--hsv-s", type=float, default=0.0, help="HSV saturation augmentation fraction.")
    parser.add_argument("--hsv-v", type=float, default=0.1, help="HSV value (brightness) augmentation fraction.")
    parser.add_argument("--scale", type=float, default=0.15, help="Scale augmentation gain (+/-).")
    parser.add_argument("--translate", type=float, default=0.05, help="Translation augmentation fraction (+/-).")
    parser.add_argument("--cls-gain", type=float, default=1.5, help="Classification loss gain (Ultralytics 'cls').")
    parser.add_argument(
        "--auto-augment",
        type=str,
        default="none",
        choices=["none", "randaugment", "autoaugment", "augmix"],
        help="Auto augmentation policy ('none' disables it).",
    )
    parser.add_argument(
        "--cos-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cosine learning-rate scheduling.",
    )
    parser.add_argument(
        "--rect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rectangular batches for training.",
    )
    return parser.parse_args()


def _resolve_device(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"cpu", "mps"}:
        return lowered

    is_cuda_request = lowered.startswith("cuda") or all(part.strip().isdigit() for part in cleaned.split(","))
    if not is_cuda_request:
        return cleaned

    try:
        import torch
    except Exception:
        return cleaned

    if not torch.cuda.is_available():
        print("[Stage-A] CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        return "cpu"

    if all(part.strip().isdigit() for part in cleaned.split(",")):
        requested_ids = [int(part.strip()) for part in cleaned.split(",") if part.strip()]
        if requested_ids and max(requested_ids) >= int(torch.cuda.device_count()):
            print("[Stage-A] Requested CUDA device index is out of range; falling back to CPU.", file=sys.stderr)
            return "cpu"

    return cleaned


def main() -> None:
    args = parse_args()
    stage_a = YoloStageA(YoloStageAConfig(seed=args.seed))
    if args.data_yaml is not None:
        data_yaml = args.data_yaml.resolve()
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    else:
        data_yaml = stage_a.build_training_data_yaml(
            page_manifest_path=args.page_manifest,
            output_dir=args.split_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency based
        raise RuntimeError("ultralytics is required for YOLO training. Install with: pip install ultralytics") from exc

    model = YOLO(args.model_checkpoint)
    auto_augment = None if str(args.auto_augment).lower() == "none" else str(args.auto_augment)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch_size),
        "workers": int(args.workers),
        "project": str(args.run_project.resolve()),
        "name": str(args.run_name),
        "seed": int(args.seed),
        "mosaic": float(args.mosaic),
        "close_mosaic": int(args.close_mosaic),
        "fliplr": float(args.fliplr),
        "erasing": float(args.erasing),
        "hsv_h": float(args.hsv_h),
        "hsv_s": float(args.hsv_s),
        "hsv_v": float(args.hsv_v),
        "scale": float(args.scale),
        "translate": float(args.translate),
        "cls": float(args.cls_gain),
        "auto_augment": auto_augment,
        "cos_lr": bool(args.cos_lr),
        "rect": bool(args.rect),
    }
    device = _resolve_device(args.device)
    if device is not None:
        train_kwargs["device"] = device
    model.train(**train_kwargs)

    run_dir = args.run_project.resolve() / args.run_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    summary = {
        "data_yaml": str(data_yaml),
        "run_dir": str(run_dir),
        "best_weights": str(best_weights) if best_weights.exists() else None,
        "last_weights": str(last_weights) if last_weights.exists() else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

