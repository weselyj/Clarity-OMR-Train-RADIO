#!/usr/bin/env python3
"""Monitor execute-mode training telemetry and flag common failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def load_step_records(step_log_path: Path) -> List[Dict[str, object]]:
    if not step_log_path.exists():
        raise FileNotFoundError(f"Step log not found: {step_log_path}")
    records: List[Dict[str, object]] = []
    with step_log_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {step_log_path}:{line_no}") from exc
            for key in ("global_step", "stage_name", "loss", "grad_norm"):
                if key not in row:
                    raise ValueError(f"Missing key '{key}' in {step_log_path}:{line_no}")
            if "lr" not in row and "lr_new_modules" not in row and "lr_dora" not in row:
                raise ValueError(
                    f"Missing learning-rate field ('lr' or 'lr_new_modules'/'lr_dora') in {step_log_path}:{line_no}"
                )
            records.append(row)
    return records


def _parse_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_yolo_results_records(yolo_results_path: Path) -> List[Dict[str, object]]:
    if not yolo_results_path.exists():
        raise FileNotFoundError(f"YOLO results file not found: {yolo_results_path}")

    records: List[Dict[str, object]] = []
    with yolo_results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        lr_columns = [name for name in fieldnames if name.startswith("lr/")]
        train_loss_columns = [name for name in ("train/box_loss", "train/cls_loss", "train/dfl_loss") if name in fieldnames]
        val_loss_columns = [name for name in ("val/box_loss", "val/cls_loss", "val/dfl_loss") if name in fieldnames]

        for row in reader:
            epoch_value = _parse_float(row.get("epoch"), default=float(len(records)))
            global_step = int(epoch_value) + 1

            losses = [_parse_float(row.get(name)) for name in train_loss_columns]
            finite_losses = [value for value in losses if math.isfinite(value)]
            if not finite_losses:
                losses = [_parse_float(row.get(name)) for name in val_loss_columns]
                finite_losses = [value for value in losses if math.isfinite(value)]
            if finite_losses:
                loss = float(sum(finite_losses))
            else:
                loss = float("nan")

            lrs = [_parse_float(row.get(name)) for name in lr_columns]
            finite_lrs = [value for value in lrs if math.isfinite(value)]
            lr_value = float(statistics.fmean(finite_lrs)) if finite_lrs else 0.0
            non_finite_loss = not math.isfinite(loss)

            records.append(
                {
                    "global_step": global_step,
                    "stage_name": "stage_a_yolo",
                    "loss": loss if math.isfinite(loss) else 0.0,
                    "lr": lr_value,
                    "lr_new_modules": lr_value,
                    "lr_dora": lr_value,
                    "grad_norm": 0.0,
                    "non_finite_loss": non_finite_loss,
                    "non_finite_grad": False,
                    "timestamp_utc": None,
                }
            )
    return records


def load_records(
    *,
    step_log_path: Optional[Path] = None,
    yolo_results_path: Optional[Path] = None,
) -> List[Dict[str, object]]:
    has_step_log = step_log_path is not None
    has_yolo_results = yolo_results_path is not None
    if has_step_log == has_yolo_results:
        raise ValueError("Provide exactly one of --step-log or --yolo-results.")
    if has_step_log:
        return load_step_records(step_log_path)
    return load_yolo_results_records(yolo_results_path)


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def analyze_records(
    records: Sequence[Dict[str, object]],
    window: int,
    spike_factor: float,
    grad_threshold: float,
    lr_min: float,
    lr_max: float,
) -> Dict[str, object]:
    if not records:
        raise ValueError("No telemetry records found.")

    losses = [float(row["loss"]) for row in records]
    new_module_lrs = [
        float(row.get("lr_new_modules", row.get("lr", row.get("lr_dora", 0.0)))) for row in records
    ]
    dora_lrs = [float(row.get("lr_dora", row.get("lr", 0.0))) for row in records]
    grad_norms = [float(row["grad_norm"]) for row in records]
    stage_names = sorted({str(row["stage_name"]) for row in records})

    latest = records[-1]
    window_size = max(1, window)
    recent_losses = losses[-window_size:]
    previous_losses = losses[-2 * window_size : -window_size] if len(losses) >= 2 * window_size else []

    non_finite_loss_steps = [
        int(row["global_step"]) for row in records if bool(row.get("non_finite_loss", False))
    ]
    non_finite_grad_steps = [
        int(row["global_step"]) for row in records if bool(row.get("non_finite_grad", False))
    ]
    high_grad_steps = [
        int(row["global_step"]) for row in records if float(row["grad_norm"]) > grad_threshold
    ]

    spike_steps: List[int] = []
    for idx in range(window_size, len(losses)):
        baseline = _mean(losses[idx - window_size : idx])
        if baseline <= 0:
            continue
        if losses[idx] > baseline * spike_factor:
            spike_steps.append(int(records[idx]["global_step"]))

    alerts: List[str] = []
    if non_finite_loss_steps:
        alerts.append(f"non_finite_loss at steps {non_finite_loss_steps[:10]}")
    if non_finite_grad_steps:
        alerts.append(f"non_finite_grad at steps {non_finite_grad_steps[:10]}")
    if high_grad_steps:
        alerts.append(f"grad_norm>{grad_threshold} at steps {high_grad_steps[:10]}")
    if spike_steps:
        alerts.append(f"loss_spikes(>{spike_factor}x baseline) at steps {spike_steps[:10]}")
    if any(lr <= lr_min for lr in new_module_lrs):
        alerts.append(f"lr <= {lr_min} detected")
    if lr_max > 0 and any(lr > lr_max for lr in new_module_lrs):
        alerts.append(f"lr > {lr_max} detected")
    if previous_losses and _mean(recent_losses) > _mean(previous_losses) * 1.15:
        alerts.append("loss_trend_worsening over recent window")

    return {
        "steps": len(records),
        "stages_seen": stage_names,
        "latest": {
            "global_step": int(latest["global_step"]),
            "stage_name": str(latest["stage_name"]),
            "loss": float(latest["loss"]),
            "lr_new_modules": float(latest.get("lr_new_modules", latest.get("lr", latest.get("lr_dora", 0.0)))),
            "lr_dora": float(latest.get("lr_dora", latest.get("lr", 0.0))),
            "grad_norm": float(latest["grad_norm"]),
            "timestamp_utc": latest.get("timestamp_utc"),
        },
        "loss": {
            "min": min(losses),
            "max": max(losses),
            "latest": losses[-1],
            "mean_recent_window": _mean(recent_losses),
            "mean_previous_window": _mean(previous_losses) if previous_losses else None,
        },
        "lr": {
            "min_new_modules": min(new_module_lrs),
            "max_new_modules": max(new_module_lrs),
            "latest_new_modules": new_module_lrs[-1],
            "distinct_new_modules": len({round(value, 12) for value in new_module_lrs}),
            "min_dora": min(dora_lrs),
            "max_dora": max(dora_lrs),
            "latest_dora": dora_lrs[-1],
            "distinct_dora": len({round(value, 12) for value in dora_lrs}),
        },
        "grad_norm": {
            "min": min(grad_norms),
            "max": max(grad_norms),
            "latest": grad_norms[-1],
            "mean_recent_window": _mean(grad_norms[-window_size:]),
        },
        "anomalies": {
            "non_finite_loss_steps": non_finite_loss_steps,
            "non_finite_grad_steps": non_finite_grad_steps,
            "high_grad_steps": high_grad_steps,
            "loss_spike_steps": spike_steps,
        },
        "alerts": alerts,
        "healthy": len(alerts) == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage-B or Stage-A training telemetry.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--step-log", type=Path, help="Path to JSONL step telemetry from train.py.")
    source_group.add_argument("--yolo-results", type=Path, help="Path to YOLO results.csv from Stage-A training.")
    parser.add_argument("--window", type=int, default=20, help="Window for trend analysis.")
    parser.add_argument("--spike-factor", type=float, default=3.0, help="Loss spike multiplier threshold.")
    parser.add_argument("--grad-threshold", type=float, default=100.0, help="Gradient norm alert threshold.")
    parser.add_argument("--lr-min", type=float, default=0.0, help="Minimum allowed LR.")
    parser.add_argument("--lr-max", type=float, default=0.0, help="Maximum allowed LR (0 disables).")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with code 2 if any alerts are detected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(step_log_path=args.step_log, yolo_results_path=args.yolo_results)
    summary = analyze_records(
        records=records,
        window=args.window,
        spike_factor=args.spike_factor,
        grad_threshold=args.grad_threshold,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if args.fail_on_alert and summary["alerts"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
