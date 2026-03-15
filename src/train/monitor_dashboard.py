#!/usr/bin/env python3
"""Live CLI dashboard for Stage-A/Stage-B training telemetry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.train.monitor_training import analyze_records, load_records


def build_state_payload(
    step_log: Optional[Path],
    yolo_results: Optional[Path],
    window: int,
    spike_factor: float,
    grad_threshold: float,
    lr_min: float,
    lr_max: float,
    tail_limit: int,
) -> Dict[str, object]:
    try:
        records = load_records(step_log_path=step_log, yolo_results_path=yolo_results)
    except FileNotFoundError:
        missing_source = step_log if step_log is not None else yolo_results
        return {
            "healthy": False,
            "alerts": [f"telemetry source not found: {missing_source}"],
            "summary": None,
            "recent_steps": [],
        }
    except ValueError as exc:
        return {
            "healthy": False,
            "alerts": [f"invalid telemetry source: {exc}"],
            "summary": None,
            "recent_steps": [],
        }

    if not records:
        return {
            "healthy": False,
            "alerts": ["telemetry source is empty"],
            "summary": None,
            "recent_steps": [],
        }

    summary = analyze_records(
        records=records,
        window=window,
        spike_factor=spike_factor,
        grad_threshold=grad_threshold,
        lr_min=lr_min,
        lr_max=lr_max,
    )
    tail = max(1, tail_limit)
    recent_steps = records[-tail:]
    return {
        "healthy": bool(summary.get("healthy", False)),
        "alerts": list(summary.get("alerts", [])),
        "summary": summary,
        "recent_steps": recent_steps,
    }


def _fmt(value: object, precision: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.{precision}g}"
        return str(value)
    text = str(value)
    return text if text else "-"


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def render_cli_dashboard(
    payload: Dict[str, object],
    *,
    telemetry_source: Path,
    refresh_ms: int,
    tail_limit: int,
) -> str:
    lines: List[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("OMR Training CLI Dashboard")
    lines.append(f"Updated: {now}")
    lines.append(f"Source: {telemetry_source}")
    lines.append(f"Refresh: {refresh_ms} ms | Tail rows: {tail_limit}")
    lines.append("=" * 110)

    summary = payload.get("summary")
    healthy = bool(payload.get("healthy", False))
    if not isinstance(summary, dict):
        lines.append(f"Health: {'HEALTHY' if healthy else 'ALERT'}")
        lines.append("No summary data available yet.")
        alerts = payload.get("alerts", [])
        if isinstance(alerts, list) and alerts:
            lines.append("Alerts:")
            for alert in alerts:
                lines.append(f"  - {alert}")
        lines.append("")
        lines.append("Press Ctrl+C to stop.")
        return "\n".join(lines)

    latest = summary.get("latest", {})
    if not isinstance(latest, dict):
        latest = {}
    lines.append(f"Health: {'HEALTHY' if healthy else 'ALERT'}")
    lines.append(
        "Latest: "
        f"step={_fmt(latest.get('global_step'))} "
        f"stage={_fmt(latest.get('stage_name'))} "
        f"loss={_fmt(latest.get('loss'))} "
        f"lr_new={_fmt(latest.get('lr_new_modules'))} "
        f"lr_dora={_fmt(latest.get('lr_dora'))} "
        f"grad={_fmt(latest.get('grad_norm'))}"
    )
    lines.append("-" * 110)

    alerts = payload.get("alerts", [])
    if isinstance(alerts, list) and alerts:
        lines.append("Alerts:")
        for alert in alerts:
            lines.append(f"  - {alert}")
    else:
        lines.append("Alerts: none")
    lines.append("-" * 110)

    lines.append("Recent steps:")
    header = f"{'step':>6} {'stage':<18} {'loss':>10} {'lr_new':>10} {'lr_dora':>10} {'grad':>10} {'nf_loss':>8} {'nf_grad':>8} {'timestamp':<22}"
    lines.append(header)
    lines.append("-" * len(header))

    recent_rows = payload.get("recent_steps", [])
    if not isinstance(recent_rows, list):
        recent_rows = []
    for row in reversed(recent_rows):
        if not isinstance(row, dict):
            continue
        stage_name = _truncate(_fmt(row.get("stage_name")), 18)
        timestamp = _truncate(_fmt(row.get("timestamp_utc")), 22)
        line = (
            f"{_fmt(row.get('global_step')):>6} "
            f"{stage_name:<18} "
            f"{_fmt(row.get('loss')):>10} "
            f"{_fmt(row.get('lr_new_modules', row.get('lr'))):>10} "
            f"{_fmt(row.get('lr_dora', row.get('lr'))):>10} "
            f"{_fmt(row.get('grad_norm')):>10} "
            f"{_fmt(row.get('non_finite_loss', False)):>8} "
            f"{_fmt(row.get('non_finite_grad', False)):>8} "
            f"{timestamp:<22}"
        )
        lines.append(line)

    lines.append("")
    lines.append("Press Ctrl+C to stop.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show a live terminal dashboard for OMR training telemetry.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--step-log", type=Path, help="JSONL telemetry from train.py --step-log.")
    source_group.add_argument("--yolo-results", type=Path, help="YOLO results.csv from Stage-A training.")
    parser.add_argument("--window", type=int, default=20, help="Trend-analysis window.")
    parser.add_argument("--spike-factor", type=float, default=3.0, help="Loss spike multiplier threshold.")
    parser.add_argument("--grad-threshold", type=float, default=100.0, help="Gradient norm threshold.")
    parser.add_argument("--lr-min", type=float, default=0.0, help="Minimum allowed learning rate.")
    parser.add_argument("--lr-max", type=float, default=0.0, help="Maximum allowed learning rate (0 disables).")
    parser.add_argument("--refresh-ms", type=int, default=2000, help="Refresh interval in milliseconds.")
    parser.add_argument("--tail-limit", type=int, default=40, help="Number of latest rows to show.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot JSON and exit.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal on each refresh.")
    parser.add_argument(
        "--max-refreshes",
        type=int,
        default=None,
        help="Optional refresh loop cap for scripting/debugging.",
    )
    # Backward-compatible no-op args from old web mode.
    parser.add_argument("--host", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    telemetry_source = args.step_log if args.step_log is not None else args.yolo_results
    assert telemetry_source is not None

    if args.once:
        snapshot = build_state_payload(
            step_log=args.step_log,
            yolo_results=args.yolo_results,
            window=args.window,
            spike_factor=args.spike_factor,
            grad_threshold=args.grad_threshold,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            tail_limit=args.tail_limit,
        )
        print(json.dumps(snapshot, indent=2))
        return

    refreshes = 0
    sleep_seconds = max(0.3, float(args.refresh_ms) / 1000.0)
    try:
        while True:
            payload = build_state_payload(
                step_log=args.step_log,
                yolo_results=args.yolo_results,
                window=args.window,
                spike_factor=args.spike_factor,
                grad_threshold=args.grad_threshold,
                lr_min=args.lr_min,
                lr_max=args.lr_max,
                tail_limit=args.tail_limit,
            )
            screen = render_cli_dashboard(
                payload,
                telemetry_source=telemetry_source,
                refresh_ms=args.refresh_ms,
                tail_limit=args.tail_limit,
            )
            if args.no_clear:
                print(screen)
                print("")
            else:
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.write(screen + "\n")
                sys.stdout.flush()

            refreshes += 1
            if args.max_refreshes is not None and refreshes >= max(1, int(args.max_refreshes)):
                break
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
