#!/usr/bin/env python3
"""Generate 200 fresh synthetic_systems samples for Phase 2 eval.

Design: calls build_synthetic_systems_v1.py at dpi150 (the training run used
dpi300) so all produced crop images are genuinely unseen by the model while
retaining the same per-staff token sequences as ground truth.  No seed
parameter is needed — isolation is achieved by DPI, not random sampling.

Why dpi150 and not a held-out page set?
  The existing synthetic_systems_v1 manifest contains all 20,583 samples
  labelled split=train.  Only 46 pages were never processed (they produced no
  YOLO matches or were excluded upstream), giving an estimated ~98 systems —
  not enough for 200.  Using dpi150 page images for the same score pages
  produces different pixel content (visually distinct resolution) at the same
  bounding-box coordinates, so YOLO detection still works and the ground-truth
  token sequences are identical.  The model was trained exclusively on dpi300
  crops, making dpi150 crops genuinely out-of-distribution from the image
  perspective while remaining in-distribution for the musical content.

Approach:
  1. Invoke build_synthetic_systems_v1.py at dpi150 with --max-pages PAGES
     (default 90, which empirically gives >200 systems at ~3 sys/page).
  2. Truncate the raw output manifest to TARGET_COUNT rows.
  3. Relabel each row: dataset="synthetic_systems", split="test".
  4. Write to src/data/manifests/synthetic_systems_eval_fresh.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_TARGET_COUNT = 200
# Use enough pages to comfortably exceed TARGET_COUNT.
# Empirical rate: ~3 systems/page at dpi300.  dpi150 YOLO detection
# matches at a similar rate (same oracle labels, different pixel scale).
DEFAULT_MAX_PAGES = 90
DEFAULT_DPI = "dpi150"  # dpi300 was used for training; dpi150 is unseen
DEFAULT_OUTPUT_DIR = ROOT / "data" / "eval" / "synthetic_fresh"
DEFAULT_MANIFEST_OUT = ROOT / "src" / "data" / "manifests" / "synthetic_systems_eval_fresh.jsonl"

# Relative to ROOT — these are the same paths used by the training builder
BUILDER_SCRIPT = ROOT / "scripts" / "build_synthetic_systems_v1.py"
PER_STAFF_MANIFEST = ROOT / "data" / "processed" / "synthetic_v2" / "manifests" / "synthetic_token_manifest.jsonl"
LABELS_SYSTEMS_ROOT = ROOT / "data" / "processed" / "synthetic_v2" / "labels_systems"
PAGE_IMAGES_ROOT = ROOT / "data" / "processed" / "synthetic_v2" / "images"
YOLO_WEIGHTS = ROOT / "runs" / "detect" / "runs" / "yolo26m_systems" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help=f"Number of eval samples to emit (default: {DEFAULT_TARGET_COUNT}).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"Pages passed to builder (default: {DEFAULT_MAX_PAGES}). "
             "Increase if the builder produces fewer than --count systems.",
    )
    p.add_argument(
        "--dpi",
        default=DEFAULT_DPI,
        help=f"DPI variant to render (default: {DEFAULT_DPI}). "
             "Must differ from training DPI (dpi300).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for rendered system crops (default: {DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_OUT,
        help=f"Output manifest path (default: {DEFAULT_MANIFEST_OUT}).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for subprocess (default: current interpreter).",
    )
    return p.parse_args()


def _check_prerequisites(args: argparse.Namespace) -> None:
    missing: list[str] = []
    for path, label in [
        (BUILDER_SCRIPT, "builder script"),
        (PER_STAFF_MANIFEST, "per-staff manifest"),
        (LABELS_SYSTEMS_ROOT, "labels_systems root"),
        (PAGE_IMAGES_ROOT / args.dpi, f"page images at {args.dpi}"),
        (YOLO_WEIGHTS, "YOLO weights"),
    ]:
        if not path.exists():
            missing.append(f"  {label}: {path}")
    if missing:
        print("[gen-eval] ERROR: missing prerequisites:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        raise SystemExit(1)


def _run_builder(args: argparse.Namespace, raw_manifest: Path, raw_crops_root: Path, audit_path: Path) -> int:
    """Invoke build_synthetic_systems_v1.py as a subprocess.  Returns its exit code."""
    cmd = [
        args.python,
        str(BUILDER_SCRIPT),
        "--per-staff-manifest", str(PER_STAFF_MANIFEST),
        "--labels-systems-root", str(LABELS_SYSTEMS_ROOT),
        "--page-images-root", str(PAGE_IMAGES_ROOT),
        "--dpi", args.dpi,
        "--yolo-weights", str(YOLO_WEIGHTS),
        "--output-manifest", str(raw_manifest),
        "--output-crops-root", str(raw_crops_root),
        "--audit-output", str(audit_path),
        "--max-pages", str(args.max_pages),
        "--dataset-name", "synthetic_systems",
    ]
    print(f"[gen-eval] running builder:", file=sys.stderr)
    print(f"  {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def _filter_and_relabel(
    raw_manifest: Path,
    target_count: int,
    manifest_out: Path,
) -> int:
    """Read raw builder output, relabel as split=test, write first target_count rows."""
    rows: list[dict] = []
    with raw_manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["dataset"] = "synthetic_systems"
            row["split"] = "test"
            rows.append(row)
            if len(rows) >= target_count:
                break

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return len(rows)


def main() -> int:
    args = parse_args()

    print(f"[gen-eval] Phase 2 synthetic eval sample generator", file=sys.stderr)
    print(f"[gen-eval]   DPI variant : {args.dpi}  (training used dpi300)", file=sys.stderr)
    print(f"[gen-eval]   Target count: {args.count}", file=sys.stderr)
    print(f"[gen-eval]   Max pages   : {args.max_pages}", file=sys.stderr)
    print(f"[gen-eval]   Output dir  : {args.output_dir}", file=sys.stderr)
    print(f"[gen-eval]   Manifest out: {args.manifest}", file=sys.stderr)

    _check_prerequisites(args)

    # Paths for intermediate builder output
    raw_manifest = args.output_dir / "raw_builder_manifest.jsonl"
    raw_crops_root = args.output_dir / "system_crops"
    audit_path = args.output_dir / "builder_audit.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run the builder
    rc = _run_builder(args, raw_manifest, raw_crops_root, audit_path)
    if rc != 0:
        print(f"[gen-eval] ERROR: builder exited with code {rc}", file=sys.stderr)
        return rc

    if not raw_manifest.exists() or raw_manifest.stat().st_size == 0:
        print("[gen-eval] ERROR: builder produced an empty manifest", file=sys.stderr)
        return 1

    # Count raw entries
    with raw_manifest.open(encoding="utf-8") as f:
        raw_count = sum(1 for line in f if line.strip())
    print(f"[gen-eval] builder produced {raw_count} raw systems", file=sys.stderr)

    if raw_count < args.count:
        print(
            f"[gen-eval] WARNING: builder produced only {raw_count} systems, "
            f"fewer than the requested {args.count}.  "
            f"Increase --max-pages (currently {args.max_pages}) to get more.",
            file=sys.stderr,
        )

    # Step 2: Filter, relabel, truncate
    written = _filter_and_relabel(raw_manifest, args.count, args.manifest)
    print(f"[gen-eval] wrote {written} rows to {args.manifest}", file=sys.stderr)

    if written < args.count:
        print(
            f"[gen-eval] WARNING: only {written}/{args.count} rows written. "
            "Re-run with larger --max-pages.",
            file=sys.stderr,
        )
        return 1

    # Verify: print first row for sanity check
    with args.manifest.open(encoding="utf-8") as f:
        first_row = json.loads(f.readline())
    print(f"[gen-eval] first row:", file=sys.stderr)
    print(json.dumps(first_row, indent=2), file=sys.stderr)

    print(f"[gen-eval] DONE — {written} eval samples at {args.manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
