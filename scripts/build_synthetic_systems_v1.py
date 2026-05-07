#!/usr/bin/env python3
"""Build the synthetic_systems_v1 manifest.

For each synthetic_v2 page, run system-level YOLO, IoU-match against oracle
system bboxes, assemble multi-staff token sequences with <staff_idx_N> markers,
crop YOLO system regions, and emit one manifest entry per matched system.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.yolo_aligned_systems import process_page_systems


def _load_token_lookup(per_staff_manifest: Path) -> dict:
    """Load synthetic_v2 per-staff manifest into {(page_id, staff_index): entry}."""
    lookup: dict = {}
    with per_staff_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry["page_id"], entry["staff_index"])
            lookup[key] = entry
    return lookup


def _group_pages(token_lookup: dict) -> dict:
    """Group token_lookup keys → {(page_id, style_id): {"image_path": str, "page_number": int}}."""
    pages: dict = {}
    for (page_id, _staff_index), entry in token_lookup.items():
        key = (page_id, entry["style_id"])
        if key not in pages:
            pages[key] = {
                "image_path": entry["image_path"],
                "page_number": entry["page_number"],
                "style_id": entry["style_id"],
            }
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-staff-manifest", type=Path, required=True,
                        help="data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl")
    parser.add_argument("--labels-systems-root", type=Path, required=True,
                        help="data/processed/synthetic_v2/labels_systems/")
    parser.add_argument("--page-images-root", type=Path, required=True,
                        help="data/processed/synthetic_v2/images/ (parent of <dpi>/<style>/ subdirs)")
    parser.add_argument("--dpi", default="dpi300",
                        help="DPI subdirectory under page-images-root (one of dpi94/dpi150/dpi300). "
                             "Parent spec §3.1 assumes 300 DPI for system-height calculations.")
    parser.add_argument("--yolo-weights", type=Path, required=True,
                        help="runs/detect/runs/yolo26m_systems/weights/best.pt")
    parser.add_argument("--output-manifest", type=Path,
                        default=Path("data/processed/synthetic_systems_v1/manifests/synthetic_token_manifest.jsonl"))
    parser.add_argument("--output-crops-root", type=Path,
                        default=Path("data/processed/synthetic_systems_v1/system_crops"))
    parser.add_argument("--audit-output", type=Path,
                        default=Path("data/processed/synthetic_systems_v1/audit.json"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Stop after N pages (for smoke testing).")
    parser.add_argument("--dataset-name", default="synthetic_systems")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[builder] loading per-staff manifest: {args.per_staff_manifest}", flush=True)
    token_lookup = _load_token_lookup(args.per_staff_manifest)
    print(f"[builder] loaded {len(token_lookup)} per-staff entries", flush=True)

    pages = _group_pages(token_lookup)
    print(f"[builder] grouped to {len(pages)} unique (page_id, style_id) pairs", flush=True)

    # Lazy import: only load YOLO when actually about to use it (allows the script
    # to be importable for tests without ultralytics installed).
    from ultralytics import YOLO  # type: ignore
    yolo_model = YOLO(str(args.yolo_weights))
    print(f"[builder] loaded YOLO weights: {args.yolo_weights}", flush=True)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_crops_root.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "pages_processed": 0,
        "yolo_boxes": 0,
        "oracle_systems": 0,
        "matches": 0,
        "dropped_yolo_fp": 0,
        "dropped_oracle_recall_gap": 0,
        "dropped_token_miss": 0,
        "dropped_degenerate": 0,
        "dropped_marker_overflow": 0,
        "errors": 0,
        "entries_written": 0,
        "staves_histogram": Counter(),
        "split_histogram": Counter(),
    }

    with args.output_manifest.open("w", encoding="utf-8") as fout:
        for i, ((page_id, style_id), page_meta) in enumerate(sorted(pages.items())):
            if args.max_pages is not None and i >= args.max_pages:
                break

            page_image_path = args.page_images_root / args.dpi / style_id / f"{page_id}.png"
            label_dir = args.labels_systems_root / style_id
            label_txt = label_dir / f"{page_id}.txt"
            staves_json = label_dir / f"{page_id}.staves.json"

            if not page_image_path.exists() or not label_txt.exists() or not staves_json.exists():
                aggregate["errors"] += 1
                if aggregate["errors"] <= 5:
                    # Surface the first few missing-file errors for diagnosis; suppress the rest.
                    print(f"[builder] missing input(s) for page_id={page_id!r} style={style_id!r}: "
                          f"image={page_image_path.exists()} txt={label_txt.exists()} json={staves_json.exists()}",
                          flush=True)
                continue

            crops_dir = args.output_crops_root / style_id
            entries, report = process_page_systems(
                page_id=page_id,
                page_image_path=page_image_path,
                oracle_label_path=label_txt,
                oracle_staves_json_path=staves_json,
                yolo_model=yolo_model,
                token_lookup=token_lookup,
                out_crops_dir=crops_dir,
                crop_path_template=f"data/processed/synthetic_systems_v1/system_crops/{style_id}/{{filename}}",
                iou_threshold=args.iou_threshold,
                imgsz=args.imgsz,
                conf=args.conf,
                dataset_name=args.dataset_name,
            )

            aggregate["pages_processed"] += 1
            for k in ("yolo_boxes", "oracle_systems", "matches",
                      "dropped_yolo_fp", "dropped_oracle_recall_gap",
                      "dropped_token_miss", "dropped_degenerate",
                      "dropped_marker_overflow"):
                aggregate[k] += report.get(k, 0)
            for entry in entries:
                fout.write(json.dumps(entry) + "\n")
                aggregate["entries_written"] += 1
                aggregate["staves_histogram"][entry["staves_in_system"]] += 1
                aggregate["split_histogram"][entry["split"]] += 1

            if (i + 1) % 100 == 0:
                print(f"[builder] {i+1}/{len(pages)} pages, {aggregate['entries_written']} entries", flush=True)

    aggregate["staves_histogram"] = dict(aggregate["staves_histogram"])
    aggregate["split_histogram"] = dict(aggregate["split_histogram"])
    aggregate["elapsed_seconds"] = round(time.time() - t0, 1)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(aggregate, indent=2))

    print(f"[builder] DONE — wrote {aggregate['entries_written']} entries to {args.output_manifest}", flush=True)
    print(f"[builder] audit written to {args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
