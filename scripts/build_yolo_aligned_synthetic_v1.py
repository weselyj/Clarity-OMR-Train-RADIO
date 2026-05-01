"""Driver: build YOLO-aligned crops + manifest for RADIO Stage 3 retrain.

For each page in data/processed/synthetic_multi_dpi/manifests/synthetic_pages.jsonl:
1. Load the full-page PNG (300 DPI) and the oracle YOLO label file.
2. Run YOLO26m Phase 2 on the page.
3. Match YOLO bboxes to oracle staves (IoU >= threshold).
4. Crop YOLO regions from the page, save to out_dir/staff_crops/<style>/.
5. Look up the token sequence for each matched staff and emit a manifest entry.

Writes:
- <out_dir>/staff_crops/<style>/<page_id>__yoloidx<NN>.png (crops)
- <out_dir>/manifests/synthetic_token_manifest.jsonl (per-staff manifest)
- <out_dir>/manifests/build_report.json (aggregate retention stats)

Run on the GPU box. 300 DPI source: data/processed/synthetic_multi_dpi/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.yolo_aligned_crops import process_page  # noqa: E402

DEFAULT_PAGES_MANIFEST = Path("data/processed/synthetic_multi_dpi/manifests/synthetic_pages.jsonl")
DEFAULT_TOKEN_MANIFEST = Path("data/processed/synthetic_multi_dpi/manifests/synthetic_token_manifest.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/synthetic_yolo_v1")
DEFAULT_YOLO_WEIGHTS = Path("runs/detect/runs/yolo26m_phase2_noise/weights/best.pt")
# The synthetic_multi_dpi corpus has 3 DPI variants; pick the 300 DPI version to match
# how lieder PDFs are rendered at inference (eval/run_lieder_eval.py renders at 300 DPI).
PNG_DPI_SUBDIR = "dpi300"


def load_token_lookup(token_manifest_path: Path) -> dict:
    """Build (page_id, staff_index) → token entry lookup."""
    lookup = {}
    for line in token_manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        lookup[(entry["page_id"], entry["staff_index"])] = entry
    return lookup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-manifest", type=Path, default=DEFAULT_PAGES_MANIFEST)
    parser.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, process only the first N pages (for smoke testing).")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Lazy import; ultralytics is heavy.
    from ultralytics import YOLO

    print(f"Loading YOLO from {args.yolo_weights} ...", flush=True)
    yolo = YOLO(str(args.yolo_weights))
    yolo.to(args.device)

    print(f"Loading token lookup from {args.token_manifest} ...", flush=True)
    token_lookup = load_token_lookup(args.token_manifest)
    print(f"Loaded {len(token_lookup)} (page_id, staff_index) entries.", flush=True)

    out_manifest_dir = args.out_dir / "manifests"
    out_manifest_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_path = out_manifest_dir / "synthetic_token_manifest.jsonl"
    out_report_path = out_manifest_dir / "build_report.json"

    pages = []
    seen_page_ids: set[str] = set()
    for line in args.pages_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        # Pages manifest may have multiple entries per page_id (one per DPI variant).
        # Dedupe to one per page; we always use the 300 DPI PNG below.
        if rec["page_id"] in seen_page_ids:
            continue
        seen_page_ids.add(rec["page_id"])
        pages.append(rec)
    if args.limit:
        pages = pages[: args.limit]
    print(f"Processing {len(pages)} unique pages...", flush=True)

    totals = defaultdict(int)
    skipped_pages: list[str] = []
    n_pages_with_match = 0

    with out_manifest_path.open("w", encoding="utf-8") as out_f:
        for i, page in enumerate(pages, 1):
            page_id = page["page_id"]
            style_id = page["style_id"]
            # Always use the 300 DPI variant — match lieder inference distribution.
            png_path = Path(f"data/processed/synthetic_multi_dpi/images/{PNG_DPI_SUBDIR}/{style_id}/{page_id}.png")
            label_path = Path(page["label_path"])

            if not png_path.exists():
                skipped_pages.append(f"{page_id} (missing png)")
                continue
            if not label_path.exists():
                skipped_pages.append(f"{page_id} (missing label)")
                continue

            out_crops_dir = args.out_dir / "staff_crops" / style_id
            crop_path_template = (
                f"data/processed/synthetic_yolo_v1/staff_crops/{style_id}/" + "{filename}"
            )

            try:
                entries, report = process_page(
                    page_id=page_id,
                    page_image_path=png_path,
                    oracle_label_path=label_path,
                    yolo_model=yolo,
                    token_lookup=token_lookup,
                    out_crops_dir=out_crops_dir,
                    crop_path_template=crop_path_template,
                    iou_threshold=args.iou_threshold,
                    imgsz=args.imgsz,
                    conf=args.conf,
                )
            except Exception as e:
                skipped_pages.append(f"{page_id} (exception: {e})")
                continue

            for k in ("yolo_boxes", "oracle_staves", "matches", "dropped_yolo_fp", "dropped_oracle_recall_gap", "dropped_token_miss", "dropped_degenerate"):
                totals[k] += report[k]
            if report["matches"] > 0:
                n_pages_with_match += 1
            for entry in entries:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if i % 250 == 0:
                retention = totals["matches"] / max(1, totals["oracle_staves"])
                print(f"  {i}/{len(pages)}  retention so far: {retention:.4f}", flush=True)

    retention = totals["matches"] / max(1, totals["oracle_staves"])
    report_out = {
        "pages_total": len(pages),
        "pages_with_match": n_pages_with_match,
        "pages_skipped": len(skipped_pages),
        **dict(totals),
        "retention": retention,
        "manifest_path": str(out_manifest_path),
        "skipped_pages_sample": skipped_pages[:50],
    }
    out_report_path.write_text(json.dumps(report_out, indent=2))
    print(json.dumps(report_out, indent=2), flush=True)

    if retention < 0.95:
        print(f"WARNING: retention {retention:.4f} < 0.95 — investigate before training!", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
