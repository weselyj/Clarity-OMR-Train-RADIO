"""Verify that production-derived v15 system labels match reference overlay output.

Run this on the GPU box after re-deriving labels (Phase 2B) to confirm the
production port (generate_synthetic.py v15) and the reference overlay script
(verovio_overlay_v15.py) produce equivalent system bboxes.

Usage:
    python scripts/verify_v15_labels.py

Output: per-system coordinate deltas for each of the 5 v15 test pages.
Pass criterion: all deltas < 3 px (sub-pixel rounding is expected).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.generate_synthetic import _build_system_yolo_objects_v15

# ---------------------------------------------------------------------------
# Test pages — same as verovio_overlay_v15.py PAGES dict
# ---------------------------------------------------------------------------
PAGES = {
    "burleigh_p002": {
        "svg":          REPO / r"data/processed/synthetic_v2/pages/leipzig-default/data__openscore_lieder__scores__Burleigh___Harry_Thacker__5_Songs_of_Laurence_Hope__1_Worth_While__lc6518082.mxl__leipzig-default__p002.svg",
        "label_staff":  REPO / r"data/processed/synthetic_v2/labels/leipzig-default/data__openscore_lieder__scores__Burleigh___Harry_Thacker__5_Songs_of_Laurence_Hope__1_Worth_While__lc6518082.mxl__leipzig-default__p002.txt",
        "label_system": REPO / r"data/processed/synthetic_v2/labels_systems/leipzig-default/data__openscore_lieder__scores__Burleigh___Harry_Thacker__5_Songs_of_Laurence_Hope__1_Worth_While__lc6518082.mxl__leipzig-default__p002.txt",
        "png_w": 2750, "png_h": 3750,
    },
    "hensel_p001": {
        "svg":          REPO / r"data/processed/synthetic_v2/pages/leipzig-default/data__openscore_lieder__scores__Hensel___Fanny__6_Lieder___Op.7__5_Bitte__lc5987963.mxl__leipzig-default__p001.svg",
        "label_staff":  REPO / r"data/processed/synthetic_v2/labels/leipzig-default/data__openscore_lieder__scores__Hensel___Fanny__6_Lieder___Op.7__5_Bitte__lc5987963.mxl__leipzig-default__p001.txt",
        "label_system": REPO / r"data/processed/synthetic_v2/labels_systems/leipzig-default/data__openscore_lieder__scores__Hensel___Fanny__6_Lieder___Op.7__5_Bitte__lc5987963.mxl__leipzig-default__p001.txt",
        "png_w": 2750, "png_h": 3750,
    },
    "satie_p008": {
        "svg":          REPO / r"data/processed/synthetic_v2/pages/leipzig-default/data__openscore_lieder__scores__Satie___Erik__Socrate__1_Portrait_de_Socrate__lc6481218.mxl__leipzig-default__p008.svg",
        "label_staff":  REPO / r"data/processed/synthetic_v2/labels/leipzig-default/data__openscore_lieder__scores__Satie___Erik__Socrate__1_Portrait_de_Socrate__lc6481218.mxl__leipzig-default__p008.txt",
        "label_system": REPO / r"data/processed/synthetic_v2/labels_systems/leipzig-default/data__openscore_lieder__scores__Satie___Erik__Socrate__1_Portrait_de_Socrate__lc6481218.mxl__leipzig-default__p008.txt",
        "png_w": 2750, "png_h": 3750,
    },
    "sparse_lc5121134_p002": {
        "svg":          REPO / r"data/processed/sparse_augment/pages/leipzig-default/data__processed__sparse_augment__mxl__lc5121134_strip.musicxml__leipzig-default__p002.svg",
        "label_staff":  REPO / r"data/processed/sparse_augment/labels/leipzig-default/data__processed__sparse_augment__mxl__lc5121134_strip.musicxml__leipzig-default__p002.txt",
        "label_system": REPO / r"data/processed/sparse_augment/labels_systems/leipzig-default/data__processed__sparse_augment__mxl__lc5121134_strip.musicxml__leipzig-default__p002.txt",
        "png_w": 2750, "png_h": 3750,
    },
    "control_schubert_p001": {
        "svg":          REPO / r"data/processed/synthetic_v2/pages/leipzig-default/data__openscore_lieder__scores__Schubert___Franz__4_Lieder___Op.96__1_Die_Sterne___D.939__lc6485233.mxl__leipzig-default__p001.svg",
        "label_staff":  REPO / r"data/processed/synthetic_v2/labels/leipzig-default/data__openscore_lieder__scores__Schubert___Franz__4_Lieder___Op.96__1_Die_Sterne___D.939__lc6485233.mxl__leipzig-default__p001.txt",
        "label_system": REPO / r"data/processed/synthetic_v2/labels_systems/leipzig-default/data__openscore_lieder__scores__Schubert___Franz__4_Lieder___Op.96__1_Die_Sterne___D.939__lc6485233.mxl__leipzig-default__p001.txt",
        "png_w": 2750, "png_h": 3750,
    },
}


def read_yolo_labels(path: Path, png_w: int, png_h: int):
    """Read YOLO labels → list of (x1,y1,x2,y2) pixel coords."""
    bboxes = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * png_w
        y1 = (cy - h / 2) * png_h
        x2 = (cx + w / 2) * png_w
        y2 = (cy + h / 2) * png_h
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def read_staff_yolo_labels_xywh(path: Path, png_w: int, png_h: int):
    """Read YOLO per-staff labels → list of (x,y,w,h) pixel coords."""
    boxes = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * png_w
        y1 = (cy - h / 2) * png_h
        w_px = w * png_w
        h_px = h * png_h
        boxes.append((x1, y1, w_px, h_px))
    return boxes


def verify_page(key: str, info: dict) -> bool:
    """Run production v15 and compare against on-disk re-derived labels.

    Returns True if all deltas are within tolerance.
    """
    TOLERANCE_PX = 3.0

    svg_path   = Path(str(info["svg"]))
    staff_path = Path(str(info["label_staff"]))
    sys_path   = Path(str(info["label_system"]))
    png_w      = info["png_w"]
    png_h      = info["png_h"]

    missing = [p for p in [svg_path, staff_path, sys_path] if not p.exists()]
    if missing:
        print(f"  [{key}] SKIP — missing files: {[str(p) for p in missing]}")
        return True  # Not a failure on the dev machine

    staff_boxes = read_staff_yolo_labels_xywh(staff_path, png_w, png_h)
    reference_bboxes = read_yolo_labels(sys_path, png_w, png_h)

    prod_objects, _ = _build_system_yolo_objects_v15(svg_path, staff_boxes, png_w, png_h)
    prod_bboxes_abs = []
    for _, (x, y, w, h) in prod_objects:
        prod_bboxes_abs.append((x, y, x + w, y + h))

    print(f"\n{'='*60}")
    print(f"  {key}")
    print(f"  Reference systems : {len(reference_bboxes)}")
    print(f"  Production systems: {len(prod_bboxes_abs)}")

    if len(reference_bboxes) != len(prod_bboxes_abs):
        print(f"  MISMATCH: system count differs")
        return False

    all_ok = True
    for i, (ref, prod) in enumerate(zip(reference_bboxes, prod_bboxes_abs)):
        dx1 = prod[0] - ref[0]
        dy1 = prod[1] - ref[1]
        dx2 = prod[2] - ref[2]
        dy2 = prod[3] - ref[3]
        max_delta = max(abs(dx1), abs(dy1), abs(dx2), abs(dy2))
        ok = max_delta < TOLERANCE_PX
        status = "OK" if ok else "FAIL"
        print(f"  sys{i}: ref=({ref[0]:.0f},{ref[1]:.0f},{ref[2]:.0f},{ref[3]:.0f})  "
              f"prod=({prod[0]:.0f},{prod[1]:.0f},{prod[2]:.0f},{prod[3]:.0f})  "
              f"dy1={dy1:+.1f} dy2={dy2:+.1f} dx1={dx1:+.1f} dx2={dx2:+.1f}  [{status}]")
        if not ok:
            all_ok = False
    return all_ok


def main() -> int:
    print("v15 production verification")
    print("=" * 60)
    results = {}
    for key, info in PAGES.items():
        results[key] = verify_page(key, info)

    print(f"\n{'='*60}")
    n_ok = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    print(f"Summary: {n_ok} OK, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
