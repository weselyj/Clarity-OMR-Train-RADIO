"""Measure current YOLO Stage A baseline on lieder eval set.

Per piece: YOLO detects individual staves on page 1. We deduplicate
overlapping detections (IoU > 0.3), then compute how many staves are
detected vs expected.

Missing-count heuristic (no system clustering required):
  - expected staves per system = len(music21_score.parts)
  - complete_systems = detected_deduped // expected
  - remaining       = detected_deduped  % expected
  - min_staves_in_a_system = remaining  if remaining > 0 else expected
  - missing_count          = max(0, expected - min_staves_in_a_system)

This correctly flags pages where the deduped detection total is not a
multiple of expected_parts (i.e. at least one system is short at least
one stave). It under-counts when multiple systems each miss staves, but
that is acceptable for a rough pre-rebuild gap baseline.

Columns:
  piece                 — lc-ID stem
  expected_p1_staves    — len(music21_score.parts)
  detected_raw_p1       — YOLO bbox count before dedup
  detected_deduped_p1   — YOLO bbox count after IoU dedup
  complete_systems      — detected_deduped // expected
  missing_count         — max(0, expected - (detected_deduped % expected))
                          when remainder > 0; else 0

PDF rendering: PyMuPDF (fitz) v1.27 — used instead of pdf2image because
poppler is not available on this Windows host.

MXL lookup: scores nested as scores/<Composer>/<Set>/<Song>/<lc_id>.mxl;
built with rglob at startup.

Metric limitations:
  - expected_p1_staves is a proxy; grand-staff parts in this corpus are
    consistently split into 2 Parts by music21, so the count is reliable.
  - missing_count uses modular arithmetic and may under-count when many
    systems each miss staves — treat it as a lower-bound estimate.
  - IoU dedup threshold 0.3 was calibrated against lc28688206; adjacent
    staves with high vertical overlap may be incorrectly merged.
"""
import csv
from pathlib import Path

import fitz  # PyMuPDF
import music21
from PIL import Image
from ultralytics import YOLO

YOLO_WEIGHTS = r"C:\Users\Jonathan Wesely\Clarity-OMR\info\yolo.pt"
EVAL_PDFS = Path(r"data/openscore_lieder/eval_pdfs")
SCORES_ROOT = Path(r"data/openscore_lieder/scores")
OUT_CSV = Path("eval/results/baseline_pre_rebuild/stage_a_recall.csv")

DPI = 300
CONF = 0.25
IMGSZ = 1920

# IoU threshold for deduplication of overlapping stave boxes
IOU_DEDUP_THRESHOLD = 0.3


def build_mxl_index(root: Path) -> dict[str, Path]:
    return {f.stem: f for f in root.rglob("*.mxl")}


def expected_staves_per_system(mxl_path: Path) -> int:
    """Return music21 part count — reliable proxy for staves-per-system
    in this corpus since piano parts are split into 2 Parts by MuseScore."""
    s = music21.converter.parse(str(mxl_path))
    return len(s.parts)


def render_page1_pil(pdf_path: Path, dpi: int = DPI) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def iou(a, b):
    """IoU between two (x1,y1,x2,y2) bboxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedup_count(boxes, iou_thresh: float = IOU_DEDUP_THRESHOLD) -> int:
    """Greedy NMS: suppress lower-confidence boxes with IoU >= iou_thresh.
    Returns count of surviving boxes."""
    entries = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf)
        entries.append((x1, y1, x2, y2, conf))
    entries.sort(key=lambda e: e[4], reverse=True)
    suppressed = set()
    kept = 0
    for i, e in enumerate(entries):
        if i in suppressed:
            continue
        kept += 1
        for j in range(i + 1, len(entries)):
            if j not in suppressed:
                if iou(e[:4], entries[j][:4]) >= iou_thresh:
                    suppressed.add(j)
    return kept


def compute_missing(detected_deduped: int, expected: int) -> tuple[int, int, int]:
    """Return (complete_systems, leftover_staves, missing_count).

    complete_systems = detected_deduped // expected
    leftover         = detected_deduped  % expected
    missing_count    = max(0, expected - leftover) when leftover > 0, else 0
    """
    if expected == 0:
        return 0, 0, 0
    complete = detected_deduped // expected
    leftover = detected_deduped % expected
    if leftover == 0:
        missing = 0
    else:
        missing = expected - leftover
    return complete, leftover, missing


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Building MXL index...")
    mxl_index = build_mxl_index(SCORES_ROOT)
    print(f"  Found {len(mxl_index)} MXL files")

    print("Loading YOLO model...")
    model = YOLO(YOLO_WEIGHTS)

    rows = []
    skipped = []
    pdfs = sorted(EVAL_PDFS.glob("*.pdf"))
    print(f"Processing {len(pdfs)} eval PDFs...")

    for i, pdf in enumerate(pdfs, 1):
        piece = pdf.stem
        mxl = mxl_index.get(piece)
        if mxl is None:
            print(f"  [{i}/{len(pdfs)}] SKIP {piece} — no matching MXL")
            skipped.append(piece)
            continue
        try:
            expected = expected_staves_per_system(mxl)
            img = render_page1_pil(pdf)
            res = model.predict(img, conf=CONF, imgsz=IMGSZ, verbose=False)
            raw = len(res[0].boxes)
            deduped = dedup_count(res[0].boxes)
            complete, leftover, missing = compute_missing(deduped, expected)
            rows.append((piece, expected, raw, deduped, complete, leftover, missing))
            flag = " ***" if missing > 0 else ""
            print(
                f"  [{i}/{len(pdfs)}] {piece}: "
                f"exp={expected} raw={raw} deduped={deduped} "
                f"complete_sys={complete} leftover={leftover} "
                f"missing={missing}{flag}"
            )
        except Exception as exc:
            import traceback
            print(f"  [{i}/{len(pdfs)}] ERROR {piece}: {exc}")
            traceback.print_exc()
            skipped.append(piece)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "piece",
            "expected_p1_staves",
            "detected_raw_p1",
            "detected_deduped_p1",
            "complete_systems",
            "leftover_staves",
            "missing_count",
        ])
        w.writerows(rows)

    total_missing = sum(r[6] for r in rows)
    pieces_with_gap = sum(1 for r in rows if r[6] > 0)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    print(f"Pieces with missing_count > 0: {pieces_with_gap} / {len(rows)}")
    print(f"Total missing (lower-bound): {total_missing}")
    if skipped:
        print(f"Skipped {len(skipped)} pieces: {skipped}")


if __name__ == "__main__":
    main()
