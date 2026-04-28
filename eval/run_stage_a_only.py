"""Lightweight Stage-A-only evaluation: PDF -> YOLO -> bbox manifest JSONL.

Skips Stage B/C entirely. Used to iterate on YOLO retrains without paying the
full pipeline cost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from PIL import Image
from ultralytics import YOLO


def _render_pages(pdf_path: Path, dpi: int) -> List[Image.Image]:
    """Render all pages of a PDF to PIL images at the given DPI.

    Uses PyMuPDF (fitz) — pure Python on Windows, no poppler dependency.
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    images: List[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def run_stage_a(
    pdf_path: Path,
    yolo_weights: Path,
    out_dir: Path,
    dpi: int = 300,
    conf: float = 0.25,
    imgsz: int = 1920,
) -> None:
    """Render a PDF, run YOLO on each page, write a JSONL manifest of bboxes.

    Output file: out_dir / f"{pdf_path.stem}_stage_a.jsonl"
    Each line: {"piece", "page", "bbox_xyxy", "conf", "page_width", "page_height"}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(yolo_weights))
    pages = _render_pages(pdf_path, dpi)

    manifest_path = out_dir / f"{pdf_path.stem}_stage_a.jsonl"
    with manifest_path.open("w") as f:
        for page_idx, img in enumerate(pages):
            results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
            for r in results:
                xyxy = r.boxes.xyxy.tolist() if hasattr(r.boxes.xyxy, "tolist") else list(r.boxes.xyxy)
                confs = r.boxes.conf.tolist() if hasattr(r.boxes.conf, "tolist") else list(r.boxes.conf)
                for box, conf_score in zip(xyxy, confs):
                    x1, y1, x2, y2 = box
                    record = {
                        "piece": pdf_path.stem,
                        "page": page_idx,
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(conf_score),
                        "page_width": img.width,
                        "page_height": img.height,
                    }
                    f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1920)
    args = parser.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    print(f"Running Stage A on {len(pdfs)} PDFs")
    for pdf in pdfs:
        run_stage_a(
            pdf, args.yolo_weights, args.out_dir,
            dpi=args.dpi, conf=args.conf, imgsz=args.imgsz,
        )
        print(f"  done: {pdf.stem}")


if __name__ == "__main__":
    main()
