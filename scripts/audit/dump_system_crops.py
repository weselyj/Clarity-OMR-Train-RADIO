"""Dump Stage A (YOLO) system crops for a single page image.

Debug aid for diagnosing Stage-B misreads (e.g. bass-clef-as-treble): isolates
whether a wrong prediction is caused by a bad Stage-A crop (wrong region /
clipped staff) versus the Stage-B decoder mis-decoding a correct crop.

Uses the exact same detector the inference pipeline uses
(`YoloStageASystems`, same default weights + conf + bbox_extended brace-margin
logic), so the dumped crops are byte-for-region-identical to what Stage B sees.

Outputs into <out-dir>:
  - system_<idx>_conf<c>.png   one crop per detected system (decode input)
  - _page_annotated.png        full page with detected bboxes drawn + indexed
  - _summary.txt               per-system bbox, conf, crop dimensions

Run on seder (where the YOLO weights + scan live):
  venv-cu132/Scripts/python.exe scripts/audit/dump_system_crops.py \
    "C:/Users/Jonathan Wesely/Downloads/Scanned_20251208-0833.jpg" \
    C:/tmp/bethlehem_crops
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from src.models.yolo_stage_a_systems import YoloStageASystems  # noqa: E402

_DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "runs/detect/runs/yolo26m_systems/weights/best.pt"


def dump_crops(
    image_path: Path,
    out_dir: Path,
    yolo_weights: Path,
    conf: float = 0.25,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Match pipeline.run_image: open as RGB.
    page = Image.open(str(image_path)).convert("RGB")

    stage_a = YoloStageASystems(yolo_weights, conf=conf)
    systems = stage_a.detect_systems(page)

    annotated = page.copy()
    draw = ImageDraw.Draw(annotated)
    summary_lines = [
        f"image: {image_path}",
        f"page size (WxH): {page.size[0]}x{page.size[1]}",
        f"yolo weights: {yolo_weights}",
        f"conf: {conf}",
        f"systems detected: {len(systems)}",
        "",
    ]

    for sys_dict in systems:
        idx = sys_dict["system_index"]
        x1, y1, x2, y2 = (int(v) for v in sys_dict["bbox_extended"])
        c = float(sys_dict["conf"])
        crop = page.crop((x1, y1, x2, y2))
        crop_name = f"system_{idx:02d}_conf{c:.3f}.png"
        crop.save(out_dir / crop_name)

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        draw.text((x1 + 6, y1 + 6), f"#{idx} ({c:.2f})", fill=(255, 0, 0))

        line = (
            f"system {idx}: bbox=({x1},{y1},{x2},{y2}) "
            f"crop={x2 - x1}x{y2 - y1} conf={c:.4f} -> {crop_name}"
        )
        summary_lines.append(line)
        print(line)

    annotated.save(out_dir / "_page_annotated.png")
    (out_dir / "_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(systems)} crops + annotated page to {out_dir}")
    return systems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="Page image (scan / JPG / PNG)")
    ap.add_argument("out_dir", type=Path, help="Directory for crop outputs")
    ap.add_argument("--yolo-weights", type=Path, default=_DEFAULT_YOLO_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="YOLO confidence threshold (pipeline default 0.25)")
    args = ap.parse_args()
    dump_crops(args.image, args.out_dir, args.yolo_weights, args.conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
