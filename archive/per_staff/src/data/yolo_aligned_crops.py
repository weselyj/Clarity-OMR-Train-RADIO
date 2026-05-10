"""YOLO-aligned crop extraction for RADIO Stage 3 training (per-staff, archived).

Replaces oracle Verovio bboxes with YOLO predictions to close the train/eval
distribution mismatch on staff crops. Per the design spec
docs/superpowers/specs/2026-05-01-radio-stage3-yolo-aligned-design.md, the
α-policy applies: staves YOLO misses are dropped from training.

Archived 2026-05-10 alongside the rest of the per-staff path; the live
per-system pipeline lives in ``src/data/yolo_aligned_systems.py`` and shares
``iou_xyxy`` + ``_yolo_predict_to_boxes`` via ``src/data/yolo_common.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from src.data.yolo_common import _yolo_predict_to_boxes, iou_xyxy


def match_yolo_to_oracle(
    yolo_boxes: Iterable[dict],
    oracle_staves: Iterable[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Match each YOLO prediction to its best-IoU oracle staff.

    α-policy: drop YOLO boxes that don't reach `iou_threshold` against any
    oracle (false positives) and oracle staves that no YOLO box matches
    (recall gaps).

    When two YOLO boxes match the same oracle, keep the higher-confidence one.

    Each yolo_box dict must contain: yolo_idx, bbox (x1,y1,x2,y2), conf.
    Each oracle dict must contain: staff_index, bbox (x1,y1,x2,y2).

    Returns a list of matches, each: {yolo_idx, staff_index, conf, iou,
    yolo_bbox, oracle_bbox}.
    """
    yolo_list = list(yolo_boxes)
    oracle_list = list(oracle_staves)
    candidates = []
    for y in yolo_list:
        best_oracle = None
        best_iou = 0.0
        for o in oracle_list:
            i = iou_xyxy(y["bbox"], o["bbox"])
            if i > best_iou:
                best_iou = i
                best_oracle = o
        if best_oracle is not None and best_iou >= iou_threshold:
            candidates.append({
                "yolo_idx": y["yolo_idx"],
                "staff_index": best_oracle["staff_index"],
                "conf": y["conf"],
                "iou": best_iou,
                "yolo_bbox": y["bbox"],
                "oracle_bbox": best_oracle["bbox"],
            })
    # Resolve duplicates: when multiple YOLO boxes match the same oracle, keep highest conf.
    by_oracle: dict[int, dict] = {}
    for c in candidates:
        sid = c["staff_index"]
        if sid not in by_oracle or c["conf"] > by_oracle[sid]["conf"]:
            by_oracle[sid] = c
    return sorted(by_oracle.values(), key=lambda c: c["staff_index"])


def load_oracle_bboxes_from_yolo_label(
    label_path: Path, page_width: int, page_height: int
) -> list[dict]:
    """Read a YOLO-format label file and return oracle bboxes in pixel xyxy.

    YOLO format: each line is `class cx cy w h`, all normalized to [0,1].
    `staff_index` is assigned by sorting on y_center top→bottom (matches
    the convention used by synthetic_token_manifest.jsonl).
    """
    rows = []
    text = Path(label_path).read_text() if Path(label_path).exists() else ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # class_id, cx, cy, w, h
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2) * page_width
        y1 = (cy - h / 2) * page_height
        x2 = (cx + w / 2) * page_width
        y2 = (cy + h / 2) * page_height
        rows.append({"y_center": cy, "bbox": (x1, y1, x2, y2)})
    rows.sort(key=lambda r: r["y_center"])
    return [{"staff_index": i, "bbox": r["bbox"]} for i, r in enumerate(rows)]


def _yolo_predict_to_boxes(yolo_model, page_image: Image.Image, imgsz: int = 1920, conf: float = 0.25) -> list[dict]:
    """Adapter: ultralytics YOLO .predict() → list of {yolo_idx, bbox, conf} dicts."""
    results = yolo_model.predict(page_image, imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []
    r = results[0]
    xyxy = r.boxes.xyxy
    confs = r.boxes.conf
    # Tolerate either torch tensors or plain lists (latter is what tests pass).
    def to_list(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)
    xyxy_list = to_list(xyxy)
    conf_list = to_list(confs)
    out = []
    for i, (box, c) in enumerate(zip(xyxy_list, conf_list)):
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        out.append({"yolo_idx": i, "bbox": (x1, y1, x2, y2), "conf": float(c)})
    return out


def process_page(
    page_id: str,
    page_image_path: Path,
    oracle_label_path: Path,
    yolo_model,
    token_lookup: dict,
    out_crops_dir: Path,
    crop_path_template: str,
    iou_threshold: float = 0.5,
    imgsz: int = 1920,
    conf: float = 0.25,
) -> tuple[list[dict], dict]:
    """Run YOLO on page, match to oracle, crop, return manifest entries + report.

    `token_lookup` maps (page_id, staff_index) → token-manifest entry dict
    that this function copies fields from (everything except image_path/sample_id).
    `crop_path_template` must contain `{filename}` and is the value written to
    each manifest entry's image_path field (relative or absolute, caller's choice).
    Crops are written under `out_crops_dir` with filename `<page_id>__yoloidx<NN>.png`.

    The returned report dict contains:
      - page_id: the page identifier
      - yolo_boxes: total YOLO predictions on this page
      - oracle_staves: total oracle staves in the label file
      - matches: number of YOLO↔oracle pairs that passed IoU threshold
      - dropped_yolo_fp: YOLO boxes that didn't match any oracle (false positives)
      - dropped_oracle_recall_gap: oracle staves that no YOLO box matched (recall gaps)
      - dropped_token_miss: matched staves skipped because (page_id, staff_index)
        had no entry in token_lookup
      - dropped_degenerate: matched staves skipped because the YOLO bbox clipped
        to an empty region (x2 <= x1 or y2 <= y1)
    """
    out_crops_dir = Path(out_crops_dir)
    out_crops_dir.mkdir(parents=True, exist_ok=True)

    page_image = Image.open(page_image_path).convert("RGB")
    page_w, page_h = page_image.size
    oracles = load_oracle_bboxes_from_yolo_label(oracle_label_path, page_w, page_h)
    yolo_boxes = _yolo_predict_to_boxes(yolo_model, page_image, imgsz=imgsz, conf=conf)
    matches = match_yolo_to_oracle(yolo_boxes, oracles, iou_threshold=iou_threshold)

    matched_staff_indices = {m["staff_index"] for m in matches}
    matched_yolo_indices = {m["yolo_idx"] for m in matches}

    dropped_token_miss = 0
    dropped_degenerate = 0
    manifest_entries: list[dict] = []
    for m in matches:
        sid = m["staff_index"]
        token_entry = token_lookup.get((page_id, sid))
        if token_entry is None:
            # Oracle has a bbox in the label file but no token entry. Skip — log via report.
            dropped_token_miss += 1
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in m["yolo_bbox"])
        # Clip to page bounds.
        x1 = max(0, min(x1, page_w))
        x2 = max(0, min(x2, page_w))
        y1 = max(0, min(y1, page_h))
        y2 = max(0, min(y2, page_h))
        if x2 <= x1 or y2 <= y1:
            dropped_degenerate += 1
            continue
        crop = page_image.crop((x1, y1, x2, y2))
        filename = f"{page_id}__yoloidx{m['yolo_idx']:02d}.png"
        crop_path = out_crops_dir / filename
        crop.save(crop_path)
        entry = {
            "sample_id": f"{page_id}__yoloidx{m['yolo_idx']:02d}",
            "dataset": token_entry.get("dataset", "synthetic_fullpage"),
            "split": token_entry.get("split", "train"),
            "image_path": crop_path_template.format(filename=filename),
            "page_id": token_entry["page_id"],
            "source_path": token_entry["source_path"],
            "style_id": token_entry["style_id"],
            "page_number": token_entry["page_number"],
            "staff_index": sid,
            "source_format": token_entry.get("source_format", "musicxml"),
            "score_type": token_entry.get("score_type", "piano"),
            "token_sequence": token_entry["token_sequence"],
            "token_count": token_entry["token_count"],
        }
        manifest_entries.append(entry)

    report = {
        "page_id": page_id,
        "yolo_boxes": len(yolo_boxes),
        "oracle_staves": len(oracles),
        "matches": len(matches),
        "dropped_yolo_fp": len(yolo_boxes) - len(matched_yolo_indices),
        "dropped_oracle_recall_gap": len(oracles) - len(matched_staff_indices),
        "dropped_token_miss": dropped_token_miss,
        "dropped_degenerate": dropped_degenerate,
    }
    return manifest_entries, report
