"""Multi-staff system-level helpers for Stage 3 data prep.

Operates on system bboxes (one label per system, multiple staves per
system). Shared geometry + ultralytics adapters live in
`src/data/yolo_common.py`; the archived per-staff variant is at
`archive/per_staff/src/data/yolo_aligned_crops.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.data.yolo_common import iou_xyxy, _yolo_predict_to_boxes
from src.tokenizer.vocab import STAFF_INDEX_MARKER_TOKENS


def load_oracle_system_bboxes(
    label_path: Path, page_width: int, page_height: int
) -> list[dict]:
    """Read YOLO-format system label file → list of `{system_index, bbox}` dicts.

    YOLO format: each line is `class cx cy w h`, normalized to [0, 1].
    Output sorted top-to-bottom by y_center (matches the convention used by
    the companion `<page>.staves.json` file).
    """
    rows: list[dict] = []
    text = Path(label_path).read_text() if Path(label_path).exists() else ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2) * page_width
        y1 = (cy - h / 2) * page_height
        x2 = (cx + w / 2) * page_width
        y2 = (cy + h / 2) * page_height
        rows.append({"y_center": cy, "bbox": (x1, y1, x2, y2)})
    rows.sort(key=lambda r: r["y_center"])
    return [{"system_index": i, "bbox": r["bbox"]} for i, r in enumerate(rows)]


def load_staves_per_system(json_path: Path) -> list[int]:
    """Read the companion `<page>.staves.json` → list of integers (staves per system)."""
    p = Path(json_path)
    if not p.exists():
        return []
    return list(json.loads(p.read_text()))


def staff_indices_for_system(system_index: int, staves_per_system: list[int]) -> list[int]:
    """Return the per-staff indices that fall within `system_index`.

    Per-staff indices are page-global, top-to-bottom. Cumulative sum of
    `staves_per_system` gives each system's starting staff index.
    """
    if system_index < 0 or system_index >= len(staves_per_system):
        return []
    start = sum(staves_per_system[:system_index])
    count = staves_per_system[system_index]
    return list(range(start, start + count))


def match_yolo_to_oracle_systems(
    yolo_boxes: Iterable[dict],
    oracle_systems: Iterable[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Match each YOLO system prediction to its best-IoU oracle system.

    α-policy: drop YOLO boxes that don't reach `iou_threshold` against any
    oracle (false positives) and oracle systems that no YOLO box matches
    (recall gaps). When two YOLO boxes match the same oracle, keep the
    higher-confidence one.

    Each yolo_box dict must contain: yolo_idx, bbox (x1,y1,x2,y2), conf.
    Each oracle dict must contain: system_index, bbox.

    Returns: sorted list of `{yolo_idx, system_index, conf, iou, yolo_bbox, oracle_bbox}`.
    """
    yolo_list = list(yolo_boxes)
    oracle_list = list(oracle_systems)
    candidates: list[dict] = []
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
                "system_index": best_oracle["system_index"],
                "conf": y["conf"],
                "iou": best_iou,
                "yolo_bbox": y["bbox"],
                "oracle_bbox": best_oracle["bbox"],
            })
    by_oracle: dict[int, dict] = {}
    for c in candidates:
        sid = c["system_index"]
        if sid not in by_oracle or c["conf"] > by_oracle[sid]["conf"]:
            by_oracle[sid] = c
    return sorted(by_oracle.values(), key=lambda c: c["system_index"])


def assemble_multi_staff_tokens(per_staff_token_sequences: list[list[str]]) -> list[str]:
    """Concatenate per-staff sequences with `<staff_idx_N>` markers per staff.

    Each input must follow the per-staff manifest convention:
        <bos> <staff_start> [content] <staff_end> <eos>

    Output:
        <bos> <staff_start> <staff_idx_0> [c_0] <staff_end>
              <staff_start> <staff_idx_1> [c_1] <staff_end> ... <eos>

    Raises ValueError if N > number of available marker tokens (8 in vocab v2).
    Raises AssertionError if a per-staff sequence doesn't have the expected
    wrapper structure.
    """
    if len(per_staff_token_sequences) > len(STAFF_INDEX_MARKER_TOKENS):
        raise ValueError(
            f"system has {len(per_staff_token_sequences)} staves but vocab v2 only "
            f"defines {len(STAFF_INDEX_MARKER_TOKENS)} marker tokens"
        )

    out: list[str] = ["<bos>"]
    for idx, seq in enumerate(per_staff_token_sequences):
        assert len(seq) >= 4, f"per-staff sequence too short: {seq[:4]}"
        assert seq[0] == "<bos>", f"expected <bos> at start, got {seq[0]}"
        assert seq[1] == "<staff_start>", f"expected <staff_start> at index 1, got {seq[1]}"
        assert seq[-2] == "<staff_end>", f"expected <staff_end> at index -2, got {seq[-2]}"
        assert seq[-1] == "<eos>", f"expected <eos> at end, got {seq[-1]}"
        content = seq[2:-2]
        out.append("<staff_start>")
        out.append(STAFF_INDEX_MARKER_TOKENS[idx])
        out.extend(content)
        out.append("<staff_end>")
    out.append("<eos>")
    return out


def process_page_systems(
    page_id: str,
    page_image_path: Path,
    oracle_label_path: Path,
    oracle_staves_json_path: Path,
    yolo_model,
    token_lookup: dict,
    out_crops_dir: Path,
    crop_path_template: str,
    iou_threshold: float = 0.5,
    imgsz: int = 1920,
    conf: float = 0.25,
    dataset_name: str = "synthetic_systems",
) -> tuple[list[dict], dict]:
    """Multi-staff variant of the per-staff `process_page`
    (archived at `archive/per_staff/src/data/yolo_aligned_crops.py`).

    Workflow:
      1. Load page image dimensions, oracle system bboxes, staves-per-system list
      2. Run YOLO → system bbox predictions
      3. Match YOLO ↔ oracle systems by IoU
      4. For each matched system, look up per-staff token entries (cumsum over staves_per_system)
      5. Assemble multi-staff token sequence with `<staff_idx_N>` markers
      6. Crop the YOLO bbox region of the page, save as `<page_id>__sys{NN}.png`
      7. Emit one manifest entry per matched system

    Drops recorded in the report (no exceptions raised):
      - YOLO false positives (no oracle match)
      - Oracle recall gaps (YOLO missed a system)
      - Token-lookup misses (any staff in a matched system without per-staff tokens)
      - Degenerate crops (clipped to empty region)
      - Marker overflow (system has > 8 staves; vocab limit)
    """
    out_crops_dir = Path(out_crops_dir)
    out_crops_dir.mkdir(parents=True, exist_ok=True)

    page_image = Image.open(page_image_path).convert("RGB")
    page_w, page_h = page_image.size
    oracles = load_oracle_system_bboxes(oracle_label_path, page_w, page_h)
    staves_per_system = load_staves_per_system(oracle_staves_json_path)
    if len(staves_per_system) != len(oracles):
        # The two files MUST agree in length; if they don't, treat as data error.
        return [], {
            "page_id": page_id,
            "yolo_boxes": 0,
            "oracle_systems": len(oracles),
            "matches": 0,
            "dropped_yolo_fp": 0,
            "dropped_oracle_recall_gap": 0,
            "dropped_token_miss": 0,
            "dropped_degenerate": 0,
            "dropped_marker_overflow": 0,
            "error": f"staves.json len {len(staves_per_system)} != labels.txt len {len(oracles)}",
        }
    yolo_boxes = _yolo_predict_to_boxes(yolo_model, page_image, imgsz=imgsz, conf=conf)
    matches = match_yolo_to_oracle_systems(yolo_boxes, oracles, iou_threshold=iou_threshold)

    matched_system_indices = {m["system_index"] for m in matches}
    matched_yolo_indices = {m["yolo_idx"] for m in matches}

    dropped_token_miss = 0
    dropped_degenerate = 0
    dropped_marker_overflow = 0
    manifest_entries: list[dict] = []

    for m in matches:
        sys_idx = m["system_index"]
        staff_indices = staff_indices_for_system(sys_idx, staves_per_system)

        # Look up per-staff tokens for this system
        per_staff_seqs = []
        per_staff_entries = []
        token_miss = False
        for s in staff_indices:
            entry = token_lookup.get((page_id, s))
            if entry is None:
                token_miss = True
                break
            per_staff_seqs.append(entry["token_sequence"])
            per_staff_entries.append(entry)
        if token_miss:
            dropped_token_miss += 1
            continue

        # Assemble multi-staff sequence; trap marker overflow (ValueError) and
        # malformed per-staff wrappers (AssertionError from assemble_multi_staff_tokens)
        # as drops rather than crashes. The docstring contract is "no exceptions raised".
        try:
            multi_seq = assemble_multi_staff_tokens(per_staff_seqs)
        except (ValueError, AssertionError):
            dropped_marker_overflow += 1
            continue

        # Crop the YOLO bbox
        x1, y1, x2, y2 = (int(round(v)) for v in m["yolo_bbox"])
        x1 = max(0, min(x1, page_w))
        x2 = max(0, min(x2, page_w))
        y1 = max(0, min(y1, page_h))
        y2 = max(0, min(y2, page_h))
        if x2 <= x1 or y2 <= y1:
            dropped_degenerate += 1
            continue
        crop = page_image.crop((x1, y1, x2, y2))
        filename = f"{page_id}__sys{sys_idx:02d}.png"
        crop_path = out_crops_dir / filename
        crop.save(crop_path)

        # Compose manifest entry. Inherit metadata from the FIRST per-staff entry of the
        # matched system (page_id, style_id, etc. are page-level).
        ref = per_staff_entries[0]
        entry = {
            "sample_id": f"{dataset_name}:{page_id}__sys{sys_idx:02d}",
            "dataset": dataset_name,
            "split": ref.get("split", "train"),
            "image_path": crop_path_template.format(filename=filename),
            "page_id": ref["page_id"],
            "source_path": ref["source_path"],
            "style_id": ref["style_id"],
            "page_number": ref["page_number"],
            "system_index": sys_idx,
            "staves_in_system": len(staff_indices),
            "staff_indices": staff_indices,
            "source_format": ref.get("source_format", "musicxml"),
            "score_type": ref.get("score_type", "piano"),
            "token_sequence": multi_seq,
            "token_count": len(multi_seq),
        }
        manifest_entries.append(entry)

    report = {
        "page_id": page_id,
        "yolo_boxes": len(yolo_boxes),
        "oracle_systems": len(oracles),
        "matches": len(matches),
        "dropped_yolo_fp": len(yolo_boxes) - len(matched_yolo_indices),
        "dropped_oracle_recall_gap": len(oracles) - len(matched_system_indices),
        "dropped_token_miss": dropped_token_miss,
        "dropped_degenerate": dropped_degenerate,
        "dropped_marker_overflow": dropped_marker_overflow,
    }
    return manifest_entries, report
