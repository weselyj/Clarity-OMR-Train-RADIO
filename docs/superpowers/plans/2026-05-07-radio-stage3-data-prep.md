# Stage 3 Data Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the four token manifests Stage 3 training needs: `synthetic_systems_v1` (multi-staff system entries with marker tokens), `primus_systems` and `cameraprimus_systems` (single-staff entries with `<staff_idx_0>` markers), and a combined `token_manifest_stage3.jsonl` mixing the four sources at the design-spec ratios. Also moves the system-level YOLO weights to a canonical path.

**Architecture:** Mirror the existing `process_page` helper (`src/data/yolo_aligned_crops.py`) by adding a multi-staff `process_page_systems` variant: load oracle system bboxes + staves-per-system list, run system-level YOLO, IoU-match, derive each system's staff-index range from cumulative sums, look up per-staff token sequences, strip per-staff wrappers and re-assemble with `<staff_idx_N>` markers, crop the matched system bbox. Build scripts then iterate this over the synthetic_v2 corpus and over primus/cameraprimus per-staff entries.

**Tech Stack:** Python 3.11+, ultralytics YOLO, PIL, pytest. All code follows the existing repo patterns (argparse + audit JSON + JSONL manifest emit).

**Preconditions verified on GPU box (2026-05-07):**
- `data\processed\synthetic_v2\manifests\synthetic_token_manifest.jsonl` (258 MB, per-staff entries)
- `data\processed\synthetic_v2\labels_systems\<style>\<page>.txt` (YOLO-format system bboxes; 3 styles × 2514 pages)
- `data\processed\synthetic_v2\labels_systems\<style>\<page>.staves.json` (staves-per-system list per page)
- `data\processed\mixed_systems_v1\` (YOLO training set; not directly needed by this plan but verifies labels_systems is populated)
- `runs\detect\runs\detect\runs\yolo26m_systems\weights\best.pt` (mAP50 0.995, will be moved by Task 1)
- `src\data\manifests\token_manifest_full_systems.jsonl` (845 MB combined manifest with primus + cameraprimus + grandstaff/_systems)

---

## File Structure

**New files:**
- `src/data/yolo_aligned_systems.py` — `process_page_systems` helper, oracle loaders, multi-staff token assembly logic
- `scripts/build_synthetic_systems_v1.py` — drives `process_page_systems` over the synthetic_v2 corpus → `synthetic_systems_v1` manifest
- `scripts/retokenize_with_staff_markers.py` — generic single-staff retokenizer (primus + cameraprimus); writes `primus_systems` and `cameraprimus_systems` manifests
- `scripts/build_stage3_combined_manifest.py` — merges synthetic_systems_v1 + grandstaff_systems + primus_systems + cameraprimus_systems into `token_manifest_stage3.jsonl`
- `tests/data/test_yolo_aligned_systems.py` — unit tests for `src/data/yolo_aligned_systems.py`
- `tests/data/test_retokenize_with_staff_markers.py` — unit tests for the retokenizer
- `tests/data/test_build_stage3_combined_manifest.py` — unit tests for the combined-manifest builder

**Modified files:** none (everything additive — these scripts produce data, they don't change training code).

---

## Task 1: Move yolo26m_systems weights to canonical path

**Files:**
- Source on GPU box: `C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\runs\detect\runs\detect\runs\yolo26m_systems\` (entire run directory, ~50 MB including artifacts)
- Destination: `C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\runs\detect\runs\yolo26m_systems\` (sibling of existing `yolo26m_phase1_clean`, `yolo26m_phase2_noise`, `yolo8m_baseline_v1`)

This is an operational move on the GPU box. No Python code; pure file operations. Do this BEFORE writing any code that references the YOLO weights.

- [ ] **Step 1: Verify destination doesn't already exist**

Run on GPU box:
```cmd
ssh 10.10.1.29 "if exist \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\runs\detect\runs\yolo26m_systems\" (echo COLLISION) else (echo CLEAR)"
```
Expected: `CLEAR`. If `COLLISION`, halt and ask the user — there's something already at the destination that wasn't expected.

- [ ] **Step 2: Move the run directory**

```cmd
ssh 10.10.1.29 "move \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\runs\detect\runs\detect\runs\yolo26m_systems\" \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\runs\detect\runs\yolo26m_systems\""
```
Expected: `1 dir(s) moved.`

- [ ] **Step 3: Verify destination has all artifacts**

```cmd
ssh 10.10.1.29 "dir \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\runs\detect\runs\yolo26m_systems\weights\best.pt\""
```
Expected: file size matches the original (44,302,745 bytes).

- [ ] **Step 4: Verify the source path is now empty / can be cleaned up**

```cmd
ssh 10.10.1.29 "dir \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\runs\detect\runs\detect\""
```
Expected: empty `runs` subdir or no entries. If anything remains there other than the now-emptied `runs\` subdir, halt — there may be other artifacts that need to move.

- [ ] **Step 5: Optional — clean up empty intermediate dirs**

```cmd
ssh 10.10.1.29 "rmdir /s /q \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\runs\detect\runs\detect\""
```
Only run if Step 4 confirmed no artifacts remain. Document the operation in the next task's commit message rather than committing this directly (no repo state change).

- [ ] **Step 6: Update `project_radio_subproject1.md` memory note**

Edit `/home/ari/.claude/projects/-home-ari/memory/project_radio_subproject1.md`:
- Replace the "non-obvious nested path" section with the new canonical path: `runs\detect\runs\yolo26m_systems\weights\best.pt`
- Add a one-line "Path moved 2026-MM-DD" note for traceability.

No git commit for Task 1 — it's an operational move on the GPU box. Subsequent tasks reference the canonical path.

---

## Task 2: Test scaffolding — oracle loaders for systems

**Files:**
- Create: `src/data/yolo_aligned_systems.py`
- Create: `tests/data/test_yolo_aligned_systems.py`

The first piece is reading two files per page from `labels_systems/<style>/`:
- `<page>.txt` — YOLO-format system bboxes (`class cx cy w h`, normalized to `[0, 1]`)
- `<page>.staves.json` — JSON list of integers, length = number of systems, each int = staves in that system

Both are ordered top-to-bottom (matches `.txt` row order; verified on actual data: a 3-system page with `[3, 3, 3]` staves.json corresponds to 3 lines in the .txt).

- [ ] **Step 1: Write the failing test (oracle bbox loader)**

```python
# tests/data/test_yolo_aligned_systems.py
from pathlib import Path
import json
import pytest

from src.data.yolo_aligned_systems import (
    load_oracle_system_bboxes,
)


def _write_label_files(tmp_path: Path, txt_lines: list[str], staves_list: list[int]) -> tuple[Path, Path]:
    txt_path = tmp_path / "page.txt"
    json_path = tmp_path / "page.staves.json"
    txt_path.write_text("\n".join(txt_lines) + "\n")
    json_path.write_text(json.dumps(staves_list))
    return txt_path, json_path


def test_load_oracle_system_bboxes_three_systems(tmp_path: Path):
    txt, _ = _write_label_files(
        tmp_path,
        [
            "0 0.5 0.15 0.9 0.25",  # top system
            "0 0.5 0.4 0.99 0.25",  # middle
            "0 0.5 0.65 0.99 0.22",  # bottom
        ],
        [3, 3, 3],
    )
    out = load_oracle_system_bboxes(txt, page_width=1000, page_height=1500)
    assert len(out) == 3
    # Returned in y-sorted order (top→bottom). Check y_centers are ascending.
    y_centers = [(b["bbox"][1] + b["bbox"][3]) / 2 for b in out]
    assert y_centers == sorted(y_centers)
    # system_index assigned by sort order
    assert [b["system_index"] for b in out] == [0, 1, 2]
    # Bbox values are in pixel space
    top_bbox = out[0]["bbox"]
    assert 0 < top_bbox[0] < 1000  # x1
    assert 0 < top_bbox[1] < 1500  # y1
    assert top_bbox[0] < top_bbox[2]  # x1 < x2
    assert top_bbox[1] < top_bbox[3]  # y1 < y2


def test_load_oracle_system_bboxes_empty_label(tmp_path: Path):
    txt = tmp_path / "page.txt"
    txt.write_text("")
    out = load_oracle_system_bboxes(txt, page_width=1000, page_height=1500)
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_yolo_aligned_systems.py::test_load_oracle_system_bboxes_three_systems -v
```
Expected: FAIL — `ImportError: cannot import name 'load_oracle_system_bboxes' from 'src.data.yolo_aligned_systems'` (module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/yolo_aligned_systems.py
"""Multi-staff system-level helpers for Stage 3 data prep.

Mirrors `src/data/yolo_aligned_crops.py` but operates on system bboxes
(one label per system, multiple staves per system) rather than per-staff.
"""
from __future__ import annotations

from pathlib import Path


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_yolo_aligned_systems.py::test_load_oracle_system_bboxes_three_systems tests/data/test_yolo_aligned_systems.py::test_load_oracle_system_bboxes_empty_label -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/data/yolo_aligned_systems.py tests/data/test_yolo_aligned_systems.py && git commit -m "feat(data): add oracle system bbox loader for Stage 3"
```

---

## Task 3: System→staff mapping from staves.json + IoU matching

**Files:**
- Modify: `src/data/yolo_aligned_systems.py`
- Modify: `tests/data/test_yolo_aligned_systems.py`

Two pieces:
1. `load_staves_per_system(json_path)` — reads `<page>.staves.json`, returns the list.
2. `match_yolo_to_oracle_systems(yolo_boxes, oracle_systems, iou_threshold=0.5)` — same shape as the existing `match_yolo_to_oracle` in `yolo_aligned_crops.py:30-75` but indexes by `system_index` instead of `staff_index`.

The cumsum mapping (system_index → staff_index range) is its own concern — `staff_indices_for_system(system_index, staves_per_system)`.

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/data/test_yolo_aligned_systems.py

from src.data.yolo_aligned_systems import (
    load_oracle_system_bboxes,
    load_staves_per_system,
    match_yolo_to_oracle_systems,
    staff_indices_for_system,
)


def test_load_staves_per_system(tmp_path: Path):
    p = tmp_path / "page.staves.json"
    p.write_text("[3, 3, 3]")
    assert load_staves_per_system(p) == [3, 3, 3]


def test_load_staves_per_system_missing(tmp_path: Path):
    # Missing file → empty list (the system bbox file is the source of truth)
    p = tmp_path / "missing.staves.json"
    assert load_staves_per_system(p) == []


def test_staff_indices_for_system_uniform():
    # 3 systems with 3 staves each
    assert staff_indices_for_system(0, [3, 3, 3]) == [0, 1, 2]
    assert staff_indices_for_system(1, [3, 3, 3]) == [3, 4, 5]
    assert staff_indices_for_system(2, [3, 3, 3]) == [6, 7, 8]


def test_staff_indices_for_system_varied():
    # Systems with different staff counts: vocal (1) + piano (2) + piano (2)
    assert staff_indices_for_system(0, [1, 2, 2]) == [0]
    assert staff_indices_for_system(1, [1, 2, 2]) == [1, 2]
    assert staff_indices_for_system(2, [1, 2, 2]) == [3, 4]


def test_match_yolo_to_oracle_systems_basic():
    yolo_boxes = [
        {"yolo_idx": 0, "bbox": (10, 10, 100, 50), "conf": 0.99},
        {"yolo_idx": 1, "bbox": (10, 60, 100, 100), "conf": 0.95},
    ]
    oracle = [
        {"system_index": 0, "bbox": (10, 10, 100, 50)},  # exact match for yolo 0
        {"system_index": 1, "bbox": (10, 60, 100, 100)},  # exact match for yolo 1
    ]
    matches = match_yolo_to_oracle_systems(yolo_boxes, oracle, iou_threshold=0.5)
    assert len(matches) == 2
    assert {m["system_index"] for m in matches} == {0, 1}
    assert all(m["iou"] > 0.99 for m in matches)


def test_match_yolo_to_oracle_systems_drops_below_threshold():
    yolo_boxes = [{"yolo_idx": 0, "bbox": (0, 0, 10, 10), "conf": 0.99}]
    oracle = [{"system_index": 0, "bbox": (50, 50, 100, 100)}]  # disjoint
    assert match_yolo_to_oracle_systems(yolo_boxes, oracle, iou_threshold=0.5) == []


def test_match_yolo_to_oracle_systems_keeps_highest_conf_on_dup():
    yolo_boxes = [
        {"yolo_idx": 0, "bbox": (10, 10, 100, 50), "conf": 0.80},
        {"yolo_idx": 1, "bbox": (12, 12, 102, 52), "conf": 0.99},  # same oracle, higher conf
    ]
    oracle = [{"system_index": 0, "bbox": (10, 10, 100, 50)}]
    matches = match_yolo_to_oracle_systems(yolo_boxes, oracle, iou_threshold=0.5)
    assert len(matches) == 1
    assert matches[0]["yolo_idx"] == 1  # higher-conf YOLO box won
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_yolo_aligned_systems.py -v
```
Expected: FAIL with ImportError on the new symbols (the loader from Task 2 still passes; the four new ones fail).

- [ ] **Step 3: Write minimal implementations**

```python
# Add to src/data/yolo_aligned_systems.py

import json
from typing import Iterable

from src.data.yolo_aligned_crops import iou_xyxy  # reuse


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_yolo_aligned_systems.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/data/yolo_aligned_systems.py tests/data/test_yolo_aligned_systems.py && git commit -m "feat(data): system bbox matching + staves_per_system mapping"
```

---

## Task 4: Multi-staff token assembly with markers

**Files:**
- Modify: `src/data/yolo_aligned_systems.py`
- Modify: `tests/data/test_yolo_aligned_systems.py`

The function `assemble_multi_staff_tokens(per_staff_token_sequences: list[list[str]]) -> list[str]` takes ordered per-staff token lists and returns a multi-staff sequence with `<staff_idx_N>` markers. Per-staff inputs follow the existing per-staff manifest convention: `<bos> <staff_start> [content] <staff_end> <eos>`. Output: `<bos> <staff_start> <staff_idx_0> [content_0] <staff_end> <staff_start> <staff_idx_1> [content_1] <staff_end> ... <eos>`.

Marker tokens come from `src/tokenizer/vocab.py:STAFF_INDEX_MARKER_TOKENS` (already in repo per Subproject 2).

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/data/test_yolo_aligned_systems.py

from src.data.yolo_aligned_systems import assemble_multi_staff_tokens


def test_assemble_multi_staff_tokens_two_staves():
    staff_0 = ["<bos>", "<staff_start>", "clef-G2", "note-C4", "<staff_end>", "<eos>"]
    staff_1 = ["<bos>", "<staff_start>", "clef-F3", "note-C2", "<staff_end>", "<eos>"]
    out = assemble_multi_staff_tokens([staff_0, staff_1])
    assert out == [
        "<bos>",
        "<staff_start>", "<staff_idx_0>", "clef-G2", "note-C4", "<staff_end>",
        "<staff_start>", "<staff_idx_1>", "clef-F3", "note-C2", "<staff_end>",
        "<eos>",
    ]


def test_assemble_multi_staff_tokens_three_staves():
    s0 = ["<bos>", "<staff_start>", "a", "<staff_end>", "<eos>"]
    s1 = ["<bos>", "<staff_start>", "b", "<staff_end>", "<eos>"]
    s2 = ["<bos>", "<staff_start>", "c", "<staff_end>", "<eos>"]
    out = assemble_multi_staff_tokens([s0, s1, s2])
    assert out == [
        "<bos>",
        "<staff_start>", "<staff_idx_0>", "a", "<staff_end>",
        "<staff_start>", "<staff_idx_1>", "b", "<staff_end>",
        "<staff_start>", "<staff_idx_2>", "c", "<staff_end>",
        "<eos>",
    ]


def test_assemble_multi_staff_tokens_single_staff():
    s = ["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"]
    out = assemble_multi_staff_tokens([s])
    assert out == ["<bos>", "<staff_start>", "<staff_idx_0>", "x", "<staff_end>", "<eos>"]


def test_assemble_multi_staff_tokens_rejects_malformed():
    # Per-staff sequence missing wrapper tokens → assertion error (defensive)
    with pytest.raises(AssertionError):
        assemble_multi_staff_tokens([["clef-G2", "note-C4"]])  # no <bos>/<eos>


def test_assemble_multi_staff_tokens_rejects_too_many_staves():
    # vocab has 8 marker tokens (<staff_idx_0> through <staff_idx_7>); 9th raises
    staves = [["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"]] * 9
    with pytest.raises(ValueError):
        assemble_multi_staff_tokens(staves)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_yolo_aligned_systems.py -v
```
Expected: 5 new failures with `ImportError` on `assemble_multi_staff_tokens`.

- [ ] **Step 3: Write the implementation**

```python
# Add to src/data/yolo_aligned_systems.py

from src.tokenizer.vocab import STAFF_INDEX_MARKER_TOKENS


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_yolo_aligned_systems.py -v
```
Expected: 12 passed total (7 from previous + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/data/yolo_aligned_systems.py tests/data/test_yolo_aligned_systems.py && git commit -m "feat(data): multi-staff token assembly with <staff_idx_N> markers"
```

---

## Task 5: process_page_systems — end-to-end per-page driver

**Files:**
- Modify: `src/data/yolo_aligned_systems.py`
- Modify: `tests/data/test_yolo_aligned_systems.py`

`process_page_systems` is the multi-staff analog of `process_page` in `yolo_aligned_crops.py:129-223`. Inputs: page image, oracle system label paths (.txt + .staves.json), YOLO model, per-staff token lookup, output crop dir. Outputs: list of multi-staff manifest entries + per-page report.

- [ ] **Step 1: Write the failing test (with mocked YOLO)**

```python
# Add to tests/data/test_yolo_aligned_systems.py

from PIL import Image
from src.data.yolo_aligned_systems import process_page_systems


class _FakeYoloResult:
    """Minimal Ultralytics-result shape that yolo_aligned_crops._yolo_predict_to_boxes accepts."""
    def __init__(self, xyxy_list, conf_list):
        class _Boxes:
            pass
        self.boxes = _Boxes()
        self.boxes.xyxy = xyxy_list
        self.boxes.conf = conf_list


class _FakeYoloModel:
    def __init__(self, predictions: list[tuple[tuple[float, float, float, float], float]]):
        self._predictions = predictions

    def predict(self, image, imgsz=1920, conf=0.25, verbose=False):
        xyxy = [list(box) for box, _ in self._predictions]
        confs = [c for _, c in self._predictions]
        return [_FakeYoloResult(xyxy, confs)]


def test_process_page_systems_end_to_end(tmp_path: Path):
    page = Image.new("RGB", (1000, 1500), color=(255, 255, 255))
    page_path = tmp_path / "page.png"
    page.save(page_path)

    # Write oracle: 2 systems
    txt = tmp_path / "page.txt"
    txt.write_text(
        "0 0.5 0.2 0.9 0.3\n"   # system 0: y_center=0.2, height=0.3 → pixel y in [75, 525], x in [50, 950]
        "0 0.5 0.7 0.9 0.3\n"   # system 1: y_center=0.7
    )
    json_path = tmp_path / "page.staves.json"
    json_path.write_text("[2, 2]")

    # Token lookup: 4 staves total
    def _staff_seq(label):
        return ["<bos>", "<staff_start>", label, "<staff_end>", "<eos>"]
    token_lookup = {
        ("p001", 0): {"page_id": "p001", "staff_index": 0, "style_id": "x", "page_number": 1, "split": "train", "source_path": "src", "source_format": "musicxml", "score_type": "piano", "token_sequence": _staff_seq("a"), "token_count": 5, "dataset": "synthetic_fullpage"},
        ("p001", 1): {"page_id": "p001", "staff_index": 1, "style_id": "x", "page_number": 1, "split": "train", "source_path": "src", "source_format": "musicxml", "score_type": "piano", "token_sequence": _staff_seq("b"), "token_count": 5, "dataset": "synthetic_fullpage"},
        ("p001", 2): {"page_id": "p001", "staff_index": 2, "style_id": "x", "page_number": 1, "split": "train", "source_path": "src", "source_format": "musicxml", "score_type": "piano", "token_sequence": _staff_seq("c"), "token_count": 5, "dataset": "synthetic_fullpage"},
        ("p001", 3): {"page_id": "p001", "staff_index": 3, "style_id": "x", "page_number": 1, "split": "train", "source_path": "src", "source_format": "musicxml", "score_type": "piano", "token_sequence": _staff_seq("d"), "token_count": 5, "dataset": "synthetic_fullpage"},
    }

    # YOLO predictions matching both oracle systems (high IoU)
    yolo_model = _FakeYoloModel([
        ((50, 75, 950, 525), 0.99),
        ((50, 825, 950, 1275), 0.97),
    ])

    crops_dir = tmp_path / "crops"
    entries, report = process_page_systems(
        page_id="p001",
        page_image_path=page_path,
        oracle_label_path=txt,
        oracle_staves_json_path=json_path,
        yolo_model=yolo_model,
        token_lookup=token_lookup,
        out_crops_dir=crops_dir,
        crop_path_template="crops/{filename}",
        iou_threshold=0.5,
    )

    assert report["yolo_boxes"] == 2
    assert report["oracle_systems"] == 2
    assert report["matches"] == 2
    assert len(entries) == 2

    # First entry: system 0, staves 0+1
    e0 = entries[0]
    assert e0["staves_in_system"] == 2
    assert e0["dataset"] == "synthetic_systems"
    assert "<staff_idx_0>" in e0["token_sequence"]
    assert "<staff_idx_1>" in e0["token_sequence"]
    assert e0["token_sequence"][0] == "<bos>"
    assert e0["token_sequence"][-1] == "<eos>"
    # Crop file actually written
    assert (crops_dir / Path(e0["image_path"]).name).exists()


def test_process_page_systems_handles_recall_gap(tmp_path: Path):
    """If YOLO misses a system, that system is dropped from output."""
    page = Image.new("RGB", (1000, 1500))
    page_path = tmp_path / "page.png"
    page.save(page_path)

    txt = tmp_path / "page.txt"
    txt.write_text("0 0.5 0.2 0.9 0.3\n0 0.5 0.7 0.9 0.3\n")
    (tmp_path / "page.staves.json").write_text("[1, 1]")

    yolo_model = _FakeYoloModel([
        ((50, 75, 950, 525), 0.99),  # only system 0
    ])
    token_lookup = {
        ("p", i): {"page_id": "p", "staff_index": i, "style_id": "x", "page_number": 1, "split": "train", "source_path": "s", "source_format": "musicxml", "score_type": "piano", "token_sequence": ["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"], "token_count": 5, "dataset": "synthetic_fullpage"}
        for i in (0, 1)
    }
    entries, report = process_page_systems(
        page_id="p", page_image_path=page_path, oracle_label_path=txt,
        oracle_staves_json_path=tmp_path / "page.staves.json",
        yolo_model=yolo_model, token_lookup=token_lookup,
        out_crops_dir=tmp_path / "crops", crop_path_template="crops/{filename}",
    )
    assert len(entries) == 1
    assert report["dropped_oracle_recall_gap"] == 1


def test_process_page_systems_handles_token_miss(tmp_path: Path):
    """If a matched system has no per-staff tokens for one of its staves, drop the system."""
    page = Image.new("RGB", (1000, 1500))
    page_path = tmp_path / "page.png"
    page.save(page_path)

    (tmp_path / "page.txt").write_text("0 0.5 0.5 0.9 0.5\n")
    (tmp_path / "page.staves.json").write_text("[3]")

    yolo_model = _FakeYoloModel([((50, 375, 950, 1125), 0.99)])
    # Only 2 of 3 staves have token entries
    token_lookup = {
        ("p", i): {"page_id": "p", "staff_index": i, "style_id": "x", "page_number": 1, "split": "train", "source_path": "s", "source_format": "musicxml", "score_type": "piano", "token_sequence": ["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"], "token_count": 5, "dataset": "synthetic_fullpage"}
        for i in (0, 1)  # missing staff 2
    }
    entries, report = process_page_systems(
        page_id="p", page_image_path=page_path, oracle_label_path=tmp_path / "page.txt",
        oracle_staves_json_path=tmp_path / "page.staves.json",
        yolo_model=yolo_model, token_lookup=token_lookup,
        out_crops_dir=tmp_path / "crops", crop_path_template="crops/{filename}",
    )
    assert len(entries) == 0
    assert report["dropped_token_miss"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_yolo_aligned_systems.py -v
```
Expected: 3 new failures.

- [ ] **Step 3: Write the implementation**

```python
# Add to src/data/yolo_aligned_systems.py

from PIL import Image

from src.data.yolo_aligned_crops import _yolo_predict_to_boxes


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
    """Multi-staff variant of `yolo_aligned_crops.process_page`.

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

        # Assemble multi-staff sequence; trap marker overflow as a drop, not a crash
        try:
            multi_seq = assemble_multi_staff_tokens(per_staff_seqs)
        except ValueError:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_yolo_aligned_systems.py -v
```
Expected: 15 passed total.

- [ ] **Step 5: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/data/yolo_aligned_systems.py tests/data/test_yolo_aligned_systems.py && git commit -m "feat(data): process_page_systems for multi-staff token assembly"
```

---

## Task 6: build_synthetic_systems_v1.py — driver script

**Files:**
- Create: `scripts/build_synthetic_systems_v1.py`

This iterates the synthetic_v2 per-staff manifest grouped by `(page_id, style_id)`, builds a token_lookup dict, runs `process_page_systems` per page, aggregates results into the new `synthetic_systems_v1` manifest + audit report. Modeled on `scripts/build_grandstaff_systems.py` and `scripts/build_mixed_v2_systems.py`.

- [ ] **Step 1: Write the script**

```python
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
                crop_path_template=f"{args.output_crops_root}/{style_id}/{{filename}}",
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
```

- [ ] **Step 2: Smoke-test the script importable (no YOLO needed)**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -c "import scripts.build_synthetic_systems_v1; print('ok')"
```
Expected: `ok`. If the import errors before main() runs, ultralytics import was eagerly imported — fix.

- [ ] **Step 3: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add scripts/build_synthetic_systems_v1.py && git commit -m "feat(data): build_synthetic_systems_v1.py driver script"
```

---

## Task 7: Run smoke test on GPU box (10 pages)

**Operational task on GPU box. No code changes.**

- [ ] **Step 1: Push the branch state to GPU box**

User runs out-of-band: pull or rsync the latest local changes to the GPU box clone of the repo. (No specific command — depends on user's normal sync workflow.)

- [ ] **Step 2: Run smoke test**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && venv-cu132\Scripts\python -u scripts\build_synthetic_systems_v1.py --per-staff-manifest data\processed\synthetic_v2\manifests\synthetic_token_manifest.jsonl --labels-systems-root data\processed\synthetic_v2\labels_systems --page-images-root data\processed\synthetic_v2\images --dpi dpi300 --yolo-weights runs\detect\runs\yolo26m_systems\weights\best.pt --output-manifest data\processed\synthetic_systems_v1\smoketest_manifest.jsonl --output-crops-root data\processed\synthetic_systems_v1\smoketest_crops --audit-output data\processed\synthetic_systems_v1\smoketest_audit.json --max-pages 10"
```

Expected:
- Script completes in < 60 seconds
- `smoketest_audit.json` shows `entries_written > 0`
- `match` rate (matches / oracle_systems) is high (≥ 0.95 expected given YOLO mAP50 0.995)
- `dropped_token_miss`, `dropped_degenerate`, `dropped_marker_overflow` all 0 (or very low) on a 10-page sample
- Crops directory has files

- [ ] **Step 3: Manual review**

Inspect 2–3 crop images visually (open in viewer or scp back to local). Crops should be tightly bounded around the system and contain readable music. If they're badly clipped or off, halt — the YOLO bbox interpretation is wrong.

If the smoke test reveals issues (mismatched style_id paths, wrong image_path resolution, marker overflow on real pages, etc.) — STOP, diagnose, fix in a separate task before proceeding.

- [ ] **Step 4: Document the smoke result**

No commit. Record the smoketest_audit.json's full output in the next task's commit message for traceability.

---

## Task 8: Run full build on GPU box (~6,730 pages × 3 styles)

**Operational task on GPU box. No code changes.**

This is the heavy run. Expected wall time: 1–3 hours depending on YOLO inference speed and disk I/O. Outputs:
- `data\processed\synthetic_systems_v1\manifests\synthetic_token_manifest.jsonl`
- `data\processed\synthetic_systems_v1\system_crops\<style>\` (~20–25k images)
- `data\processed\synthetic_systems_v1\audit.json`

- [ ] **Step 1: Verify smoke artifacts cleaned up**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && (if exist data\processed\synthetic_systems_v1\smoketest_manifest.jsonl del data\processed\synthetic_systems_v1\smoketest_manifest.jsonl) && (if exist data\processed\synthetic_systems_v1\smoketest_crops rmdir /s /q data\processed\synthetic_systems_v1\smoketest_crops) && (if exist data\processed\synthetic_systems_v1\smoketest_audit.json del data\processed\synthetic_systems_v1\smoketest_audit.json)"
```
Expected: clean exit (no error if files don't exist).

- [ ] **Step 2: Run full build**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && venv-cu132\Scripts\python -u scripts\build_synthetic_systems_v1.py --per-staff-manifest data\processed\synthetic_v2\manifests\synthetic_token_manifest.jsonl --labels-systems-root data\processed\synthetic_v2\labels_systems --page-images-root data\processed\synthetic_v2\images --dpi dpi300 --yolo-weights runs\detect\runs\yolo26m_systems\weights\best.pt 2>&1 | tee logs\build_synthetic_systems_v1.log"
```

Run with the default output paths (no `--output-*` overrides). Expected duration 1–3h. Use `tee` so the log persists.

- [ ] **Step 3: Verify the audit**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && type data\processed\synthetic_systems_v1\audit.json"
```

Expected:
- `pages_processed` ~ 7500 (6,730 unique pages × 3 styles, with attrition for missing labels)
- `entries_written` somewhere in the 20-30k range (depends on systems-per-page distribution)
- `match rate = matches / oracle_systems` ≥ 0.95
- `dropped_marker_overflow` should be 0 — if non-zero, some pages have systems with > 8 staves (vocab limit). Document and decide whether to extend vocab.
- `staves_histogram` shows the distribution of `staves_in_system`. Spec §3.1 expects ~10% 1-staff, ~5% 2-staff, ~70% 3-staff, ~15% 4–5-staff. Verify the empirical mass matches roughly.

- [ ] **Step 4: Sanity-check the output manifest**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && powershell -c \"$lines = Get-Content data\processed\synthetic_systems_v1\manifests\synthetic_token_manifest.jsonl -ReadCount 1; $first = $lines[0] | ConvertFrom-Json; $first | ConvertTo-Json -Depth 4\""
```
Expected: a single multi-staff entry, with `<staff_idx_0>`, `<staff_idx_1>`, ... in `token_sequence`, `staves_in_system` consistent with the marker count, `staff_indices` matching cumsum.

- [ ] **Step 5: Commit the audit artifact**

The manifest itself is ~30 GB and shouldn't be committed (data, not code). The audit.json is small and useful — copy back to local clone and commit:

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && scp 10.10.1.29:'"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\data\processed\synthetic_systems_v1\audit.json"' data/processed/synthetic_systems_v1/audit.json && git add data/processed/synthetic_systems_v1/audit.json && git commit -m "data: synthetic_systems_v1 build audit ($(jq -r '.entries_written' data/processed/synthetic_systems_v1/audit.json) entries)"
```

If the audit's match rate is below 0.95 OR `dropped_marker_overflow > 0`, halt and decide before proceeding.

---

## Task 9: Generic single-staff retokenizer (primus + cameraprimus)

**Files:**
- Create: `scripts/retokenize_with_staff_markers.py`
- Create: `tests/data/test_retokenize_with_staff_markers.py`

The transformation is one-line per entry:
- Input token_sequence: `<bos> <staff_start> [content] <staff_end> <eos>`
- Output token_sequence: `<bos> <staff_start> <staff_idx_0> [content] <staff_end> <eos>`

The script reads a source manifest (or filters a combined manifest by `dataset` field), applies the transform, writes a new manifest. `dataset` field gets renamed from `primus` → `primus_systems` and `cameraprimus` → `cameraprimus_systems`. `staves_in_system` set to 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_retokenize_with_staff_markers.py
import json
from pathlib import Path
import subprocess
import sys


def _make_entry(dataset: str, sample_id: str, content_tokens: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "split": "train",
        "image_path": f"data/{dataset}/x.png",
        "source_path": f"data/{dataset}/x.semantic",
        "source_format": "semantic",
        "token_sequence": ["<bos>", "<staff_start>"] + content_tokens + ["<staff_end>", "<eos>"],
        "token_count": 4 + len(content_tokens),
    }


def test_retokenize_adds_staff_idx_0_marker(tmp_path: Path):
    src = tmp_path / "input.jsonl"
    src.write_text(
        json.dumps(_make_entry("primus", "primus:001", ["clef-G2", "note-C4"])) + "\n"
        + json.dumps(_make_entry("primus", "primus:002", ["clef-G2", "rest"])) + "\n"
        + json.dumps(_make_entry("cameraprimus", "cameraprimus:001", ["clef-G2", "note-D4"])) + "\n"
    )

    out = tmp_path / "output.jsonl"
    cmd = [
        sys.executable,
        "scripts/retokenize_with_staff_markers.py",
        "--input-manifest", str(src),
        "--source-dataset", "primus",
        "--target-dataset", "primus_systems",
        "--output-manifest", str(out),
    ]
    result = subprocess.run(cmd, cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = [json.loads(L) for L in out.read_text().splitlines()]
    # Only primus entries selected (not cameraprimus)
    assert len(lines) == 2
    for e in lines:
        assert e["dataset"] == "primus_systems"
        assert e["staves_in_system"] == 1
        assert e["token_sequence"][0] == "<bos>"
        assert e["token_sequence"][1] == "<staff_start>"
        assert e["token_sequence"][2] == "<staff_idx_0>"
        assert e["token_sequence"][-2] == "<staff_end>"
        assert e["token_sequence"][-1] == "<eos>"
        # token_count updated to reflect inserted marker
        assert e["token_count"] == len(e["token_sequence"])


def test_retokenize_rejects_malformed_input(tmp_path: Path):
    src = tmp_path / "bad.jsonl"
    src.write_text(json.dumps({
        "sample_id": "x",
        "dataset": "primus",
        "split": "train",
        "image_path": "x.png",
        "source_path": "x",
        "source_format": "semantic",
        "token_sequence": ["clef-G2"],  # missing wrappers entirely
        "token_count": 1,
    }) + "\n")

    out = tmp_path / "out.jsonl"
    cmd = [
        sys.executable,
        "scripts/retokenize_with_staff_markers.py",
        "--input-manifest", str(src),
        "--source-dataset", "primus",
        "--target-dataset", "primus_systems",
        "--output-manifest", str(out),
    ]
    result = subprocess.run(cmd, cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
                            capture_output=True, text=True)
    # Should exit non-zero (malformed input is a builder bug, not silent skip)
    assert result.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_retokenize_with_staff_markers.py -v
```
Expected: FAIL — script doesn't exist yet.

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python3
"""Generic single-staff retokenizer.

Filters a manifest to a single source dataset, prepends `<staff_idx_0>` to the
content of each entry, rewrites `dataset` and `staves_in_system` fields. Used
to produce primus_systems and cameraprimus_systems from the existing combined
or per-dataset manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _retokenize_entry(entry: dict, target_dataset: str) -> dict:
    seq = entry["token_sequence"]
    if not (len(seq) >= 4 and seq[0] == "<bos>" and seq[1] == "<staff_start>"
            and seq[-2] == "<staff_end>" and seq[-1] == "<eos>"):
        raise ValueError(
            f"malformed token_sequence for entry {entry.get('sample_id', '?')}: "
            f"first 2 = {seq[:2]}, last 2 = {seq[-2:]}"
        )
    new_seq = ["<bos>", "<staff_start>", "<staff_idx_0>"] + seq[2:]  # insert marker after staff_start
    new_entry = dict(entry)
    new_entry["dataset"] = target_dataset
    new_entry["staves_in_system"] = 1
    new_entry["token_sequence"] = new_seq
    new_entry["token_count"] = len(new_seq)
    return new_entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True,
                        help="Filter input to entries with this `dataset` field (e.g., 'primus').")
    parser.add_argument("--target-dataset", required=True,
                        help="Set the output `dataset` field to this (e.g., 'primus_systems').")
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    with args.input_manifest.open("r", encoding="utf-8") as fin, \
         args.output_manifest.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            n_in += 1
            if entry.get("dataset") != args.source_dataset:
                continue
            new_entry = _retokenize_entry(entry, args.target_dataset)
            fout.write(json.dumps(new_entry) + "\n")
            n_out += 1

    print(f"[retokenize] read {n_in} entries; wrote {n_out} for source={args.source_dataset} → target={args.target_dataset}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_retokenize_with_staff_markers.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add scripts/retokenize_with_staff_markers.py tests/data/test_retokenize_with_staff_markers.py && git commit -m "feat(data): retokenize_with_staff_markers.py for primus/cameraprimus"
```

---

## Task 10: Run retokenizer on primus + cameraprimus

**Operational task on GPU box. No code changes.**

The source manifest is the existing combined `src\data\manifests\token_manifest_full_systems.jsonl` (845 MB, contains primus + cameraprimus + grandstaff/_systems).

- [ ] **Step 1: Run for primus**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && venv-cu132\Scripts\python -u scripts\retokenize_with_staff_markers.py --input-manifest src\data\manifests\token_manifest_full_systems.jsonl --source-dataset primus --target-dataset primus_systems --output-manifest data\processed\primus_systems\manifests\synthetic_token_manifest.jsonl"
```
Expected output: `[retokenize] read ~390000 entries; wrote ~87678 for source=primus → target=primus_systems`.

- [ ] **Step 2: Run for cameraprimus**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && venv-cu132\Scripts\python -u scripts\retokenize_with_staff_markers.py --input-manifest src\data\manifests\token_manifest_full_systems.jsonl --source-dataset cameraprimus --target-dataset cameraprimus_systems --output-manifest data\processed\cameraprimus_systems\manifests\synthetic_token_manifest.jsonl"
```
Expected output: `wrote ~87678 for source=cameraprimus → target=cameraprimus_systems`.

- [ ] **Step 3: Sanity-check output schema**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && powershell -c \"Get-Content data\processed\primus_systems\manifests\synthetic_token_manifest.jsonl -TotalCount 1 | ConvertFrom-Json | ConvertTo-Json -Depth 4\""
```
Expected: token_sequence has `<staff_idx_0>` as the third element; `dataset = primus_systems`; `staves_in_system = 1`.

Repeat for cameraprimus_systems.

- [ ] **Step 4: Record entry counts**

No commit needed. Note the entry counts for the next task's combined-manifest builder.

---

## Task 11: Combined Stage 3 manifest builder

**Files:**
- Create: `scripts/build_stage3_combined_manifest.py`
- Create: `tests/data/test_build_stage3_combined_manifest.py`

This script reads the four source manifests and concatenates them into a single `token_manifest_stage3.jsonl`. The trainer's existing `WeightedRandomSampler` (`_build_weighted_sampler` in `src/train/train.py:702-785`) handles the 70/10/10/10 ratio at runtime; the combined manifest just needs to provide entries from all four datasets in one file.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_build_stage3_combined_manifest.py
import json
import subprocess
import sys
from pathlib import Path


def _make_entries(dataset: str, n: int) -> list[dict]:
    return [
        {
            "sample_id": f"{dataset}:{i}",
            "dataset": dataset,
            "split": "train",
            "image_path": f"data/{dataset}/{i}.png",
            "token_sequence": ["<bos>", "<staff_start>", "<staff_idx_0>", "x", "<staff_end>", "<eos>"],
            "token_count": 6,
            "staves_in_system": 1,
        }
        for i in range(n)
    ]


def _write_manifest(path: Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_build_combined_manifest_concatenates_four_sources(tmp_path: Path):
    syn = tmp_path / "synth.jsonl"
    gs = tmp_path / "gs.jsonl"
    pr = tmp_path / "primus.jsonl"
    cp = tmp_path / "cp.jsonl"
    _write_manifest(syn, _make_entries("synthetic_systems", 5))
    _write_manifest(gs, _make_entries("grandstaff_systems", 3))
    _write_manifest(pr, _make_entries("primus_systems", 2))
    _write_manifest(cp, _make_entries("cameraprimus_systems", 2))

    out = tmp_path / "combined.jsonl"
    audit = tmp_path / "combined_audit.json"
    cmd = [
        sys.executable, "scripts/build_stage3_combined_manifest.py",
        "--synthetic-systems-manifest", str(syn),
        "--grandstaff-systems-manifest", str(gs),
        "--primus-systems-manifest", str(pr),
        "--cameraprimus-systems-manifest", str(cp),
        "--output-manifest", str(out),
        "--audit-output", str(audit),
    ]
    result = subprocess.run(cmd, cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    lines = [json.loads(L) for L in out.read_text().splitlines()]
    assert len(lines) == 12
    # All four datasets represented
    datasets = {e["dataset"] for e in lines}
    assert datasets == {"synthetic_systems", "grandstaff_systems",
                        "primus_systems", "cameraprimus_systems"}

    audit_data = json.loads(audit.read_text())
    assert audit_data["total_entries"] == 12
    assert audit_data["per_dataset"]["synthetic_systems"] == 5
    assert audit_data["per_dataset"]["grandstaff_systems"] == 3
    assert audit_data["per_dataset"]["primus_systems"] == 2
    assert audit_data["per_dataset"]["cameraprimus_systems"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_build_stage3_combined_manifest.py -v
```
Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python3
"""Build the combined Stage 3 token manifest by concatenating four source manifests."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _stream_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-systems-manifest", type=Path, required=True)
    parser.add_argument("--grandstaff-systems-manifest", type=Path, required=True)
    parser.add_argument("--primus-systems-manifest", type=Path, required=True)
    parser.add_argument("--cameraprimus-systems-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    sources = [
        ("synthetic_systems", args.synthetic_systems_manifest),
        ("grandstaff_systems", args.grandstaff_systems_manifest),
        ("primus_systems", args.primus_systems_manifest),
        ("cameraprimus_systems", args.cameraprimus_systems_manifest),
    ]

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    per_dataset = Counter()
    per_split = Counter()
    total = 0
    with args.output_manifest.open("w", encoding="utf-8") as fout:
        for expected_dataset, src_path in sources:
            for entry in _stream_manifest(src_path):
                ds = entry.get("dataset")
                if ds != expected_dataset:
                    raise ValueError(
                        f"manifest {src_path} has entry with dataset={ds!r}, "
                        f"expected {expected_dataset!r}"
                    )
                fout.write(json.dumps(entry) + "\n")
                per_dataset[ds] += 1
                per_split[entry.get("split", "train")] += 1
                total += 1

    audit = {
        "total_entries": total,
        "per_dataset": dict(per_dataset),
        "per_split": dict(per_split),
        "sources": [str(p) for _, p in sources],
        "output": str(args.output_manifest),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2))
    print(f"[combined] wrote {total} entries; per_dataset={dict(per_dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_build_stage3_combined_manifest.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add scripts/build_stage3_combined_manifest.py tests/data/test_build_stage3_combined_manifest.py && git commit -m "feat(data): build_stage3_combined_manifest.py"
```

---

## Task 12: Run combined-manifest builder on GPU box

**Operational task on GPU box. No code changes.**

- [ ] **Step 1: Run with all four sources**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && venv-cu132\Scripts\python -u scripts\build_stage3_combined_manifest.py --synthetic-systems-manifest data\processed\synthetic_systems_v1\manifests\synthetic_token_manifest.jsonl --grandstaff-systems-manifest data\processed\grandstaff_systems\manifests\synthetic_token_manifest.jsonl --primus-systems-manifest data\processed\primus_systems\manifests\synthetic_token_manifest.jsonl --cameraprimus-systems-manifest data\processed\cameraprimus_systems\manifests\synthetic_token_manifest.jsonl --output-manifest src\data\manifests\token_manifest_stage3.jsonl --audit-output src\data\manifests\token_manifest_stage3_audit.json"
```

Expected entry counts:
- synthetic_systems: 20–30k (from Task 8)
- grandstaff_systems: ~107,724 (from existing manifest — verified earlier)
- primus_systems: ~87,678
- cameraprimus_systems: ~87,678
- Total: ~300–320k

- [ ] **Step 2: Verify audit**

```cmd
ssh 10.10.1.29 "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\" && type src\data\manifests\token_manifest_stage3_audit.json"
```
Confirm `per_dataset` counts and total. If any source's count is unexpectedly low, halt.

- [ ] **Step 3: Copy audit back to local clone and commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && scp 10.10.1.29:'"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\src\data\manifests\token_manifest_stage3_audit.json"' src/data/manifests/token_manifest_stage3_audit.json && git add src/data/manifests/token_manifest_stage3_audit.json && git commit -m "data: stage3 combined manifest audit"
```

---

## Self-Review

Spec coverage:
- §3.3.1 (system crop generation, IoU match, per-staff token lookup) — covered by Tasks 2–8.
- §3.3.2 (multi-staff token assembly with markers) — covered by Task 4.
- §3.3.3 (primus retokenization with `<staff_idx_0>` markers) — covered by Tasks 9–10.
- Stage 3 redesign extension (cameraprimus retokenization) — covered by Task 10 Step 2.
- Combined Stage 3 training manifest — covered by Tasks 11–12.

Placeholder scan: no "TBD" or "TODO" present. The `--max-pages` arg in Task 6 is intentional infrastructure for the smoke-test step in Task 7 (concrete, not placeholder).

Type consistency: `system_index` and `staves_in_system` used consistently across Tasks 2–6. `staff_indices` (list[int] of page-global staff positions in a matched system) used in Task 5 manifest entries. `dataset` field uses `_systems` suffix consistently after retokenization.

Scope check: this plan produces working data prep. The next plan (Phase 0 cache infra, "Plan B" per the brainstorm) consumes the manifest produced by Task 12. Cleanly separable.

Token-assembly invariants: the per-staff manifest's wrapper convention (`<bos> <staff_start> [content] <staff_end> <eos>`) is verified by both `assemble_multi_staff_tokens` (multi-staff path) and `_retokenize_entry` (single-staff path). Both raise on malformed input.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-radio-stage3-data-prep.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review between tasks. Tasks 1, 7, 8, 10, 12 are operational tasks that need user-driven SSH; subagent surfaces those for execution and verifies outputs.

2. **Inline Execution** — execute tasks in this session using executing-plans, with checkpoints for review.

Which approach?
