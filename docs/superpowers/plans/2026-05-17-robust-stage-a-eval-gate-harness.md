# Robust Stage-A — Sub-plan A: Strict Eval & Gate Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the measuring stick — a CPU-unit-tested harness that scores any Stage-A checkpoint against the strict ship gate: per-scenario zero-bbox-error on a held-out real archetype set, lieder no-regression vs the committed 0.930 baseline, and a lyric-system recall sub-metric.

**Architecture:** Pure, side-effect-free scoring/gate logic (IoU matching, per-scenario binary verdict, geometry/lyric-clip checks, lyric-system recall, combined gate) lives in `eval/robust_stage_a/` and is fully CPU-unit-tested with synthetic fixtures. A thin CLI orchestrator runs YOLO inference (GPU/seder, reusing the existing inference path) and the existing lieder scorer, then feeds plain numbers/boxes into the pure gate. This separation is deliberate: the gate is verifiable without a GPU or the real held-out set.

**Tech Stack:** Python 3.14 (CPU for the gate/tests), pytest, the existing `eval/score_stage_a_only.py` lieder scorer, Ultralytics YOLO (inference only, on seder, via the existing Stage-A inference path). No torch import in the pure gate module.

**Spec:** [docs/superpowers/specs/2026-05-17-robust-stage-a-clutter-detection-design.md](../specs/2026-05-17-robust-stage-a-clutter-detection-design.md)

---

## Pinned decisions (resolving spec "deferred decisions" for this sub-plan)

- **Match IoU threshold** `MATCH_IOU = 0.5` — a predicted box is the same object as a GT system iff IoU ≥ 0.5 (standard detection matching).
- **Geometry / "Stage-B-usable"**: a matched prediction is geometry-good iff it **contains the GT system box** AND **contains every lyric-band** of that system, within `CONTAIN_TOL = 2.0` px on each side. A GT lyric-band not contained ⇒ counts as both a geometry failure and a `lyric_clip`.
- **Per-scenario verdict (binary)**: pass iff `false == 0 AND missed == 0 AND geometry_fail == 0`. For an `is_non_music` scenario (GT has zero systems): pass iff there are **zero predictions** (any prediction is a false system).
- **No-regression epsilon** `NO_REGRESSION_EPS = 0.0` — strict: new lieder recall must be `>=` the committed baseline recall; new lyric-system recall must be `>=` its snapshot baseline. (Exposed as a parameter; default strict per the spec.)
- **Lieder baseline source**: recompute from the committed `eval/results/stagea_baseline_pre_faintink.csv` as `sum(detected)/sum(expected)`, not a hardcoded 0.930.
- **Lyric-system-recall baseline**: there is no prior value; the harness snapshots it on first run against the current model (`--lyric-baseline-out`) and compares on subsequent runs (`--lyric-baseline`), mirroring how the 0.930 lieder baseline was snapshotted.

## File structure

**New files:**
- `eval/robust_stage_a/__init__.py` — package marker (empty).
- `eval/robust_stage_a/manifest.py` — held-out set manifest schema + loader/validator. Pure.
- `eval/robust_stage_a/gate.py` — pure scoring/gate logic (IoU, matching, geometry/lyric-clip, per-scenario verdict, lyric-system recall, combined gate). No torch/YOLO import.
- `eval/robust_stage_a/run_gate.py` — thin CLI orchestrator: load manifest, run Stage-A inference (reuse existing path), run the existing lieder scorer, feed numbers/boxes into `gate.py`, print + write a combined report.
- `tests/robust_stage_a/__init__.py` — package marker (empty).
- `tests/robust_stage_a/test_manifest.py` — CPU unit tests for the manifest loader/validator.
- `tests/robust_stage_a/test_gate.py` — CPU unit tests for all pure gate logic.

**Why `tests/robust_stage_a/`:** `tests/conftest.py` CUDA-gates only paths containing `{"inference","pipeline","cli","models","train"}`. `robust_stage_a` contains none of those tokens, so these tests run on CPU/locally (verified prior in this project's history that the gate keys on path components). The gate module must never import torch so the tests stay CPU-only.

---

## Task 1: Manifest schema + loader/validator

**Files:**
- Create: `eval/robust_stage_a/__init__.py`
- Create: `eval/robust_stage_a/manifest.py`
- Create: `tests/robust_stage_a/__init__.py`
- Test: `tests/robust_stage_a/test_manifest.py`

- [ ] **Step 1: Create the empty package markers**

```bash
mkdir -p eval/robust_stage_a tests/robust_stage_a
: > eval/robust_stage_a/__init__.py
: > tests/robust_stage_a/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/robust_stage_a/test_manifest.py`:

```python
"""CPU unit tests for the held-out archetype manifest loader/validator."""
import json
from pathlib import Path

import pytest

from eval.robust_stage_a.manifest import (
    GtSystem,
    Scenario,
    load_manifest,
)


def _write(tmp_path: Path, obj) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_loads_music_scenario(tmp_path):
    p = _write(tmp_path, [
        {
            "scenario_id": "title_over_system_01",
            "archetype": "title_over_system",
            "image": "scans/a.png",
            "is_non_music": False,
            "gt_systems": [
                {"box": [10, 20, 300, 80], "has_lyrics": True,
                 "lyric_bands": [[12, 70, 298, 80]]},
                {"box": [10, 100, 300, 160], "has_lyrics": False,
                 "lyric_bands": []},
            ],
        }
    ])
    scenarios = load_manifest(p)
    assert len(scenarios) == 1
    s = scenarios[0]
    assert isinstance(s, Scenario)
    assert s.scenario_id == "title_over_system_01"
    assert s.archetype == "title_over_system"
    assert s.is_non_music is False
    assert len(s.gt_systems) == 2
    assert isinstance(s.gt_systems[0], GtSystem)
    assert s.gt_systems[0].box == (10.0, 20.0, 300.0, 80.0)
    assert s.gt_systems[0].has_lyrics is True
    assert s.gt_systems[0].lyric_bands == [(12.0, 70.0, 298.0, 80.0)]
    assert s.gt_systems[1].has_lyrics is False
    assert s.gt_systems[1].lyric_bands == []


def test_loads_non_music_scenario_with_no_systems(tmp_path):
    p = _write(tmp_path, [
        {
            "scenario_id": "warranty_deed_01",
            "archetype": "pure_non_music",
            "image": "scans/deed.png",
            "is_non_music": True,
            "gt_systems": [],
        }
    ])
    scenarios = load_manifest(p)
    assert scenarios[0].is_non_music is True
    assert scenarios[0].gt_systems == []


def test_rejects_missing_required_field(tmp_path):
    p = _write(tmp_path, [{"scenario_id": "x", "archetype": "a",
                           "image": "i.png", "gt_systems": []}])  # no is_non_music
    with pytest.raises(ValueError, match="is_non_music"):
        load_manifest(p)


def test_rejects_non_music_with_systems(tmp_path):
    p = _write(tmp_path, [
        {"scenario_id": "x", "archetype": "a", "image": "i.png",
         "is_non_music": True,
         "gt_systems": [{"box": [0, 0, 1, 1], "has_lyrics": False,
                         "lyric_bands": []}]}
    ])
    with pytest.raises(ValueError, match="is_non_music.*gt_systems"):
        load_manifest(p)


def test_rejects_bad_box_arity(tmp_path):
    p = _write(tmp_path, [
        {"scenario_id": "x", "archetype": "a", "image": "i.png",
         "is_non_music": False,
         "gt_systems": [{"box": [0, 0, 1], "has_lyrics": False,
                         "lyric_bands": []}]}
    ])
    with pytest.raises(ValueError, match="box"):
        load_manifest(p)


def test_rejects_duplicate_scenario_ids(tmp_path):
    p = _write(tmp_path, [
        {"scenario_id": "dup", "archetype": "a", "image": "i.png",
         "is_non_music": True, "gt_systems": []},
        {"scenario_id": "dup", "archetype": "a", "image": "j.png",
         "is_non_music": True, "gt_systems": []},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/robust_stage_a/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.robust_stage_a.manifest'`

- [ ] **Step 4: Implement the manifest module**

Create `eval/robust_stage_a/manifest.py`:

```python
"""Held-out real archetype-set manifest: schema, loader, validator.

Pure (no torch / no YOLO). One JSON file = a list of scenario objects.

Scenario JSON schema (all fields required):
  scenario_id  : str, unique across the manifest
  archetype    : str, the failure-mode label (free-form, e.g. "title_over_system")
  image        : str, path to the scan image (relative to the manifest or absolute)
  is_non_music : bool, True => a pure non-music page; gt_systems MUST be []
  gt_systems   : list of {box:[x1,y1,x2,y2], has_lyrics:bool,
                          lyric_bands:[[x1,y1,x2,y2], ...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]


def _as_box(value, where: str) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{where}: box must be 4 numbers [x1,y1,x2,y2], got {value!r}")
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: box values must be numeric, got {value!r}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{where}: box must have x2>x1 and y2>y1, got {value!r}")
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class GtSystem:
    box: Box
    has_lyrics: bool
    lyric_bands: list[Box]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    archetype: str
    image: str
    is_non_music: bool
    gt_systems: list[GtSystem]


_REQUIRED = ("scenario_id", "archetype", "image", "is_non_music", "gt_systems")


def load_manifest(path: str | Path) -> list[Scenario]:
    """Parse + validate the manifest. Raises ValueError on any malformation."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("manifest root must be a JSON list of scenarios")

    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        where = f"scenario[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: must be an object")
        for field in _REQUIRED:
            if field not in entry:
                raise ValueError(f"{where}: missing required field '{field}'")

        sid = str(entry["scenario_id"])
        if sid in seen:
            raise ValueError(f"{where}: duplicate scenario_id {sid!r}")
        seen.add(sid)

        is_non_music = entry["is_non_music"]
        if not isinstance(is_non_music, bool):
            raise ValueError(f"{where}: is_non_music must be a bool")

        gt_raw = entry["gt_systems"]
        if not isinstance(gt_raw, list):
            raise ValueError(f"{where}: gt_systems must be a list")
        if is_non_music and gt_raw:
            raise ValueError(
                f"{where}: is_non_music=True requires empty gt_systems")

        gt_systems: list[GtSystem] = []
        for j, g in enumerate(gt_raw):
            gwhere = f"{where}.gt_systems[{j}]"
            if not isinstance(g, dict) or "box" not in g:
                raise ValueError(f"{gwhere}: must be an object with a 'box'")
            box = _as_box(g["box"], gwhere)
            has_lyrics = bool(g.get("has_lyrics", False))
            bands_raw = g.get("lyric_bands", []) or []
            if not isinstance(bands_raw, list):
                raise ValueError(f"{gwhere}: lyric_bands must be a list")
            bands = [_as_box(b, f"{gwhere}.lyric_bands[{k}]")
                     for k, b in enumerate(bands_raw)]
            gt_systems.append(GtSystem(box=box, has_lyrics=has_lyrics,
                                       lyric_bands=bands))

        scenarios.append(Scenario(
            scenario_id=sid,
            archetype=str(entry["archetype"]),
            image=str(entry["image"]),
            is_non_music=is_non_music,
            gt_systems=gt_systems,
        ))
    return scenarios
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/robust_stage_a/test_manifest.py -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add eval/robust_stage_a/__init__.py eval/robust_stage_a/manifest.py \
        tests/robust_stage_a/__init__.py tests/robust_stage_a/test_manifest.py
git commit -m "feat(eval): robust-stage-a held-out manifest schema + validator"
git push origin main
```

---

## Task 2: IoU + greedy prediction↔GT matching

**Files:**
- Create: `eval/robust_stage_a/gate.py`
- Test: `tests/robust_stage_a/test_gate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/robust_stage_a/test_gate.py`:

```python
"""CPU unit tests for the pure Stage-A strict-gate scoring logic."""
import pytest

from eval.robust_stage_a.gate import Pred, iou, contains, match_predictions


def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # a=100 area, b=100 area, intersection 50 -> 50/150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_contains_true_within_tol():
    assert contains((0, 0, 100, 100), (1, 1, 99, 99), tol=2.0) is True
    # inner pokes 1px outside but within 2px tol
    assert contains((0, 0, 100, 100), (-1, 0, 100, 100), tol=2.0) is True


def test_contains_false_outside_tol():
    assert contains((0, 0, 100, 100), (-5, 0, 100, 100), tol=2.0) is False


def test_match_greedy_by_confidence():
    gt = [(0, 0, 10, 10), (100, 100, 110, 110)]
    preds = [
        Pred(box=(0, 0, 10, 10), conf=0.9),       # matches gt0
        Pred(box=(100, 100, 110, 110), conf=0.8),  # matches gt1
        Pred(box=(200, 200, 210, 210), conf=0.7),  # false
    ]
    m = match_predictions(gt, preds, match_iou=0.5)
    assert m.matched == [(0, 0), (1, 1)]  # (gt_idx, pred_idx)
    assert m.missed_gt == []
    assert m.false_pred == [2]


def test_match_missed_when_below_iou():
    gt = [(0, 0, 10, 10)]
    preds = [Pred(box=(7, 0, 17, 10), conf=0.9)]  # iou 30/170 < 0.5
    m = match_predictions(gt, preds, match_iou=0.5)
    assert m.matched == []
    assert m.missed_gt == [0]
    assert m.false_pred == [0]


def test_match_one_pred_cannot_take_two_gt():
    gt = [(0, 0, 10, 10), (0, 0, 10, 10)]
    preds = [Pred(box=(0, 0, 10, 10), conf=0.9)]
    m = match_predictions(gt, preds, match_iou=0.5)
    assert len(m.matched) == 1
    assert m.missed_gt == [1]
    assert m.false_pred == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.robust_stage_a.gate'`

- [ ] **Step 3: Implement IoU/contains/matching in gate.py**

Create `eval/robust_stage_a/gate.py`:

```python
"""Pure Stage-A strict-gate scoring. No torch / no YOLO — CPU, deterministic.

Coordinate convention: Box = (x1, y1, x2, y2) in pixels, x2>x1, y2>y1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Pred:
    box: Box
    conf: float


@dataclass(frozen=True)
class MatchResult:
    matched: list[tuple[int, int]]  # (gt_idx, pred_idx)
    missed_gt: list[int]            # gt indices with no matching pred
    false_pred: list[int]          # pred indices matching no gt


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def contains(outer: Box, inner: Box, tol: float = 2.0) -> bool:
    """True if `inner` lies inside `outer`, allowing `tol` px slack per side."""
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] <= outer[2] + tol
        and inner[3] <= outer[3] + tol
    )


def match_predictions(
    gt_boxes: list[Box], preds: list[Pred], match_iou: float = 0.5
) -> MatchResult:
    """Greedy match: highest-confidence preds first, each to the best
    still-unmatched GT with IoU >= match_iou. One pred ↔ at most one GT."""
    order = sorted(range(len(preds)), key=lambda i: preds[i].conf, reverse=True)
    used_gt: set[int] = set()
    matched: list[tuple[int, int]] = []
    false_pred: list[int] = []
    for pi in order:
        best_gi, best_iou = -1, match_iou
        for gi, gb in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            v = iou(gb, preds[pi].box)
            if v >= best_iou:
                best_gi, best_iou = gi, v
        if best_gi >= 0:
            used_gt.add(best_gi)
            matched.append((best_gi, pi))
        else:
            false_pred.append(pi)
    matched.sort()
    missed_gt = [gi for gi in range(len(gt_boxes)) if gi not in used_gt]
    return MatchResult(matched=matched, missed_gt=missed_gt,
                       false_pred=sorted(false_pred))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/robust_stage_a/gate.py tests/robust_stage_a/test_gate.py
git commit -m "feat(eval): robust-stage-a IoU + greedy pred/GT matching"
git push origin main
```

---

## Task 3: Per-scenario scoring (geometry, lyric-clip, binary verdict)

**Files:**
- Modify: `eval/robust_stage_a/gate.py`
- Test: `tests/robust_stage_a/test_gate.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/robust_stage_a/test_gate.py`:

```python
from eval.robust_stage_a.gate import score_scenario  # noqa: E402
from eval.robust_stage_a.manifest import GtSystem, Scenario  # noqa: E402


def _sc(gt_systems, is_non_music=False):
    return Scenario("s", "arch", "img.png", is_non_music, gt_systems)


def test_scenario_perfect_passes():
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    preds = [Pred((0, 0, 100, 50), 0.9)]
    r = score_scenario(sc, preds)
    assert (r.false, r.missed, r.geometry_fail, r.lyric_clip) == (0, 0, 0, 0)
    assert r.passed is True


def test_scenario_false_system_fails():
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    preds = [Pred((0, 0, 100, 50), 0.9), Pred((300, 300, 400, 350), 0.8)]
    r = score_scenario(sc, preds)
    assert r.false == 1 and r.passed is False


def test_scenario_missed_system_fails():
    sc = _sc([GtSystem((0, 0, 100, 50), False, []),
              GtSystem((0, 100, 100, 150), False, [])])
    preds = [Pred((0, 0, 100, 50), 0.9)]
    r = score_scenario(sc, preds)
    assert r.missed == 1 and r.passed is False


def test_scenario_geometry_fail_when_pred_does_not_cover_gt():
    # pred matches by IoU>=0.5 but does not contain the full GT system
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    preds = [Pred((0, 0, 100, 40), 0.9)]  # iou 4000/5000=0.8, but clips bottom
    r = score_scenario(sc, preds)
    assert r.geometry_fail == 1 and r.lyric_clip == 0 and r.passed is False


def test_scenario_lyric_clip_counts_as_geometry_and_lyricclip():
    sc = _sc([GtSystem((0, 0, 100, 60), True, [(5, 50, 95, 60)])])
    # pred covers the staff but cuts off the lyric band (y stops at 45)
    preds = [Pred((0, 0, 100, 45), 0.95)]
    r = score_scenario(sc, preds)
    assert r.geometry_fail == 1 and r.lyric_clip == 1 and r.passed is False


def test_non_music_scenario_passes_with_zero_preds():
    sc = _sc([], is_non_music=True)
    assert score_scenario(sc, []).passed is True


def test_non_music_scenario_any_pred_is_false():
    sc = _sc([], is_non_music=True)
    r = score_scenario(sc, [Pred((0, 0, 10, 10), 0.99)])
    assert r.false == 1 and r.passed is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v -k scenario`
Expected: FAIL — `ImportError: cannot import name 'score_scenario'`

- [ ] **Step 3: Implement `ScenarioResult` + `score_scenario`**

Append to `eval/robust_stage_a/gate.py`:

```python
from eval.robust_stage_a.manifest import Scenario  # noqa: E402


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    archetype: str
    false: int
    missed: int
    geometry_fail: int
    lyric_clip: int
    passed: bool


def _geometry_ok(gt, pred_box: Box, tol: float) -> tuple[bool, bool]:
    """Returns (geometry_ok, lyric_clipped). geometry_ok requires the pred to
    contain the GT system box and every lyric band; a missed lyric band sets
    lyric_clipped True (and geometry_ok False)."""
    sys_ok = contains(pred_box, gt.box, tol)
    lyric_clipped = any(
        not contains(pred_box, lb, tol) for lb in gt.lyric_bands
    )
    return (sys_ok and not lyric_clipped), lyric_clipped


def score_scenario(
    scenario: Scenario,
    preds: list[Pred],
    match_iou: float = 0.5,
    contain_tol: float = 2.0,
) -> ScenarioResult:
    if scenario.is_non_music:
        n_false = len(preds)
        return ScenarioResult(
            scenario.scenario_id, scenario.archetype,
            false=n_false, missed=0, geometry_fail=0, lyric_clip=0,
            passed=(n_false == 0),
        )

    gt_boxes = [g.box for g in scenario.gt_systems]
    m = match_predictions(gt_boxes, preds, match_iou=match_iou)
    geometry_fail = 0
    lyric_clip = 0
    for gi, pi in m.matched:
        ok, clipped = _geometry_ok(
            scenario.gt_systems[gi], preds[pi].box, contain_tol)
        if not ok:
            geometry_fail += 1
        if clipped:
            lyric_clip += 1
    false = len(m.false_pred)
    missed = len(m.missed_gt)
    passed = (false == 0 and missed == 0 and geometry_fail == 0)
    return ScenarioResult(
        scenario.scenario_id, scenario.archetype,
        false=false, missed=missed, geometry_fail=geometry_fail,
        lyric_clip=lyric_clip, passed=passed,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v`
Expected: all PASS (Task 2 + Task 3 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/robust_stage_a/gate.py tests/robust_stage_a/test_gate.py
git commit -m "feat(eval): robust-stage-a per-scenario scoring (geometry, lyric-clip, binary verdict)"
git push origin main
```

---

## Task 4: Lyric-system recall + combined gate verdict

**Files:**
- Modify: `eval/robust_stage_a/gate.py`
- Test: `tests/robust_stage_a/test_gate.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/robust_stage_a/test_gate.py`:

```python
from eval.robust_stage_a.gate import (  # noqa: E402
    lyric_system_recall,
    combined_gate,
)


def test_lyric_system_recall_counts_only_lyric_systems():
    sc = _sc([
        GtSystem((0, 0, 100, 60), True, [(5, 50, 95, 60)]),    # lyric, detected
        GtSystem((0, 100, 100, 160), True, [(5, 150, 95, 160)]),  # lyric, clipped
        GtSystem((0, 200, 100, 250), False, []),                # non-lyric, ignored
    ])
    preds = [
        Pred((0, 0, 100, 60), 0.9),       # covers sys0 + its lyric band
        Pred((0, 100, 100, 145), 0.9),    # clips sys1 lyric band
        Pred((0, 200, 100, 250), 0.9),
    ]
    # 2 lyric systems, 1 detected cleanly -> 0.5
    assert lyric_system_recall([sc], {"s": preds}) == pytest.approx(0.5)


def test_lyric_system_recall_is_one_when_no_lyric_systems():
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    assert lyric_system_recall([sc], {"s": [Pred((0, 0, 100, 50), 0.9)]}) == 1.0


def test_combined_gate_pass():
    sc = _sc([GtSystem((0, 0, 100, 50), False, [])])
    results = [score_scenario(sc, [Pred((0, 0, 100, 50), 0.9)])]
    v = combined_gate(results, lyric_recall=1.0, lyric_recall_baseline=1.0,
                       lieder_recall=0.94, lieder_baseline=0.93)
    assert v.passed is True
    assert v.n_failed_scenarios == 0


def test_combined_gate_fails_on_one_bad_scenario():
    sc_ok = Scenario("ok", "a", "i", False, [GtSystem((0, 0, 10, 10), False, [])])
    sc_bad = Scenario("bad", "a", "j", True, [])
    results = [
        score_scenario(sc_ok, [Pred((0, 0, 10, 10), 0.9)]),
        score_scenario(sc_bad, [Pred((0, 0, 5, 5), 0.9)]),  # pred on non-music
    ]
    v = combined_gate(results, 1.0, 1.0, 0.94, 0.93)
    assert v.passed is False
    assert v.failed_scenario_ids == ["bad"]


def test_combined_gate_fails_on_lieder_regression():
    sc = _sc([GtSystem((0, 0, 10, 10), False, [])])
    results = [score_scenario(sc, [Pred((0, 0, 10, 10), 0.9)])]
    v = combined_gate(results, 1.0, 1.0, lieder_recall=0.92, lieder_baseline=0.93)
    assert v.passed is False


def test_combined_gate_fails_on_lyric_recall_regression():
    sc = _sc([GtSystem((0, 0, 10, 10), False, [])])
    results = [score_scenario(sc, [Pred((0, 0, 10, 10), 0.9)])]
    v = combined_gate(results, lyric_recall=0.80, lyric_recall_baseline=0.95,
                      lieder_recall=0.94, lieder_baseline=0.93)
    assert v.passed is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v -k "lyric or combined"`
Expected: FAIL — `ImportError: cannot import name 'lyric_system_recall'`

- [ ] **Step 3: Implement the sub-metric + combined gate**

Append to `eval/robust_stage_a/gate.py`:

```python
NO_REGRESSION_EPS = 0.0  # strict: new must be >= baseline (per spec)


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    n_scenarios: int
    n_failed_scenarios: int
    failed_scenario_ids: list[str]
    lyric_recall: float
    lyric_recall_baseline: float
    lieder_recall: float
    lieder_baseline: float


def lyric_system_recall(
    scenarios: list[Scenario],
    preds_by_scenario: dict[str, list[Pred]],
    match_iou: float = 0.5,
    contain_tol: float = 2.0,
) -> float:
    """Fraction of GT systems that carry lyrics which are detected with a
    matching pred whose box also fully contains the lyric band(s).
    1.0 when there are no lyric-bearing GT systems."""
    total = 0
    detected = 0
    for sc in scenarios:
        if sc.is_non_music:
            continue
        preds = preds_by_scenario.get(sc.scenario_id, [])
        m = match_predictions([g.box for g in sc.gt_systems], preds,
                              match_iou=match_iou)
        matched_pred_for_gt = {gi: pi for gi, pi in m.matched}
        for gi, g in enumerate(sc.gt_systems):
            if not g.has_lyrics:
                continue
            total += 1
            pi = matched_pred_for_gt.get(gi)
            if pi is None:
                continue
            ok, _ = _geometry_ok(g, preds[pi].box, contain_tol)
            if ok:
                detected += 1
    return 1.0 if total == 0 else detected / total


def combined_gate(
    scenario_results: list[ScenarioResult],
    lyric_recall: float,
    lyric_recall_baseline: float,
    lieder_recall: float,
    lieder_baseline: float,
    eps: float = NO_REGRESSION_EPS,
) -> GateVerdict:
    failed = [r.scenario_id for r in scenario_results if not r.passed]
    all_scenarios_pass = not failed
    no_lieder_regression = lieder_recall >= lieder_baseline - eps
    no_lyric_regression = lyric_recall >= lyric_recall_baseline - eps
    return GateVerdict(
        passed=(all_scenarios_pass and no_lieder_regression
                and no_lyric_regression),
        n_scenarios=len(scenario_results),
        n_failed_scenarios=len(failed),
        failed_scenario_ids=failed,
        lyric_recall=lyric_recall,
        lyric_recall_baseline=lyric_recall_baseline,
        lieder_recall=lieder_recall,
        lieder_baseline=lieder_baseline,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add eval/robust_stage_a/gate.py tests/robust_stage_a/test_gate.py
git commit -m "feat(eval): robust-stage-a lyric-system recall + combined gate verdict"
git push origin main
```

---

## Task 5: Lieder-baseline recall reader (reuse existing scorer output)

**Files:**
- Modify: `eval/robust_stage_a/gate.py`
- Test: `tests/robust_stage_a/test_gate.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `tests/robust_stage_a/test_gate.py`:

```python
from eval.robust_stage_a.gate import recall_from_stagea_csv  # noqa: E402


def test_recall_from_stagea_csv(tmp_path):
    # mirrors eval/score_stage_a_only.py output schema
    csv = tmp_path / "r.csv"
    csv.write_text(
        "piece,expected_p1_staves,detected_p1_staves,missing_count\n"
        "a,3,3,0\n"
        "b,4,2,2\n",
        encoding="utf-8",
    )
    # total expected 7, detected 5 -> 5/7
    assert recall_from_stagea_csv(csv) == pytest.approx(5 / 7)


def test_recall_from_stagea_csv_empty_is_zero(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("piece,expected_p1_staves,detected_p1_staves,missing_count\n",
                   encoding="utf-8")
    assert recall_from_stagea_csv(csv) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v -k recall_from`
Expected: FAIL — `ImportError: cannot import name 'recall_from_stagea_csv'`

- [ ] **Step 3: Implement the CSV recall reader**

Append to `eval/robust_stage_a/gate.py`:

```python
import csv as _csv  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def recall_from_stagea_csv(path: str | _Path) -> float:
    """Aggregate recall from an eval/score_stage_a_only.py CSV:
    sum(detected_p1_staves) / sum(expected_p1_staves). 0.0 if no rows."""
    total_expected = 0
    total_detected = 0
    with _Path(path).open(newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            total_expected += int(row["expected_p1_staves"])
            total_detected += int(row["detected_p1_staves"])
    return (total_detected / total_expected) if total_expected else 0.0
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add eval/robust_stage_a/gate.py tests/robust_stage_a/test_gate.py
git commit -m "feat(eval): robust-stage-a lieder-recall reader for score_stage_a_only CSV"
git push origin main
```

---

## Task 6: CLI orchestrator (`run_gate.py`) + usage doc

`run_gate.py` is the thin GPU/seder integration seam: it runs Stage-A inference over the held-out images, runs the existing lieder scorer for the new checkpoint, then feeds plain boxes/numbers into the pure gate. Inference reuses the project's existing Stage-A path (the same one `scripts/audit/dump_system_crops.py` uses); this task wires it, it does not reinvent detection.

**Files:**
- Create: `eval/robust_stage_a/run_gate.py`
- Modify: `eval/robust_stage_a/gate.py` (add `verdict_to_report` text formatter)
- Test: `tests/robust_stage_a/test_gate.py` (append — formatter only; inference is integration-run on seder)

- [ ] **Step 1: Append the failing test (report formatter)**

Append to `tests/robust_stage_a/test_gate.py`:

```python
from eval.robust_stage_a.gate import verdict_to_report  # noqa: E402


def test_verdict_to_report_pass_and_fail():
    sc = _sc([GtSystem((0, 0, 10, 10), False, [])])
    ok = score_scenario(sc, [Pred((0, 0, 10, 10), 0.9)])
    v = combined_gate([ok], 1.0, 1.0, 0.94, 0.93)
    txt = verdict_to_report(v, [ok])
    assert "GATE: PASS" in txt
    assert "scenarios: 1/1 passed" in txt

    bad = score_scenario(Scenario("bad", "deed", "j", True, []),
                          [Pred((0, 0, 5, 5), 0.9)])
    v2 = combined_gate([bad], 1.0, 1.0, 0.94, 0.93)
    txt2 = verdict_to_report(v2, [bad])
    assert "GATE: FAIL" in txt2
    assert "bad" in txt2 and "deed" in txt2
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v -k verdict_to_report`
Expected: FAIL — `ImportError: cannot import name 'verdict_to_report'`

- [ ] **Step 3: Implement `verdict_to_report`**

Append to `eval/robust_stage_a/gate.py`:

```python
def verdict_to_report(
    verdict: GateVerdict, scenario_results: list[ScenarioResult]
) -> str:
    lines = []
    lines.append(f"GATE: {'PASS' if verdict.passed else 'FAIL'}")
    lines.append(
        f"scenarios: {verdict.n_scenarios - verdict.n_failed_scenarios}"
        f"/{verdict.n_scenarios} passed"
    )
    lines.append(
        f"lieder_recall: {verdict.lieder_recall:.4f} "
        f"(baseline {verdict.lieder_baseline:.4f}, "
        f"{'OK' if verdict.lieder_recall >= verdict.lieder_baseline else 'REGRESSION'})"
    )
    lines.append(
        f"lyric_system_recall: {verdict.lyric_recall:.4f} "
        f"(baseline {verdict.lyric_recall_baseline:.4f}, "
        f"{'OK' if verdict.lyric_recall >= verdict.lyric_recall_baseline else 'REGRESSION'})"
    )
    for r in scenario_results:
        flag = "PASS" if r.passed else "FAIL"
        lines.append(
            f"  [{flag}] {r.scenario_id} ({r.archetype}) "
            f"false={r.false} missed={r.missed} "
            f"geom={r.geometry_fail} lyric_clip={r.lyric_clip}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/robust_stage_a/test_gate.py -v`
Expected: all PASS

- [ ] **Step 5: Implement the CLI orchestrator**

Create `eval/robust_stage_a/run_gate.py`:

```python
"""Run the strict Stage-A gate for a given YOLO checkpoint.

Inference (GPU) + the existing lieder scorer are orchestrated here; the
verdict logic is the pure eval.robust_stage_a.gate module. Intended to run
on seder (GPU). The pure gate is unit-tested separately on CPU.

Example (on seder, repo root):
  venv-cu132/Scripts/python.exe -u -m eval.robust_stage_a.run_gate \
    --manifest eval/robust_stage_a/heldout/manifest.json \
    --yolo-weights runs/detect/runs/yolo26m_systems_faintink/weights/best.pt \
    --lieder-csv eval/results/stagea_faintink.csv \
    --lieder-baseline-csv eval/results/stagea_baseline_pre_faintink.csv \
    --lyric-baseline-out eval/robust_stage_a/lyric_baseline.json \
    --report-out eval/robust_stage_a/gate_report.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.robust_stage_a.gate import (  # noqa: E402
    Pred,
    combined_gate,
    lyric_system_recall,
    recall_from_stagea_csv,
    score_scenario,
    verdict_to_report,
)
from eval.robust_stage_a.manifest import load_manifest  # noqa: E402


def _infer(image_path: str, weights: str, conf: float) -> list[Pred]:
    """Run the Stage-A YOLO detector on one image. Uses Ultralytics directly
    (same engine the existing scripts/audit/dump_system_crops.py uses)."""
    from ultralytics import YOLO  # local import: keeps the module CPU-importable

    model = YOLO(weights)
    res = model.predict(image_path, conf=conf, verbose=False)[0]
    out: list[Pred] = []
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        out.append(Pred(box=(x1, y1, x2, y2), conf=float(b.conf[0])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--image-root", type=Path, default=None,
                    help="prefix for relative manifest image paths")
    ap.add_argument("--lieder-csv", type=Path, required=True,
                    help="score_stage_a_only.py CSV for the NEW checkpoint")
    ap.add_argument("--lieder-baseline-csv", type=Path,
                    default=Path("eval/results/stagea_baseline_pre_faintink.csv"))
    ap.add_argument("--lyric-baseline", type=Path, default=None,
                    help="JSON with prior lyric_system_recall to compare against")
    ap.add_argument("--lyric-baseline-out", type=Path, default=None,
                    help="write the computed lyric_system_recall here (snapshot)")
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args()

    scenarios = load_manifest(args.manifest)
    preds_by_scenario: dict[str, list[Pred]] = {}
    results = []
    for sc in scenarios:
        img = sc.image
        if args.image_root is not None and not Path(img).is_absolute():
            img = str(args.image_root / img)
        preds = _infer(img, str(args.yolo_weights), args.conf)
        preds_by_scenario[sc.scenario_id] = preds
        results.append(score_scenario(sc, preds))

    lyric_recall = lyric_system_recall(scenarios, preds_by_scenario)
    if args.lyric_baseline_out is not None:
        args.lyric_baseline_out.write_text(
            json.dumps({"lyric_system_recall": lyric_recall}), encoding="utf-8")
    if args.lyric_baseline is not None:
        lyric_baseline = json.loads(
            args.lyric_baseline.read_text(encoding="utf-8"))["lyric_system_recall"]
    else:
        lyric_baseline = lyric_recall  # first run: self-baseline (no regression)

    lieder_recall = recall_from_stagea_csv(args.lieder_csv)
    lieder_baseline = recall_from_stagea_csv(args.lieder_baseline_csv)

    verdict = combined_gate(results, lyric_recall, lyric_baseline,
                            lieder_recall, lieder_baseline)
    report = verdict_to_report(verdict, results)
    print(report)
    if args.report_out is not None:
        args.report_out.write_text(report + "\n", encoding="utf-8")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Smoke the CLI argument wiring (no GPU)**

Run: `python3 -m eval.robust_stage_a.run_gate --help`
Expected: argparse help prints listing `--manifest`, `--yolo-weights`, `--lieder-csv`, `--lyric-baseline-out`, etc., exit 0. (Confirms imports/wiring; full inference runs on seder.)

- [ ] **Step 7: Full test sweep + commit**

Run: `python3 -m pytest tests/robust_stage_a/ -v`
Expected: all PASS (manifest + gate suites), none skipped (CPU, not CUDA-gated).

```bash
git add eval/robust_stage_a/run_gate.py eval/robust_stage_a/gate.py \
        tests/robust_stage_a/test_gate.py
git commit -m "feat(eval): robust-stage-a gate CLI orchestrator + text report"
git push origin main
```

---

## Self-review

- **Spec coverage:** spec § "Eval & ship gate" → Tasks 2–4 (IoU match, per-scenario binary, geometry/lyric-clip, lieder no-regression, lyric-system recall sub-metric) + Task 6 (combined report/exit code). Spec § "Data engine" source 3 (held-out real set) → Task 1 manifest schema/validator (the user-provided set conforms to it; `is_non_music` + `lyric_bands` modeled). Spec § "lyrics-as-garbage" mitigation #4 (lyrics eval slice, clipped-lyrics = fail, sub-metric can't hide a regression) → Tasks 3–4 (`lyric_clip`, `lyric_system_recall`, strict `eps=0`). Spec § "Testing strategy" (CPU-unit-testable gate/IoU/lyric-slice, GPU only for inference) → all pure logic in `gate.py`/`manifest.py` with CPU tests; GPU isolated to `run_gate._infer`. Out-of-scope items (data engine build, retrain-hardening, the loop) are correctly NOT in this sub-plan (they are Sub-plans C/B/D).
- **Placeholder scan:** no TBD/TODO; every code step has complete runnable code; pinned-decisions section resolves the spec's deferred knobs with concrete values; the lieder/lyric baselines are sourced concretely (CSV recompute / snapshot-on-first-run), not "fill in later".
- **Type consistency:** `Box=(x1,y1,x2,y2)` float 4-tuple used uniformly across `manifest.py` and `gate.py`; `Pred(box,conf)`, `GtSystem(box,has_lyrics,lyric_bands)`, `Scenario(...)`, `ScenarioResult(...)`, `GateVerdict(...)`, `MatchResult(matched,missed_gt,false_pred)` defined once and consumed with identical signatures in later tasks and in `run_gate.py`. `match_predictions`, `score_scenario`, `lyric_system_recall`, `combined_gate`, `recall_from_stagea_csv`, `verdict_to_report` names are stable across tasks. `gate.py` has no torch import (CPU-test invariant) — inference torch/ultralytics import is local inside `run_gate._infer`.
