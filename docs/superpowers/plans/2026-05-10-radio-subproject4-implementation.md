# Subproject 4 — Per-System Inference Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-system end-to-end inference pipeline (PDF → MusicXML) plus the 50-piece lieder eval driver, as specified in [`docs/superpowers/specs/2026-05-10-radio-subproject4-design.md`](../specs/2026-05-10-radio-subproject4-design.md).

**Architecture:** Library + thin CLI for inference (`SystemInferencePipeline` loads YOLO + Stage B once via a shared `load_stage_b_for_inference` helper). Two-pass eval: Phase 1 inference loop is in-process (resilient via per-piece try/except, writes `.musicxml` + `.musicxml.diagnostics.json` immediately); Phase 2 scoring stays subprocess-isolated (lifted from archive — protects against music21/zss memory accumulation that previously hit 43 GB at piece 6/20).

**Tech Stack:** Python 3.14, PyTorch (GPU box only), ultralytics YOLO, PyMuPDF, music21 (in scoring subprocess only), pytest with `unittest.mock`.

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `src/inference/checkpoint_load.py` | NEW | `StageBInferenceBundle` dataclass + `load_stage_b_for_inference()` — single source of truth for Stage B inference loading |
| `src/models/yolo_stage_a_systems.py` | NEW | `YoloStageASystems` — system-level YOLO wrapper, applies brace margin extension |
| `src/pipeline/assemble_score.py` | EDIT | Add `assemble_score_from_system_predictions()` next to existing `assemble_score()` |
| `src/inference/system_pipeline.py` | NEW | `SystemInferencePipeline` library class with `run_pdf` / `run_page` / `run_system_crop` / `export_musicxml` |
| `src/cli/__init__.py` | NEW (empty) | Make `src/cli/` a package |
| `src/cli/run_system_inference.py` | NEW | argparse CLI wrapping `SystemInferencePipeline` |
| `eval/run_lieder_eval.py` | LIFT + EDIT | From `archive/per_staff/eval/`. Inference-only (no scoring inline); rewired to `SystemInferencePipeline` |
| `eval/score_lieder_eval.py` | LIFT + EDIT | From `archive/per_staff/eval/`. Add `--tedn` flag (default off). Keep subprocess-per-piece + memory throttle |
| `tests/inference/__init__.py` | NEW (empty) | Make `tests/inference/` a package if not already |
| `tests/inference/test_checkpoint_load.py` | NEW | Mocks heavy construction; exercises real chain through factory config + components |
| `tests/models/test_yolo_stage_a_systems.py` | NEW | Fake YOLO; verifies sort, brace extension, system_index assignment |
| `tests/pipeline/test_assemble_from_system_predictions.py` | NEW | Synthetic system token lists; verifies split + even-split y + system_index_hint |
| `tests/inference/test_system_pipeline.py` | NEW | Mocked YOLO + Stage B bundle; verifies single-page chain |
| `tests/inference/test_system_pipeline_pdf.py` | NEW | Single-page fixture + mocks; verifies `run_pdf` shape and `export_musicxml` writes both artifacts |
| `tests/cli/__init__.py` | NEW (empty) | Package marker |
| `tests/cli/test_run_system_inference.py` | NEW | argparse smoke; asserts no `--vocab` flag |
| `eval/tests/test_run_lieder_eval.py` | LIFT + EDIT | From archive; asserts no inline scoring + subprocess for `--run-scoring` |

---

## Task 0: Branch off main

**Files:** none (git only)

- [ ] **Step 0.1: Create the branch from current main**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b feat/subproject4-system-inference
```

Expected: branch `feat/subproject4-system-inference` created at HEAD `06aa4be` (or later if main has advanced).

- [ ] **Step 0.2: Verify clean state**

```bash
git status
```

Expected: `nothing to commit, working tree clean` on the new branch.

---

## Task 1: `StageBInferenceBundle` dataclass + scaffolding

**Files:**
- Create: `src/inference/checkpoint_load.py`
- Test: `tests/inference/test_checkpoint_load.py`
- Create (if missing): `tests/inference/__init__.py`

- [ ] **Step 1.1: Ensure tests/inference is a package**

```bash
ls tests/inference/__init__.py 2>/dev/null || touch tests/inference/__init__.py
```

- [ ] **Step 1.2: Write the failing test for the bundle dataclass shape**

Create `tests/inference/test_checkpoint_load.py`:

```python
"""Unit tests for src.inference.checkpoint_load."""
from __future__ import annotations

from dataclasses import is_dataclass


def test_bundle_is_dataclass_with_expected_fields():
    from src.inference.checkpoint_load import StageBInferenceBundle

    assert is_dataclass(StageBInferenceBundle)
    fields = {f.name for f in StageBInferenceBundle.__dataclass_fields__.values()}
    assert fields == {
        "model",
        "decode_model",
        "vocab",
        "token_to_idx",
        "use_fp16",
        "factory_cfg",
    }
```

- [ ] **Step 1.3: Run the test, confirm it fails**

```bash
python3 -m pytest tests/inference/test_checkpoint_load.py::test_bundle_is_dataclass_with_expected_fields -x --no-header -q
```

Expected: `ModuleNotFoundError: No module named 'src.inference.checkpoint_load'`.

- [ ] **Step 1.4: Create the module with just the dataclass**

Create `src/inference/checkpoint_load.py`:

```python
"""Single-source-of-truth loader for Stage B checkpoints used by inference.

Factored from src/eval/evaluate_stage_b_checkpoint.py:285-345 — keeps the
8-step Stage B inference loading sequence in one place so callers
(SystemInferencePipeline, debugging notebooks, future refactors) cannot
get the order or arguments wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn  # noqa: F401
    from src.tokenizer.vocab import OMRVocabulary
    from src.train.model_factory import ModelFactoryConfig


@dataclass
class StageBInferenceBundle:
    """Fully-loaded, eval-mode, inference-ready Stage B model bundle."""
    model: "nn.Module"
    decode_model: "nn.Module"
    vocab: "OMRVocabulary"
    token_to_idx: Dict[str, int]
    use_fp16: bool
    factory_cfg: "ModelFactoryConfig"
```

- [ ] **Step 1.5: Run the test, confirm it passes**

```bash
python3 -m pytest tests/inference/test_checkpoint_load.py::test_bundle_is_dataclass_with_expected_fields -x --no-header -q
```

Expected: `1 passed`.

- [ ] **Step 1.6: Commit**

```bash
git add src/inference/checkpoint_load.py tests/inference/test_checkpoint_load.py tests/inference/__init__.py
git commit -m "feat(inference): scaffold StageBInferenceBundle dataclass"
```

---

## Task 2: `load_stage_b_for_inference()` function

**Files:**
- Modify: `src/inference/checkpoint_load.py`
- Modify: `tests/inference/test_checkpoint_load.py`

- [ ] **Step 2.1: Write the failing test for the loader**

Append to `tests/inference/test_checkpoint_load.py`:

```python
def test_load_stage_b_for_inference_chains_existing_helpers(monkeypatch, tmp_path):
    """Verify the loader exercises the real factory_cfg + components chain
    while mocking heavy torch.load + checkpoint loading + decoder prep."""
    from unittest.mock import MagicMock

    import src.inference.checkpoint_load as cl

    # Mock torch.load to return a fake payload
    fake_payload = {"factory_config": {"stage_b_vocab_size": 1234, "decoder_d_model": 512}}
    monkeypatch.setattr(cl, "torch", MagicMock(load=MagicMock(return_value=fake_payload)))

    fake_vocab = MagicMock()
    fake_vocab.size = 1234
    fake_vocab.tokens = ["<bos>", "<eos>", "<staff_end>"]
    monkeypatch.setattr(cl, "build_default_vocabulary", lambda: fake_vocab)

    fake_factory_cfg = MagicMock()
    monkeypatch.setattr(
        cl,
        "model_factory_config_from_checkpoint_payload",
        lambda payload, vocab_size, fallback: fake_factory_cfg,
    )

    fake_model = MagicMock()
    fake_dora_cfg = MagicMock()
    monkeypatch.setattr(
        cl,
        "build_stage_b_components",
        lambda factory_cfg: {"model": fake_model, "dora_config": fake_dora_cfg},
    )

    monkeypatch.setattr(
        cl,
        "load_stage_b_checkpoint",
        lambda **kwargs: {"_model": kwargs["model"], "checkpoint_format": "v1",
                          "loaded_keys": [], "load_ratio": 1.0},
    )

    fake_decode_model = MagicMock()
    monkeypatch.setattr(
        cl,
        "_prepare_model_for_inference",
        lambda model, device, *, use_fp16=False, quantize=False: (fake_decode_model, use_fp16),
    )

    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"")  # exists for path validation

    bundle = cl.load_stage_b_for_inference(ckpt, device="cpu", use_fp16=True)

    assert bundle.model is fake_model
    assert bundle.decode_model is fake_decode_model
    assert bundle.vocab is fake_vocab
    assert bundle.use_fp16 is True
    assert bundle.factory_cfg is fake_factory_cfg
    assert bundle.token_to_idx == {"<bos>": 0, "<eos>": 1, "<staff_end>": 2}
    fake_model.eval.assert_called_once()
```

- [ ] **Step 2.2: Run the test, confirm it fails**

```bash
python3 -m pytest tests/inference/test_checkpoint_load.py::test_load_stage_b_for_inference_chains_existing_helpers -x --no-header -q
```

Expected: `AttributeError: module 'src.inference.checkpoint_load' has no attribute 'load_stage_b_for_inference'`.

- [ ] **Step 2.3: Implement the loader**

Append to `src/inference/checkpoint_load.py`:

```python
import torch  # noqa: E402

from src.checkpoint_io import load_stage_b_checkpoint  # noqa: E402
from src.inference.decoder_runtime import _prepare_model_for_inference  # noqa: E402
from src.tokenizer.vocab import build_default_vocabulary  # noqa: E402
from src.train.model_factory import (  # noqa: E402
    ModelFactoryConfig,
    build_stage_b_components,
    model_factory_config_from_checkpoint_payload,
)


def load_stage_b_for_inference(
    checkpoint_path: Path,
    device,
    *,
    use_fp16: bool = False,
    quantize: bool = False,
) -> StageBInferenceBundle:
    """Build vocab → torch.load payload → infer ModelFactoryConfig →
    build_stage_b_components → load_stage_b_checkpoint → eval() →
    _prepare_model_for_inference.

    Returns the fully-prepared inference-ready bundle.
    """
    vocab = build_default_vocabulary()
    payload = torch.load(str(checkpoint_path), map_location=device)
    fallback = ModelFactoryConfig(stage_b_vocab_size=vocab.size)
    factory_cfg = model_factory_config_from_checkpoint_payload(
        payload, vocab_size=vocab.size, fallback=fallback,
    )
    components = build_stage_b_components(factory_cfg)
    model = components["model"]
    ckpt_result = load_stage_b_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        dora_config=components.get("dora_config"),
        min_coverage=0.50,
    )
    model = ckpt_result["_model"]
    model.eval()
    decode_model, use_fp16_resolved = _prepare_model_for_inference(
        model, device, use_fp16=use_fp16, quantize=quantize,
    )
    token_to_idx = {token: idx for idx, token in enumerate(vocab.tokens)}
    return StageBInferenceBundle(
        model=model,
        decode_model=decode_model,
        vocab=vocab,
        token_to_idx=token_to_idx,
        use_fp16=use_fp16_resolved,
        factory_cfg=factory_cfg,
    )
```

- [ ] **Step 2.4: Run both tests in the file, confirm they pass**

```bash
python3 -m pytest tests/inference/test_checkpoint_load.py -x --no-header -q
```

Expected: `2 passed`.

- [ ] **Step 2.5: Commit**

```bash
git add src/inference/checkpoint_load.py tests/inference/test_checkpoint_load.py
git commit -m "feat(inference): add load_stage_b_for_inference helper"
```

---

## Task 3: `YoloStageASystems` wrapper

**Files:**
- Create: `src/models/yolo_stage_a_systems.py`
- Create: `tests/models/test_yolo_stage_a_systems.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/models/test_yolo_stage_a_systems.py`:

```python
"""Unit tests for src.models.yolo_stage_a_systems."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


def _fake_yolo_results(boxes_xyxy, confs):
    """Build a minimal Ultralytics-result shape that _yolo_predict_to_boxes accepts."""
    class _Boxes:
        pass
    boxes = _Boxes()
    boxes.xyxy = np.array(boxes_xyxy, dtype=np.float32)
    boxes.conf = np.array(confs, dtype=np.float32)
    result = MagicMock()
    result.boxes = boxes
    return [result]


def test_detect_systems_sorts_top_to_bottom_and_extends_left():
    from src.models.yolo_stage_a_systems import YoloStageASystems
    from src.data.generate_synthetic import V15_LEFTWARD_BRACKET_MARGIN_PX

    page = Image.new("RGB", (1000, 800), color="white")
    fake_model = MagicMock()
    # YOLO returns boxes out of order: bottom system first, top system second.
    fake_model.predict.return_value = _fake_yolo_results(
        boxes_xyxy=[[200, 500, 800, 700], [200, 100, 800, 300]],
        confs=[0.9, 0.85],
    )

    with patch("src.models.yolo_stage_a_systems.YOLO", return_value=fake_model):
        wrapper = YoloStageASystems(weights_path="dummy.pt")

    systems = wrapper.detect_systems(page)

    assert len(systems) == 2
    # Sorted top-to-bottom by y_center.
    assert systems[0]["system_index"] == 0
    assert systems[1]["system_index"] == 1
    # Top system bbox came from the second YOLO entry.
    top_x1, top_y1, _, _ = systems[0]["bbox_extended"]
    bot_y1 = systems[1]["bbox_extended"][1]
    assert top_y1 < bot_y1
    # Brace margin extension applied (x1 reduced by margin, clamped to 0).
    expected_top_x1 = max(0, 200 - V15_LEFTWARD_BRACKET_MARGIN_PX)
    assert top_x1 == expected_top_x1


def test_detect_systems_clamps_to_page_bounds():
    from src.models.yolo_stage_a_systems import YoloStageASystems

    page = Image.new("RGB", (500, 400), color="white")
    fake_model = MagicMock()
    # YOLO returns a box whose x1 is already at the left edge.
    fake_model.predict.return_value = _fake_yolo_results(
        boxes_xyxy=[[5, 100, 400, 300]],
        confs=[0.95],
    )

    with patch("src.models.yolo_stage_a_systems.YOLO", return_value=fake_model):
        wrapper = YoloStageASystems(weights_path="dummy.pt")

    systems = wrapper.detect_systems(page)
    assert len(systems) == 1
    x1, _, _, _ = systems[0]["bbox_extended"]
    assert x1 >= 0


def test_detect_systems_empty_page():
    from src.models.yolo_stage_a_systems import YoloStageASystems

    page = Image.new("RGB", (500, 400), color="white")
    fake_model = MagicMock()
    fake_model.predict.return_value = _fake_yolo_results(boxes_xyxy=[], confs=[])

    with patch("src.models.yolo_stage_a_systems.YOLO", return_value=fake_model):
        wrapper = YoloStageASystems(weights_path="dummy.pt")

    assert wrapper.detect_systems(page) == []
```

- [ ] **Step 3.2: Run the test, confirm it fails**

```bash
python3 -m pytest tests/models/test_yolo_stage_a_systems.py -x --no-header -q
```

Expected: `ModuleNotFoundError: No module named 'src.models.yolo_stage_a_systems'`.

- [ ] **Step 3.3: Implement the wrapper**

Create `src/models/yolo_stage_a_systems.py`:

```python
"""System-level Stage A YOLO wrapper for inference.

Loads the system-detection YOLO checkpoint, runs it on a page image, and
returns sorted top-to-bottom system bboxes with the brace-margin extension
already applied. Designed for inference only; training-time data prep uses
src/data/yolo_aligned_systems.py with oracle matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.data.yolo_common import _yolo_predict_to_boxes
from src.models.system_postprocess import extend_left_for_brace


class YoloStageASystems:
    def __init__(self, weights_path: Path, *, conf: float = 0.25, imgsz: int = 1920):
        self._model = YOLO(str(weights_path))
        self._conf = conf
        self._imgsz = imgsz

    def detect_systems(self, page_image: Image.Image) -> List[dict]:
        """Run YOLO on the page, sort top-to-bottom, apply brace margin extension.

        Returns: list of {system_index, bbox_extended, conf} dicts.
        bbox_extended is a 4-tuple (x1, y1, x2, y2). Sorted top-to-bottom by
        y_center; system_index is assigned 0..N-1 after sorting.
        """
        raw = _yolo_predict_to_boxes(
            self._model, page_image, imgsz=self._imgsz, conf=self._conf,
        )
        if not raw:
            return []

        # Sort top-to-bottom by y_center.
        raw.sort(key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2)

        # Apply brace margin extension in one batch.
        boxes_array = np.array([b["bbox"] for b in raw], dtype=np.float64)
        extended = extend_left_for_brace(boxes_array, page_w=page_image.width)

        return [
            {
                "system_index": idx,
                "bbox_extended": tuple(float(v) for v in extended[idx]),
                "conf": float(raw[idx]["conf"]),
            }
            for idx in range(len(raw))
        ]
```

- [ ] **Step 3.4: Run the tests, confirm they pass**

```bash
python3 -m pytest tests/models/test_yolo_stage_a_systems.py -x --no-header -q
```

Expected: `3 passed`.

- [ ] **Step 3.5: Commit**

```bash
git add src/models/yolo_stage_a_systems.py tests/models/test_yolo_stage_a_systems.py
git commit -m "feat(models): add YoloStageASystems inference wrapper"
```

---

## Task 4: `assemble_score_from_system_predictions()` extension

**Files:**
- Modify: `src/pipeline/assemble_score.py` (append new function)
- Create: `tests/pipeline/test_assemble_from_system_predictions.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/pipeline/test_assemble_from_system_predictions.py`:

```python
"""Unit tests for assemble_score_from_system_predictions."""
from __future__ import annotations


def _make_grandstaff_system_tokens():
    """Two-staff (piano grand-staff) system with minimal note content."""
    return [
        "<bos>",
        "<staff_start>", "<staff_idx_0>",
        "<measure_start>", "note-C4-quarter", "<measure_end>", "<staff_end>",
        "<staff_start>", "<staff_idx_1>",
        "<measure_start>", "note-C3-quarter", "<measure_end>", "<staff_end>",
        "<eos>",
    ]


def test_single_system_two_staves_yields_two_staves_in_assembled_score():
    from src.pipeline.assemble_score import assemble_score_from_system_predictions

    sys_tokens = _make_grandstaff_system_tokens()
    sys_loc = {
        "system_index": 0,
        "bbox": (50.0, 100.0, 950.0, 300.0),
        "page_index": 0,
        "conf": 0.95,
    }

    score = assemble_score_from_system_predictions([sys_tokens], [sys_loc])

    # Existing AssembledScore exposes per-system staff structure.
    assert len(score.systems) == 1
    assert len(score.systems[0].staves) == 2


def test_even_split_y_coords_within_a_system():
    """Each staff's StaffLocation y-range is the system bbox split by N=2."""
    from src.pipeline.assemble_score import assemble_score_from_system_predictions

    sys_tokens = _make_grandstaff_system_tokens()
    sys_loc = {
        "system_index": 3,
        "bbox": (0.0, 100.0, 1000.0, 300.0),  # height = 200, two staves => 100 each
        "page_index": 0,
        "conf": 0.9,
    }
    score = assemble_score_from_system_predictions([sys_tokens], [sys_loc])

    staves = score.systems[0].staves
    # First staff: y_top=100, y_bottom=200. Second: y_top=200, y_bottom=300.
    assert staves[0].location.y_top == 100.0
    assert staves[0].location.y_bottom == 200.0
    assert staves[1].location.y_top == 200.0
    assert staves[1].location.y_bottom == 300.0
    assert staves[0].location.x_left == 0.0
    assert staves[0].location.x_right == 1000.0


def test_system_index_hint_is_preserved():
    from src.pipeline.assemble_score import assemble_score_from_system_predictions

    sys_tokens = _make_grandstaff_system_tokens()
    sys_loc = {
        "system_index": 7,
        "bbox": (0.0, 0.0, 100.0, 100.0),
        "page_index": 2,
        "conf": 0.5,
    }
    score = assemble_score_from_system_predictions([sys_tokens], [sys_loc])

    for staff in score.systems[0].staves:
        assert staff.system_index_hint == 7
        assert staff.location.page_index == 2


def test_empty_system_token_list_yields_empty_score():
    from src.pipeline.assemble_score import assemble_score_from_system_predictions

    score = assemble_score_from_system_predictions([], [])
    assert len(score.systems) == 0
```

- [ ] **Step 4.2: Run the test, confirm it fails**

```bash
python3 -m pytest tests/pipeline/test_assemble_from_system_predictions.py -x --no-header -q
```

Expected: `ImportError: cannot import name 'assemble_score_from_system_predictions'`.

- [ ] **Step 4.3: Append the new function**

Append to `src/pipeline/assemble_score.py` (after the existing `assemble_score` function):

```python


def assemble_score_from_system_predictions(
    system_token_lists: Sequence[List[str]],
    system_locations: Sequence[Dict],
    *,
    sample_id_prefix: str = "",
) -> AssembledScore:
    """Compose StaffRecognitionResult list from per-system token sequences.

    Splits each system's tokens at <staff_end> boundaries via
    `_split_staff_sequences_for_validation`, even-splits the system bbox
    vertically by N (number of emitted staves), and delegates to the
    existing assemble_score().

    Each `system_locations` entry must contain: system_index, bbox (4-tuple
    x1,y1,x2,y2), page_index, conf.
    """
    from src.data.convert_tokens import _split_staff_sequences_for_validation

    staves: List[StaffRecognitionResult] = []
    for sys_tokens, sys_loc in zip(system_token_lists, system_locations):
        per_staff_lists = _split_staff_sequences_for_validation(sys_tokens)
        if not per_staff_lists:
            continue
        n = len(per_staff_lists)
        sys_idx = int(sys_loc["system_index"])
        page_idx = int(sys_loc.get("page_index", 0))
        x1, y1, x2, y2 = sys_loc["bbox"]
        sys_h = float(y2) - float(y1)
        for i, staff_tokens in enumerate(per_staff_lists):
            y_top = float(y1) + i * sys_h / n
            y_bottom = float(y1) + (i + 1) * sys_h / n
            location = StaffLocation(
                page_index=page_idx,
                y_top=y_top,
                y_bottom=y_bottom,
                x_left=float(x1),
                x_right=float(x2),
            )
            sample_id = (
                f"{sample_id_prefix}page{page_idx:04d}"
                f"_sys{sys_idx:02d}_staff{i:02d}"
            )
            staves.append(
                StaffRecognitionResult(
                    sample_id=sample_id,
                    tokens=list(staff_tokens),
                    location=location,
                    system_index_hint=sys_idx,
                )
            )

    return assemble_score(staves)
```

If `Sequence`, `List`, or `Dict` aren't already imported at the top of the file, add the necessary imports from `typing`. Verify by reading the existing imports.

- [ ] **Step 4.4: Run the tests, confirm they pass**

```bash
python3 -m pytest tests/pipeline/test_assemble_from_system_predictions.py -x --no-header -q
```

Expected: `4 passed`.

- [ ] **Step 4.5: Commit**

```bash
git add src/pipeline/assemble_score.py tests/pipeline/test_assemble_from_system_predictions.py
git commit -m "feat(pipeline): add assemble_score_from_system_predictions"
```

---

## Task 5: `SystemInferencePipeline.__init__` + scaffolding

**Files:**
- Create: `src/inference/system_pipeline.py`
- Create: `tests/inference/test_system_pipeline.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/inference/test_system_pipeline.py`:

```python
"""Unit tests for src.inference.system_pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_pipeline_init_loads_yolo_and_stage_b_once():
    """__init__ should construct YoloStageASystems and call
    load_stage_b_for_inference exactly once each."""
    with patch("src.inference.system_pipeline.YoloStageASystems") as fake_yolo, \
         patch("src.inference.system_pipeline.load_stage_b_for_inference") as fake_loader:

        fake_loader.return_value = MagicMock(use_fp16=False)
        from src.inference.system_pipeline import SystemInferencePipeline

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt",
            stage_b_ckpt="stage_b.pt",
            device="cpu",
        )

    fake_yolo.assert_called_once_with("yolo.pt")
    fake_loader.assert_called_once()
    # Pipeline must NOT call _prepare_model_for_inference directly — the loader does.
    assert pipeline is not None
```

- [ ] **Step 5.2: Run the test, confirm it fails**

```bash
python3 -m pytest tests/inference/test_system_pipeline.py::test_pipeline_init_loads_yolo_and_stage_b_once -x --no-header -q
```

Expected: `ModuleNotFoundError: No module named 'src.inference.system_pipeline'`.

- [ ] **Step 5.3: Create the module with `__init__` only**

Create `src/inference/system_pipeline.py`:

```python
"""End-to-end per-system inference pipeline.

Library class that loads YOLO + Stage B once and exposes
`run_pdf` / `run_page` / `run_system_crop` / `export_musicxml`. See the
spec at docs/superpowers/specs/2026-05-10-radio-subproject4-design.md.

The class is designed so the eval driver can hold one instance for an
entire 50-piece run (Phase 1 inference only — scoring stays in subprocess).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from src.inference.checkpoint_load import StageBInferenceBundle, load_stage_b_for_inference
from src.models.yolo_stage_a_systems import YoloStageASystems


class SystemInferencePipeline:
    def __init__(
        self,
        yolo_weights,
        stage_b_ckpt,
        *,
        device: str = "cuda",
        beam_width: int = 1,
        max_decode_steps: int = 2048,
        page_dpi: int = 300,
        image_height: int = 250,
        image_max_width: int = 2500,
        length_penalty_alpha: float = 0.4,
        use_fp16: bool = False,
        quantize: bool = False,
    ):
        self._device = torch.device(device)
        self._stage_a = YoloStageASystems(yolo_weights)
        self._bundle: StageBInferenceBundle = load_stage_b_for_inference(
            stage_b_ckpt, self._device, use_fp16=use_fp16, quantize=quantize,
        )
        self._beam_width = beam_width
        self._max_decode_steps = max_decode_steps
        self._page_dpi = page_dpi
        self._image_height = image_height
        self._image_max_width = image_max_width
        self._length_penalty_alpha = length_penalty_alpha
```

- [ ] **Step 5.4: Run the test, confirm it passes**

```bash
python3 -m pytest tests/inference/test_system_pipeline.py::test_pipeline_init_loads_yolo_and_stage_b_once -x --no-header -q
```

Expected: `1 passed`.

- [ ] **Step 5.5: Commit**

```bash
git add src/inference/system_pipeline.py tests/inference/test_system_pipeline.py
git commit -m "feat(inference): scaffold SystemInferencePipeline.__init__"
```

---

## Task 6: `SystemInferencePipeline.run_system_crop`

**Files:**
- Modify: `src/inference/system_pipeline.py`
- Modify: `tests/inference/test_system_pipeline.py`

- [ ] **Step 6.1: Write the failing test**

Append to `tests/inference/test_system_pipeline.py`:

```python
def test_run_system_crop_decodes_via_bundle_and_returns_staves():
    """run_system_crop saves the crop, loads it via _load_stage_b_crop_tensor,
    runs encoder + decoder, and returns N StaffRecognitionResult objects."""
    from PIL import Image as _Image

    fake_bundle = MagicMock(use_fp16=False, vocab=MagicMock(tokens=[]))
    fake_token_seq = [
        "<bos>",
        "<staff_start>", "<staff_idx_0>",
        "<measure_start>", "note-C4-quarter", "<measure_end>", "<staff_end>",
        "<eos>",
    ]

    with patch("src.inference.system_pipeline.YoloStageASystems"), \
         patch("src.inference.system_pipeline.load_stage_b_for_inference",
               return_value=fake_bundle), \
         patch("src.inference.system_pipeline._load_stage_b_crop_tensor",
               return_value=MagicMock()) as fake_load_crop, \
         patch("src.inference.system_pipeline._encode_staff_image",
               return_value=MagicMock()) as fake_encode, \
         patch("src.inference.system_pipeline._decode_stage_b_tokens",
               return_value=fake_token_seq) as fake_decode:

        from src.inference.system_pipeline import SystemInferencePipeline

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt", stage_b_ckpt="stage_b.pt", device="cpu",
        )

        crop = _Image.new("RGB", (200, 100), color="white")
        sys_loc = {
            "system_index": 0,
            "bbox": (0.0, 0.0, 200.0, 100.0),
            "page_index": 0,
            "conf": 0.9,
        }
        staves = pipeline.run_system_crop(crop, system_index=0, system_location=sys_loc)

    assert len(staves) == 1
    assert staves[0].system_index_hint == 0
    fake_load_crop.assert_called_once()
    fake_encode.assert_called_once()
    fake_decode.assert_called_once()
```

- [ ] **Step 6.2: Run the test, confirm it fails**

```bash
python3 -m pytest tests/inference/test_system_pipeline.py::test_run_system_crop_decodes_via_bundle_and_returns_staves -x --no-header -q
```

Expected: `AttributeError: 'SystemInferencePipeline' object has no attribute 'run_system_crop'`.

- [ ] **Step 6.3: Implement `run_system_crop`**

In `src/inference/system_pipeline.py`, add imports and the method.

Add to the imports block at the top:

```python
import tempfile

from src.inference.decoder_runtime import (
    _decode_stage_b_tokens,
    _encode_staff_image,
    _load_stage_b_crop_tensor,
)
from src.pipeline.assemble_score import (
    AssembledScore,
    StaffRecognitionResult,
    assemble_score_from_system_predictions,
)
```

Add this method to the class:

```python
    def run_system_crop(
        self,
        crop: Image.Image,
        system_index: int,
        system_location: dict,
    ) -> List[StaffRecognitionResult]:
        """Decode a single system crop and return per-staff
        StaffRecognitionResult objects."""
        tokens = self._decode_one_crop(crop)
        score = assemble_score_from_system_predictions(
            [tokens], [system_location],
        )
        # Flatten staves from the single-system AssembledScore.
        result: List[StaffRecognitionResult] = []
        for system in score.systems:
            for staff in system.staves:
                result.append(
                    StaffRecognitionResult(
                        sample_id=staff.sample_id,
                        tokens=staff.tokens,
                        location=staff.location,
                        system_index_hint=staff.system_index_hint,
                    )
                )
        return result

    def _decode_one_crop(self, crop: Image.Image) -> List[str]:
        """Save crop to a temp file, run encoder + decoder, return token list."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            crop.save(tmp_path)
        try:
            pixel_values = _load_stage_b_crop_tensor(
                tmp_path,
                image_height=self._image_height,
                image_max_width=self._image_max_width,
                device=self._device,
            )
            if self._bundle.use_fp16:
                pixel_values = pixel_values.half()
            memory = _encode_staff_image(self._bundle.decode_model, pixel_values)
            return _decode_stage_b_tokens(
                model=self._bundle.model,
                pixel_values=pixel_values,
                vocabulary=self._bundle.vocab,
                beam_width=self._beam_width,
                max_decode_steps=self._max_decode_steps,
                length_penalty_alpha=self._length_penalty_alpha,
                _precomputed={
                    "decode_model": self._bundle.decode_model,
                    "memory": memory,
                    "token_to_idx": self._bundle.token_to_idx,
                    "use_fp16": self._bundle.use_fp16,
                },
            )
        finally:
            tmp_path.unlink(missing_ok=True)
```

**Note:** the spec doc returned `List[StaffRecognitionResult]` from `assemble_score_from_system_predictions` indirectly via the `AssembledScore`. Inspecting the actual `AssembledScore` shape: it has `.systems`, each system has `.staves`. The flatten loop above turns it back into a flat list, since `run_system_crop` is documented to return a flat list of staves. (This is the documented spec API; both shapes are useful — the assembler delegates to the existing `assemble_score()` which expects flat staves and produces the system-grouped structure.)

- [ ] **Step 6.4: Run the test, confirm it passes**

```bash
python3 -m pytest tests/inference/test_system_pipeline.py::test_run_system_crop_decodes_via_bundle_and_returns_staves -x --no-header -q
```

Expected: `1 passed`.

- [ ] **Step 6.5: Commit**

```bash
git add src/inference/system_pipeline.py tests/inference/test_system_pipeline.py
git commit -m "feat(inference): SystemInferencePipeline.run_system_crop"
```

---

## Task 7: `SystemInferencePipeline.run_page`

**Files:**
- Modify: `src/inference/system_pipeline.py`
- Modify: `tests/inference/test_system_pipeline.py`

- [ ] **Step 7.1: Write the failing test**

Append to `tests/inference/test_system_pipeline.py`:

```python
def test_run_page_iterates_systems_and_collects_staves():
    """run_page calls the Stage A wrapper, crops each system, decodes each."""
    from PIL import Image as _Image

    fake_bundle = MagicMock(use_fp16=False, vocab=MagicMock(tokens=[]))
    fake_yolo_instance = MagicMock()
    fake_yolo_instance.detect_systems.return_value = [
        {"system_index": 0, "bbox_extended": (0.0, 0.0, 100.0, 50.0), "conf": 0.9},
        {"system_index": 1, "bbox_extended": (0.0, 50.0, 100.0, 100.0), "conf": 0.85},
    ]
    fake_token_seq = [
        "<bos>",
        "<staff_start>", "<staff_idx_0>",
        "<measure_start>", "note-C4-quarter", "<measure_end>", "<staff_end>",
        "<eos>",
    ]

    with patch("src.inference.system_pipeline.YoloStageASystems",
               return_value=fake_yolo_instance), \
         patch("src.inference.system_pipeline.load_stage_b_for_inference",
               return_value=fake_bundle), \
         patch("src.inference.system_pipeline._load_stage_b_crop_tensor",
               return_value=MagicMock()), \
         patch("src.inference.system_pipeline._encode_staff_image",
               return_value=MagicMock()), \
         patch("src.inference.system_pipeline._decode_stage_b_tokens",
               return_value=fake_token_seq):

        from src.inference.system_pipeline import SystemInferencePipeline

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt", stage_b_ckpt="stage_b.pt", device="cpu",
        )
        page = _Image.new("RGB", (1000, 800), color="white")
        staves = pipeline.run_page(page, page_index=4)

    assert len(staves) == 2  # one staff per system in fake tokens
    assert {s.system_index_hint for s in staves} == {0, 1}
    assert all(s.location.page_index == 4 for s in staves)
```

- [ ] **Step 7.2: Run the test, confirm it fails**

```bash
python3 -m pytest tests/inference/test_system_pipeline.py::test_run_page_iterates_systems_and_collects_staves -x --no-header -q
```

Expected: `AttributeError: 'SystemInferencePipeline' object has no attribute 'run_page'`.

- [ ] **Step 7.3: Implement `run_page`**

Add this method to `SystemInferencePipeline`:

```python
    def run_page(
        self,
        page_image: Image.Image,
        page_index: int = 0,
    ) -> List[StaffRecognitionResult]:
        """Detect systems on the page, decode each, return flat staves list."""
        systems = self._stage_a.detect_systems(page_image)
        all_staves: List[StaffRecognitionResult] = []
        for sys in systems:
            x1, y1, x2, y2 = sys["bbox_extended"]
            crop = page_image.crop((int(x1), int(y1), int(x2), int(y2)))
            sys_loc = {
                "system_index": sys["system_index"],
                "bbox": sys["bbox_extended"],
                "page_index": page_index,
                "conf": sys["conf"],
            }
            staves = self.run_system_crop(crop, sys["system_index"], sys_loc)
            all_staves.extend(staves)
        return all_staves
```

- [ ] **Step 7.4: Run the test, confirm it passes**

```bash
python3 -m pytest tests/inference/test_system_pipeline.py::test_run_page_iterates_systems_and_collects_staves -x --no-header -q
```

Expected: `1 passed`.

- [ ] **Step 7.5: Commit**

```bash
git add src/inference/system_pipeline.py tests/inference/test_system_pipeline.py
git commit -m "feat(inference): SystemInferencePipeline.run_page"
```

---

## Task 8: `SystemInferencePipeline.run_pdf`

**Files:**
- Modify: `src/inference/system_pipeline.py`
- Create: `tests/inference/test_system_pipeline_pdf.py`

- [ ] **Step 8.1: Write the failing test**

Create `tests/inference/test_system_pipeline_pdf.py`:

```python
"""Integration test: SystemInferencePipeline.run_pdf + export_musicxml."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_run_pdf_produces_assembled_score_from_pdf():
    """run_pdf opens PDF via PyMuPDF, runs run_page on each page, assembles."""
    from PIL import Image as _Image

    fake_bundle = MagicMock(use_fp16=False, vocab=MagicMock(tokens=[]))
    fake_yolo_instance = MagicMock()
    fake_yolo_instance.detect_systems.return_value = [
        {"system_index": 0, "bbox_extended": (0.0, 0.0, 100.0, 50.0), "conf": 0.9},
    ]
    fake_token_seq = [
        "<bos>",
        "<staff_start>", "<staff_idx_0>",
        "<measure_start>", "note-C4-quarter", "<measure_end>", "<staff_end>",
        "<eos>",
    ]

    # Two-page fake PDF.
    fake_pixmap = MagicMock(width=1000, height=800)
    # samples must be a real bytes object the PIL frombytes call can consume.
    fake_pixmap.samples = bytes([255]) * (1000 * 800 * 3)
    fake_page = MagicMock()
    fake_page.get_pixmap.return_value = fake_pixmap
    fake_doc = MagicMock(__iter__=lambda self: iter([fake_page, fake_page]))
    fake_doc.__enter__ = MagicMock(return_value=fake_doc)
    fake_doc.__exit__ = MagicMock(return_value=False)

    with patch("src.inference.system_pipeline.YoloStageASystems",
               return_value=fake_yolo_instance), \
         patch("src.inference.system_pipeline.load_stage_b_for_inference",
               return_value=fake_bundle), \
         patch("src.inference.system_pipeline._load_stage_b_crop_tensor",
               return_value=MagicMock()), \
         patch("src.inference.system_pipeline._encode_staff_image",
               return_value=MagicMock()), \
         patch("src.inference.system_pipeline._decode_stage_b_tokens",
               return_value=fake_token_seq), \
         patch("src.inference.system_pipeline.fitz") as fake_fitz:

        fake_fitz.open.return_value = fake_doc

        from src.inference.system_pipeline import SystemInferencePipeline

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt", stage_b_ckpt="stage_b.pt", device="cpu",
        )
        score = pipeline.run_pdf("fake.pdf")

    # Two pages × one system × one staff each.
    total_staves = sum(len(s.staves) for s in score.systems)
    assert total_staves == 2
```

- [ ] **Step 8.2: Run the test, confirm it fails**

```bash
python3 -m pytest tests/inference/test_system_pipeline_pdf.py::test_run_pdf_produces_assembled_score_from_pdf -x --no-header -q
```

Expected: `AttributeError: 'SystemInferencePipeline' object has no attribute 'run_pdf'`.

- [ ] **Step 8.3: Implement `run_pdf`**

Add to imports in `src/inference/system_pipeline.py`:

```python
import fitz  # PyMuPDF
```

Add this method to `SystemInferencePipeline`:

```python
    def run_pdf(
        self,
        pdf_path,
        *,
        diagnostics=None,
    ) -> AssembledScore:
        """Render PDF pages, run Stage A + Stage B per page, assemble.

        `diagnostics` is currently a placeholder for future Stage-D skip
        recording during decode (assemble currently doesn't take a
        diagnostics arg). Pass-through is preserved so callers don't need
        to know which stages consume it.
        """
        all_token_lists = []
        all_locations = []
        with fitz.open(str(pdf_path)) as doc:
            for page_index, page in enumerate(doc):
                pix = page.get_pixmap(dpi=self._page_dpi)
                img = Image.frombytes(
                    "RGB", (pix.width, pix.height), pix.samples,
                )
                systems = self._stage_a.detect_systems(img)
                for sys in systems:
                    x1, y1, x2, y2 = sys["bbox_extended"]
                    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                    tokens = self._decode_one_crop(crop)
                    all_token_lists.append(tokens)
                    all_locations.append({
                        "system_index": sys["system_index"],
                        "bbox": sys["bbox_extended"],
                        "page_index": page_index,
                        "conf": sys["conf"],
                    })
        return assemble_score_from_system_predictions(
            all_token_lists, all_locations,
        )
```

- [ ] **Step 8.4: Run the test, confirm it passes**

```bash
python3 -m pytest tests/inference/test_system_pipeline_pdf.py::test_run_pdf_produces_assembled_score_from_pdf -x --no-header -q
```

Expected: `1 passed`.

- [ ] **Step 8.5: Commit**

```bash
git add src/inference/system_pipeline.py tests/inference/test_system_pipeline_pdf.py
git commit -m "feat(inference): SystemInferencePipeline.run_pdf via PyMuPDF"
```

---

## Task 9: `SystemInferencePipeline.export_musicxml` + diagnostics sidecar

**Files:**
- Modify: `src/inference/system_pipeline.py`
- Modify: `tests/inference/test_system_pipeline_pdf.py`

- [ ] **Step 9.1: Write the failing test**

Append to `tests/inference/test_system_pipeline_pdf.py`:

```python
def test_export_musicxml_writes_xml_and_diagnostics_sidecar(tmp_path):
    """export_musicxml writes both the .musicxml file and a
    .musicxml.diagnostics.json sidecar (matching the contract eval/_scoring_utils.py expects)."""
    fake_bundle = MagicMock(use_fp16=False, vocab=MagicMock(tokens=[]))

    fake_music_score = MagicMock()
    fake_music_score.write = MagicMock()

    with patch("src.inference.system_pipeline.YoloStageASystems"), \
         patch("src.inference.system_pipeline.load_stage_b_for_inference",
               return_value=fake_bundle), \
         patch("src.inference.system_pipeline.assembled_score_to_music21_with_diagnostics",
               return_value=fake_music_score) as fake_export:

        from src.inference.system_pipeline import SystemInferencePipeline
        from src.pipeline.export_musicxml import StageDExportDiagnostics

        pipeline = SystemInferencePipeline(
            yolo_weights="yolo.pt", stage_b_ckpt="stage_b.pt", device="cpu",
        )
        out_path = tmp_path / "out.musicxml"
        diags = StageDExportDiagnostics()
        diags.skipped_notes = 7  # populate to verify it gets serialized
        # Score is opaque to export_musicxml; use a sentinel.
        fake_score = MagicMock()

        # Make fake_music_score.write actually create the file (music21 .write would).
        def _write_stub(_format, path):
            from pathlib import Path as _P
            _P(path).write_text("<score-partwise/>")
        fake_music_score.write.side_effect = _write_stub

        pipeline.export_musicxml(fake_score, out_path, diagnostics=diags)

    fake_export.assert_called_once_with(fake_score, diags, strict=False)
    assert out_path.exists()
    sidecar = out_path.with_suffix(out_path.suffix + ".diagnostics.json")
    assert sidecar.exists()

    import json
    payload = json.loads(sidecar.read_text())
    assert payload["skipped_notes"] == 7
```

- [ ] **Step 9.2: Run the test, confirm it fails**

```bash
python3 -m pytest tests/inference/test_system_pipeline_pdf.py::test_export_musicxml_writes_xml_and_diagnostics_sidecar -x --no-header -q
```

Expected: `AttributeError: 'SystemInferencePipeline' object has no attribute 'export_musicxml'`.

- [ ] **Step 9.3: Implement `export_musicxml`**

Add to imports in `src/inference/system_pipeline.py`:

```python
import dataclasses
import json

from src.pipeline.export_musicxml import (
    StageDExportDiagnostics,
    assembled_score_to_music21_with_diagnostics,
)
```

Add this method to `SystemInferencePipeline`:

```python
    def export_musicxml(
        self,
        score: AssembledScore,
        out_path,
        *,
        diagnostics: Optional[StageDExportDiagnostics] = None,
    ) -> None:
        """Write the predicted MusicXML and the .diagnostics.json sidecar.

        Mirrors archive/per_staff/src/pdf_to_musicxml.py:395-465 (the
        production export pattern), but does not include the lenient
        re-export fallback. If music21.write fails, the exception
        propagates — the eval driver wraps individual pieces in try/except.
        """
        if diagnostics is None:
            diagnostics = StageDExportDiagnostics()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        music_score = assembled_score_to_music21_with_diagnostics(
            score, diagnostics, strict=False,
        )
        music_score.write("musicxml", str(out_path))

        diag_path = out_path.with_suffix(out_path.suffix + ".diagnostics.json")
        diag_dict = dataclasses.asdict(diagnostics)
        diag_path.write_text(json.dumps(diag_dict, indent=2), encoding="utf-8")
```

- [ ] **Step 9.4: Run all system_pipeline tests, confirm they pass**

```bash
python3 -m pytest tests/inference/test_system_pipeline.py tests/inference/test_system_pipeline_pdf.py -x --no-header -q
```

Expected: `4 passed`.

- [ ] **Step 9.5: Commit**

```bash
git add src/inference/system_pipeline.py tests/inference/test_system_pipeline_pdf.py
git commit -m "feat(inference): export_musicxml with diagnostics sidecar"
```

---

## Task 10: CLI `run_system_inference`

**Files:**
- Create: `src/cli/__init__.py`
- Create: `src/cli/run_system_inference.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/cli/test_run_system_inference.py`

- [ ] **Step 10.1: Create package markers**

```bash
touch src/cli/__init__.py tests/cli/__init__.py
```

- [ ] **Step 10.2: Write the failing test**

Create `tests/cli/test_run_system_inference.py`:

```python
"""argparse smoke for src.cli.run_system_inference."""
from __future__ import annotations


def test_argparser_does_not_require_vocab_flag():
    """The CLI must NOT require a --vocab flag — vocabulary is built
    in-code via build_default_vocabulary()."""
    from src.cli.run_system_inference import build_argparser

    parser = build_argparser()
    args = parser.parse_args([
        "--pdf", "x.pdf",
        "--out", "out.musicxml",
        "--yolo-weights", "yolo.pt",
        "--stage-b-ckpt", "stage_b.pt",
    ])
    assert str(args.pdf) == "x.pdf"
    assert str(args.out) == "out.musicxml"


def test_argparser_optional_flags_have_defaults():
    from src.cli.run_system_inference import build_argparser

    parser = build_argparser()
    args = parser.parse_args([
        "--pdf", "x.pdf", "--out", "out.musicxml",
        "--yolo-weights", "yolo.pt", "--stage-b-ckpt", "stage_b.pt",
    ])
    assert args.device == "cuda"
    assert args.beam_width == 1
    assert args.max_decode_steps == 2048
    assert args.page_dpi == 300
    assert args.length_penalty_alpha == 0.4
    assert args.fp16 is False
    assert args.quantize is False
```

- [ ] **Step 10.3: Run the test, confirm it fails**

```bash
python3 -m pytest tests/cli/test_run_system_inference.py -x --no-header -q
```

Expected: `ModuleNotFoundError: No module named 'src.cli.run_system_inference'`.

- [ ] **Step 10.4: Implement the CLI**

Create `src/cli/run_system_inference.py`:

```python
"""Thin CLI wrapping SystemInferencePipeline for one-off and smoke runs.

Usage:
    python -m src.cli.run_system_inference \\
        --pdf path/to/score.pdf \\
        --out out.musicxml \\
        --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt \\
        --stage-b-ckpt checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_system_inference",
        description="Per-system end-to-end inference: PDF -> MusicXML + diagnostics sidecar.",
    )
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output .musicxml path; .musicxml.diagnostics.json written alongside.")
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--stage-b-ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--max-decode-steps", type=int, default=2048)
    parser.add_argument("--page-dpi", type=int, default=300)
    parser.add_argument("--length-penalty-alpha", type=float, default=0.4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    from src.inference.system_pipeline import SystemInferencePipeline
    from src.pipeline.export_musicxml import StageDExportDiagnostics

    pipeline = SystemInferencePipeline(
        yolo_weights=args.yolo_weights,
        stage_b_ckpt=args.stage_b_ckpt,
        device=args.device,
        beam_width=args.beam_width,
        max_decode_steps=args.max_decode_steps,
        page_dpi=args.page_dpi,
        length_penalty_alpha=args.length_penalty_alpha,
        use_fp16=args.fp16,
        quantize=args.quantize,
    )
    diags = StageDExportDiagnostics()
    score = pipeline.run_pdf(args.pdf, diagnostics=diags)
    pipeline.export_musicxml(score, args.out, diagnostics=diags)
    print(f"wrote {args.out} + {args.out}.diagnostics.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.5: Run the test, confirm it passes**

```bash
python3 -m pytest tests/cli/test_run_system_inference.py -x --no-header -q
```

Expected: `2 passed`.

- [ ] **Step 10.6: Commit**

```bash
git add src/cli/__init__.py src/cli/run_system_inference.py tests/cli/__init__.py tests/cli/test_run_system_inference.py
git commit -m "feat(cli): add run_system_inference thin wrapper"
```

---

## REVIEW CHECKPOINT 1

After Task 10, the inference library + CLI are complete and unit-tested. Before moving to the eval driver, verify the full local test suite passes:

```bash
python3 -m pytest tests/inference tests/models/test_yolo_stage_a_systems.py tests/pipeline/test_assemble_from_system_predictions.py tests/cli/test_run_system_inference.py -x --no-header -q
```

Expected: all green. If any failures appear, stop and resolve before continuing.

---

## Task 11: Lift `eval/run_lieder_eval.py` and `eval/score_lieder_eval.py` from archive

**Files:**
- Move: `archive/per_staff/eval/run_lieder_eval.py` → `eval/run_lieder_eval.py`
- Move: `archive/per_staff/eval/score_lieder_eval.py` → `eval/score_lieder_eval.py`
- Move: `archive/per_staff/eval/tests/test_score_lieder_eval.py` → `eval/tests/test_score_lieder_eval.py`
- Move: `archive/per_staff/tests/test_run_lieder_eval.py` → `eval/tests/test_run_lieder_eval.py`

- [ ] **Step 11.1: Lift the files via git mv**

```bash
git mv archive/per_staff/eval/run_lieder_eval.py eval/run_lieder_eval.py
git mv archive/per_staff/eval/score_lieder_eval.py eval/score_lieder_eval.py
git mv archive/per_staff/eval/tests/test_score_lieder_eval.py eval/tests/test_score_lieder_eval.py
git mv archive/per_staff/tests/test_run_lieder_eval.py eval/tests/test_run_lieder_eval.py
```

- [ ] **Step 11.2: Verify status**

```bash
git status --short
```

Expected: 4 `R` (renamed) entries.

- [ ] **Step 11.3: Run the lifted tests as a baseline (some may fail — note which)**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py eval/tests/test_score_lieder_eval.py --no-header -q 2>&1 | tail -30
```

Note any failures. Most likely the `test_run_lieder_eval.py` is currently coupled to the archived inference path (`subprocess` invoking `python -m src.pdf_to_musicxml`); we'll fix it in Task 12. Some tests may pass as-is.

- [ ] **Step 11.4: Commit the bare lift (no behavior changes yet)**

```bash
git commit -m "chore(eval): lift run_lieder_eval + score_lieder_eval back from archive"
```

---

## Task 12: Rewire `run_inference()` to in-process `SystemInferencePipeline`

**Files:**
- Modify: `eval/run_lieder_eval.py`
- Modify: `eval/tests/test_run_lieder_eval.py`

- [ ] **Step 12.1: Read the existing `run_inference()` signature**

```bash
sed -n '139,210p' eval/run_lieder_eval.py
```

The current body invokes `subprocess.run([..., "src.pdf_to_musicxml", ...], ...)`. We replace its body with an in-process pipeline call.

- [ ] **Step 12.2: Write the failing test for the new in-process behavior**

Replace or append in `eval/tests/test_run_lieder_eval.py`:

```python
def test_run_inference_uses_in_process_pipeline_not_subprocess(monkeypatch, tmp_path):
    """run_inference should call pipeline.run_pdf + export_musicxml in-process,
    NEVER subprocess into src.pdf_to_musicxml (which is archived)."""
    from unittest.mock import MagicMock

    # Build a fake pipeline with the API surface run_inference uses.
    fake_pipeline = MagicMock()
    fake_score = MagicMock()
    fake_pipeline.run_pdf.return_value = fake_score

    # subprocess.run must NOT be called for inference.
    fake_subprocess = MagicMock()
    fake_subprocess.run = MagicMock(side_effect=AssertionError(
        "run_inference should not subprocess for inference"
    ))

    monkeypatch.setattr("eval.run_lieder_eval.subprocess", fake_subprocess)

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    out = tmp_path / "out.musicxml"
    workdir = tmp_path / "work"

    import eval.run_lieder_eval as rle
    rle.run_inference(
        pipeline=fake_pipeline,
        pdf=pdf,
        out=out,
        work_dir=workdir,
    )

    fake_pipeline.run_pdf.assert_called_once()
    fake_pipeline.export_musicxml.assert_called_once()
```

The test asserts a new `run_inference(pipeline=..., pdf=..., out=..., work_dir=...)` signature.

- [ ] **Step 12.3: Run the test, confirm it fails**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py::test_run_inference_uses_in_process_pipeline_not_subprocess -x --no-header -q
```

Expected: `TypeError` (old `run_inference()` had different parameters) or `AssertionError` (old behavior subprocesses).

- [ ] **Step 12.4: Replace `run_inference()` body**

In `eval/run_lieder_eval.py`, replace the entire `def run_inference(...)` function (lines ~139-208) with:

```python
def run_inference(
    *,
    pipeline,
    pdf: Path,
    out: Path,
    work_dir: Path,
) -> None:
    """Run inference on a single piece via the in-process SystemInferencePipeline.

    Writes the predicted .musicxml and the .musicxml.diagnostics.json sidecar
    next to `out`. Does NOT score; metric scoring happens in Phase 2 via
    eval.score_lieder_eval (subprocess-isolated to prevent music21/zss memory
    accumulation — see PR #26 motivation in this file's header).
    """
    from src.pipeline.export_musicxml import StageDExportDiagnostics

    work_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    diagnostics = StageDExportDiagnostics()
    score = pipeline.run_pdf(pdf, diagnostics=diagnostics)
    pipeline.export_musicxml(score, out, diagnostics=diagnostics)
```

Also remove imports / module-level helpers that only the old subprocess path used (e.g. building the `cmd` list). Do not delete the file's header comment — it documents the OOM history and is load-bearing per the spec.

- [ ] **Step 12.5: Run the test, confirm it passes**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py::test_run_inference_uses_in_process_pipeline_not_subprocess -x --no-header -q
```

Expected: `1 passed`.

- [ ] **Step 12.6: Update `main()` to construct a single pipeline**

In the same file, find `main()` and update the per-piece loop. The old `main()` constructed `cmd`-style args per piece. Replace the loop scaffold so that:
1. Before the loop, instantiate one `SystemInferencePipeline` from CLI args.
2. Inside the loop, call `run_inference(pipeline=pipeline, pdf=..., out=..., work_dir=...)`.
3. Wrap each call in try/except; on success status="ok", on exception status=f"failed:{type(e).__name__}".
4. Write status JSONL row immediately (this pattern already exists; preserve it).

Concrete sketch — adapt to the existing `main()` structure:

```python
def main() -> None:
    args = _build_argparser().parse_args()  # use the existing argparser
    eval_pieces = get_eval_pieces()
    if args.max_pieces is not None:
        eval_pieces = eval_pieces[: args.max_pieces]

    from src.inference.system_pipeline import SystemInferencePipeline
    pipeline = SystemInferencePipeline(
        yolo_weights=args.stage_a_yolo,
        stage_b_ckpt=args.checkpoint,
        device=args.stage_b_device,
        beam_width=args.beam_width,
        max_decode_steps=args.max_decode_steps,
    )

    for piece_idx, pdf_path in enumerate(eval_pieces):
        piece_id = pdf_path.stem
        if piece_id in already_completed:
            continue
        out_path = args.predictions_dir / f"{piece_id}.musicxml"
        work_dir = args.predictions_dir / "_work" / piece_id
        t_start = time.monotonic()
        try:
            run_inference(
                pipeline=pipeline,
                pdf=pdf_path,
                out=out_path,
                work_dir=work_dir,
            )
            record = {"piece_id": piece_id, "status": "ok",
                      "musicxml": str(out_path),
                      "wall_sec": time.monotonic() - t_start}
        except Exception as e:
            record = {"piece_id": piece_id, "status": f"failed:{type(e).__name__}",
                      "error": str(e)[:200],
                      "wall_sec": time.monotonic() - t_start}
            log.exception("piece %s inference failed", piece_id)
        _write_status_record(args.status_jsonl, record)

    if args.run_scoring:
        # Phase 2 (next task)
        ...
```

Use the actual existing helper names from the file — `_write_status_record` / `_format_eta` / etc. Read the file to see exact patterns; this is structural, not a verbatim replacement.

- [ ] **Step 12.7: Run the full eval test file**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py --no-header -q
```

Note any other failures from the lift; fix obvious ones (e.g. tests that asserted old subprocess args). For tests that exercise `_run_piece` or other archive-specific helpers, either delete (if the helper no longer exists) or rewrite. Keep tests that exercise: resume from status, ETA formatting, status JSONL writes, format_progress.

- [ ] **Step 12.8: Commit**

```bash
git add eval/run_lieder_eval.py eval/tests/test_run_lieder_eval.py
git commit -m "feat(eval): rewire run_inference to in-process SystemInferencePipeline"
```

---

## Task 13: Add `--run-scoring` and `--tedn` flags to `run_lieder_eval.py`

**Files:**
- Modify: `eval/run_lieder_eval.py`
- Modify: `eval/tests/test_run_lieder_eval.py`

- [ ] **Step 13.1: Write the failing test**

Append to `eval/tests/test_run_lieder_eval.py`:

```python
def test_run_scoring_spawns_subprocess_not_inline(monkeypatch, tmp_path):
    """When --run-scoring is set, eval driver must spawn score_lieder_eval
    as a subprocess (NOT call score_one_piece inline). This guards against
    accidentally re-introducing the in-process music21 OOM."""
    from unittest.mock import MagicMock
    import sys

    captured_calls = []

    def _fake_run(cmd, **kwargs):
        captured_calls.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr("eval.run_lieder_eval.subprocess.run", _fake_run)

    import eval.run_lieder_eval as rle
    rle.invoke_scoring_phase(
        predictions_dir=tmp_path / "preds",
        ground_truth_dir=tmp_path / "gt",
        out_csv=tmp_path / "scores.csv",
        with_tedn=True,
    )

    assert len(captured_calls) == 1
    cmd = captured_calls[0]
    assert sys.executable in cmd[0:1] or "python" in str(cmd[0]).lower()
    assert "eval.score_lieder_eval" in cmd
    assert "--tedn" in cmd
```

- [ ] **Step 13.2: Run the test, confirm it fails**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py::test_run_scoring_spawns_subprocess_not_inline -x --no-header -q
```

Expected: `AttributeError: module 'eval.run_lieder_eval' has no attribute 'invoke_scoring_phase'`.

- [ ] **Step 13.3: Add the helper**

Add to `eval/run_lieder_eval.py`:

```python
import sys


def invoke_scoring_phase(
    *,
    predictions_dir: Path,
    ground_truth_dir: Path,
    out_csv: Path,
    with_tedn: bool = False,
) -> int:
    """Spawn eval.score_lieder_eval as a subprocess for Phase 2 scoring.

    NEVER imports the scorer inline — keeps music21/zss isolated from the
    long-running inference process.
    """
    cmd = [
        sys.executable, "-m", "eval.score_lieder_eval",
        "--predictions-dir", str(predictions_dir),
        "--ground-truth-dir", str(ground_truth_dir),
        "--out-csv", str(out_csv),
    ]
    if with_tedn:
        cmd.append("--tedn")
    result = subprocess.run(cmd, check=False)
    return result.returncode
```

In the argparser, add `--run-scoring` (flag, default off) and `--tedn` (flag, default off). In `main()`, after the inference loop, if `args.run_scoring`, call `invoke_scoring_phase(...)` with `with_tedn=args.tedn`.

- [ ] **Step 13.4: Run the test, confirm it passes**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py::test_run_scoring_spawns_subprocess_not_inline -x --no-header -q
```

Expected: `1 passed`.

- [ ] **Step 13.5: Commit**

```bash
git add eval/run_lieder_eval.py eval/tests/test_run_lieder_eval.py
git commit -m "feat(eval): add --run-scoring + --tedn flags; invoke scorer subprocess"
```

---

## Task 14: Add `--tedn` flag to `score_lieder_eval.py`

**Files:**
- Modify: `eval/score_lieder_eval.py`
- Modify: `eval/tests/test_score_lieder_eval.py`

- [ ] **Step 14.1: Write the failing test**

Append to `eval/tests/test_score_lieder_eval.py`:

```python
def test_tedn_flag_default_off_skips_tedn_computation(monkeypatch, tmp_path):
    """When --tedn is NOT passed, score_one_piece must skip the TEDN computation
    entirely (no music21->kern conversion, no zss tree-edit-distance)."""
    from unittest.mock import MagicMock

    fake_compute_tedn = MagicMock()
    monkeypatch.setattr("eval.score_lieder_eval.compute_tedn", fake_compute_tedn,
                        raising=False)

    import eval.score_lieder_eval as sle

    parser = sle._build_argparser() if hasattr(sle, "_build_argparser") else sle.build_argparser()
    args = parser.parse_args([
        "--predictions-dir", str(tmp_path),
        "--ground-truth-dir", str(tmp_path),
        "--out-csv", str(tmp_path / "scores.csv"),
    ])
    assert args.tedn is False
```

- [ ] **Step 14.2: Run the test, confirm it fails**

```bash
python3 -m pytest eval/tests/test_score_lieder_eval.py::test_tedn_flag_default_off_skips_tedn_computation -x --no-header -q
```

Expected: `AttributeError: 'Namespace' object has no attribute 'tedn'`.

- [ ] **Step 14.3: Add the flag**

In `eval/score_lieder_eval.py`, find the argparser (likely a `_build_argparser` or inline in `main`). Add:

```python
parser.add_argument(
    "--tedn",
    action="store_true",
    help="Compute TEDN (Tree-Edit Distance on kern) — slow, ~300s/piece worst case. Default off.",
)
```

Then thread `args.tedn` through to wherever `score_one_piece` (or its child invocation) is called, gating any TEDN-specific work behind it. The exact hook point depends on how the file is structured — read it to find the call site.

If TEDN gating already exists via a different name (e.g. `--metrics` list), align the flag with the convention but keep the behavior: TEDN off by default.

- [ ] **Step 14.4: Run the test, confirm it passes**

```bash
python3 -m pytest eval/tests/test_score_lieder_eval.py::test_tedn_flag_default_off_skips_tedn_computation -x --no-header -q
```

Expected: `1 passed`.

- [ ] **Step 14.5: Run the full lifted test suite**

```bash
python3 -m pytest eval/tests/ --no-header -q
```

Note any pre-existing failures unrelated to the new flag; do not block on them but document in the commit.

- [ ] **Step 14.6: Commit**

```bash
git add eval/score_lieder_eval.py eval/tests/test_score_lieder_eval.py
git commit -m "feat(eval): add --tedn flag to score_lieder_eval (default off)"
```

---

## REVIEW CHECKPOINT 2

After Task 14, the eval driver is fully wired. Verify the local fast tests are green:

```bash
python3 -m pytest tests/inference tests/models/test_yolo_stage_a_systems.py tests/pipeline/test_assemble_from_system_predictions.py tests/cli eval/tests --no-header -q
```

Expected: all green for the new functionality. Pre-existing torch-dep / ImageMagick failures unrelated to this work are acceptable but should be noted.

If everything is green, push the branch and prepare for the GPU box smoke run.

```bash
git push -u origin feat/subproject4-system-inference
```

Then on the GPU box (`seder` / 10.10.1.29):

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && git fetch origin && git checkout feat/subproject4-system-inference && git pull --ff-only"
```

---

## Task 15: Smoke test on `lc6623145` (GPU box)

**Files:** none (run-only)

- [ ] **Step 15.1: SSH to GPU box and run the smoke**

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && python -m src.cli.run_system_inference --pdf data\\openscore_lieder\\eval_pdfs\\lc6623145.pdf --out smoke_lc6623145.musicxml --yolo-weights runs\\detect\\runs\\yolo26m_systems\\weights\\best.pt --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt --device cuda"
```

Expected output: `wrote smoke_lc6623145.musicxml + smoke_lc6623145.musicxml.diagnostics.json`.

If the command crashes, capture the traceback and stop. Common likely errors: missing torchvision dep on a fresh env, YOLO weights path typo, OOM during model load. Fix and re-run.

- [ ] **Step 15.2: Verify both artifacts exist**

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && dir smoke_lc6623145*"
```

Expected: both `smoke_lc6623145.musicxml` and `smoke_lc6623145.musicxml.diagnostics.json` present.

- [ ] **Step 15.3: Score the smoke against ground truth**

Confirm the ground-truth .mxl exists somewhere under `data/openscore_lieder/scores/`:

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && dir /s /b data\\openscore_lieder\\scores\\*lc6623145.mxl"
```

The scorer's `_discover_predictions(predictions_dir)` (in `eval/score_lieder_eval.py:130`) iterates files in a predictions dir and matches against ground truth by piece_id stem. So we put the smoke MusicXML + sidecar in a small predictions dir and run the scorer:

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && (if not exist smoke_predictions mkdir smoke_predictions) && copy /Y smoke_lc6623145.musicxml smoke_predictions\\lc6623145.musicxml && copy /Y smoke_lc6623145.musicxml.diagnostics.json smoke_predictions\\lc6623145.musicxml.diagnostics.json && python -m eval.score_lieder_eval --predictions-dir smoke_predictions --ground-truth-dir data\\openscore_lieder\\scores --out-csv smoke_scores.csv"
```

Expected: a `smoke_scores.csv` with one data row for `lc6623145` containing numeric `onset_f1` and `linearized_ser` columns.

If the scorer's CLI flags differ from the above, run `python -m eval.score_lieder_eval --help` and adjust to the actual flag names. The intent is fixed: a single-piece smoke score.

- [ ] **Step 15.4: Verify smoke onset_f1 is well above 0.067**

Read the resulting CSV. The piece's `onset_f1` should be well above the broken-pipeline baseline of `0.067`. Per the spec ship-gate criterion 2, near-baseline scores are a stop-and-investigate signal; do NOT proceed to the full 50-piece eval until smoke is clean.

If smoke `onset_f1` is reasonable, proceed.

- [ ] **Step 15.5: Document the smoke result**

Append a brief note to the branch (e.g. via a follow-up commit on a `docs/superpowers/handoffs/2026-MM-DD-subproject4-smoke.md`):

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && type smoke_lc6623145.musicxml | findstr /c:\"<note\" | find /c \"<note\""
# (counts notes as a sanity signal)
```

Record the smoke result locally:

```bash
git checkout -b docs/subproject4-smoke
# write a short handoff with date, smoke onset_f1, any notes
```

(This is optional but recommended before the long corpus run.)

---

## Task 16: Full 50-piece lieder eval (GPU box)

**Files:** none (run-only)

- [ ] **Step 16.1: Run Phase 1 (inference) on the GPU box**

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && python -m eval.run_lieder_eval --checkpoint checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt --stage-a-weights runs\\detect\\runs\\yolo26m_systems\\weights\\best.pt --output-dir eval\\results\\subproject4_run\\predictions --max-pieces 50 --stage-b-device cuda --beam-width 1"
```

Expected wall time: ~2-3 hours per the spec estimate. Watch for status JSONL appended per piece. The driver writes status to `eval/results/lieder_<name>_inference_status.jsonl` automatically (path derived from `--name`; there is no override flag). Read the file's existing CLI flags via `python -m eval.run_lieder_eval --help` for exact spelling.

- [ ] **Step 16.2: Verify Phase 1 completion**

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && type eval\\results\\lieder_<name>_inference_status.jsonl | find /c \"\\\"piece_id\\\"\""
```

Expected: 50 piece_id rows. Per the ship-gate, ≥ 40 must have `status=ok`.

- [ ] **Step 16.3: Run Phase 2 (scoring)**

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && python -m eval.score_lieder_eval --predictions-dir eval\\results\\subproject4_run\\predictions --ground-truth-dir data\\openscore_lieder\\scores --out-csv eval\\results\\subproject4_run\\scores.csv --memory-limit-gb 60"
```

(Skip `--tedn` for the architectural ship-gate run.) Expected wall time: ~30-50 minutes for cheap metrics on 50 pieces.

- [ ] **Step 16.4: Verify Phase 2 completion**

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && type eval\\results\\subproject4_run\\scores.csv | find /c /v \"\""
```

Expected: 51 rows (50 pieces + header). Per the ship-gate, ≥ 40 must have valid scores.

- [ ] **Step 16.5: Compute aggregate `mean(onset_f1)` across status=ok pieces**

```bash
ssh 10.10.1.29 "cd /d \"C:\\Users\\Jonathan Wesely\\Clarity-OMR-Train-RADIO\" && python -c \"import csv,statistics; rows=[r for r in csv.DictReader(open('eval/results/subproject4_run/scores.csv')) if r.get('status','ok')=='ok' and r.get('onset_f1','')]; print('N=',len(rows),'mean_onset_f1=',statistics.mean(float(r['onset_f1']) for r in rows))\""
```

Record this number. It is the architectural ship-gate result. The spec does not gate on the value — but the result is the architectural ship-gate output and informs the next phase.

- [ ] **Step 16.6: Pull the run artifacts back to local**

```bash
mkdir -p /home/ari/work/Clarity-OMR-Train-RADIO/eval/results/subproject4_run
scp 10.10.1.29:"\"C:/Users/Jonathan Wesely/Clarity-OMR-Train-RADIO/eval/results/subproject4_run/status.jsonl\"" /home/ari/work/Clarity-OMR-Train-RADIO/eval/results/subproject4_run/
scp 10.10.1.29:"\"C:/Users/Jonathan Wesely/Clarity-OMR-Train-RADIO/eval/results/subproject4_run/scores.csv\"" /home/ari/work/Clarity-OMR-Train-RADIO/eval/results/subproject4_run/
```

- [ ] **Step 16.7: Commit the run artifacts (small files only)**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git add eval/results/subproject4_run/status.jsonl eval/results/subproject4_run/scores.csv
git commit -m "data(eval): subproject4 50-piece lieder run results"
```

(`predictions/*.musicxml` are large; do not commit unless small. Use `.gitignore` if needed.)

- [ ] **Step 16.8: Write the handoff**

Create `docs/superpowers/handoffs/<YYYY-MM-DD>-subproject4-shipped.md` using this template (replace the bracketed values):

```markdown
# Subproject 4 — Shipped (<date>)

**Branch:** `feat/subproject4-system-inference` at HEAD `<sha>`
**PR:** #<num> (link)

## Smoke result (lc6623145)

- onset_f1: <value>
- linearized_ser: <value>
- Pass/fail: <above 0.067 baseline? yes/no>

## 50-piece corpus result

- Phase 1 status counts: <ok=NN, failed:X=N, ...>
- Phase 2 scored counts: <NN scored, MM timeouts, ...>
- mean(onset_f1) across status=ok: <value>
- mean(linearized_ser) across status=ok: <value>
- Per-piece distribution: see eval/results/subproject4_run/scores.csv

## Ship-gate verdict (per spec section "Ship-gate")

- [x] 1. Smoke produces parseable MusicXML + numeric F1
- [x] 2. Smoke F1 well above 0.067 baseline (<value> vs 0.067)
- [x] 3. Full Phase 1 + Phase 2 run completes — every piece has a row
- [x] 4. ≥80% Phase 1 ok AND ≥80% Phase 2 scored: <NN/50>, <MM/50>
- [x] 5. Aggregate mean(onset_f1) reported: <value>

**Verdict: SHIP** / DO NOT SHIP — investigate <cluster>

## Notes for next phase

- <observation 1: e.g., systems with N>2 staves cluster as failures, suggests staff_idx_3 token coverage>
- <observation 2: e.g., onset_f1 floor on cameraprimus-style scans suggests retraining with more font variety>

## Deferred follow-ups still open

(Per the spec's "Deferred follow-ups" section — none of these were done in Subproject 4.)

- TorchAO inference quantization experiment
- Refactor evaluate_stage_b_checkpoint.py onto load_stage_b_for_inference
- Subprocess-per-piece inference fallback if any future run shows OOMs
```

Then commit:

```bash
git add docs/superpowers/handoffs/<YYYY-MM-DD>-subproject4-shipped.md
git commit -m "docs(handoffs): subproject4 shipped"
```

---

## Final REVIEW CHECKPOINT — open the PR

If all four ship-gate criteria pass:

```bash
git push origin feat/subproject4-system-inference
gh pr create --title "feat: Subproject 4 — per-system inference pipeline + lieder eval" --body "..."
```

PR body should reference:
- Spec: `docs/superpowers/specs/2026-05-10-radio-subproject4-design.md`
- Plan: `docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`
- Handoff: `docs/superpowers/handoffs/2026-MM-DD-subproject4-shipped.md`
- Aggregate `mean(onset_f1)` from the 50-piece run

If any ship-gate criterion fails, do NOT open the PR. Investigate the failure cluster (per spec section "Ship-gate" criteria 4 escape hatch) and fix on-branch before re-running.

---

## Notes for the executing agent

- **Trust the spec, not your guesses.** When a step says "the existing X helper", read X and confirm the signature before writing code that calls it.
- **One commit per step where the step ends in commit.** Do not batch unrelated changes.
- **If a test fails after you implement what the plan says**, read the traceback carefully. The plan's mock setup may have drifted from the real signature; adjust the mock, not the production code, unless the production code is genuinely wrong.
- **Pre-existing failures in `tests/data/test_multi_dpi.py` (ImageMagick on Windows path) and `tests/data/test_encoder_cache.py` (torch missing locally) are environment issues, not regressions.** Skip them when verifying.
- **The eval files are LIFTED from `archive/per_staff/`. They reference `from eval.lieder_split import ...` — that import already works because `eval/lieder_split.py` is in the live tree.** Do not lift it from the archive.
- **Smoke before corpus.** Per the spec ship-gate criteria 2, never run the 50-piece corpus eval until the smoke piece scores well above the 0.067 broken-pipeline baseline.
