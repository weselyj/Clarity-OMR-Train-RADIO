# Repo De-personalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Clarity-OMR-Train-RADIO from a single-author scratchpad into a clean external-user-facing repo: address all six findings from the 2026-05-10 codex gap review, declare the project CUDA-only consistently, restructure the docs around external users, and move personal artifacts to `archive/`.

**Architecture:** One coherent push on a `feat/repo-depersonalization` branch. Code fixes first (TDD), then archive moves (mechanical), then doc creation/rewrite, then a de-personalization grep audit, then verification. The spec at `docs/superpowers/specs/2026-05-10-repo-depersonalization-design.md` is the source of truth for content and decisions.

**Tech Stack:** Python 3.13/3.14, PyTorch (cu132 nightly), pytest, ultralytics YOLO, music21, music21, NVIDIA RADIO. Repo lives at `/home/ari/work/Clarity-OMR-Train-RADIO/`.

---

## Pre-Flight: Branch Setup

Implementation runs on a feature branch (not `main`). Use the `superpowers:using-git-worktrees` skill at execution time if running in a worktree; otherwise create the branch directly:

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git checkout main && git pull --ff-only
git checkout -b feat/repo-depersonalization
```

Verify the spec is on `main` (commit `133ecb5`):

```bash
git log main --oneline -3 | grep -F "133ecb5"
```

Expected: `133ecb5 docs(spec): repo de-personalization and external-user readiness design`

---

## Task 1: CUDA Test-Gating Conftest

**Why:** Codex finding #1. With strict CUDA-only stance, tests under CUDA-required directories must skip cleanly when CUDA is unavailable, not raise import-time `RuntimeError: operator torchvision::nms does not exist`. After this task, the engineer can run `pytest -q` on the local Linux/CPU box and see clean skips instead of errors, which makes verifying the rest of this plan possible.

**Files:**
- Create: `tests/conftest.py`
- Modify: `pytest.ini` (add `cuda` marker)
- Test: `tests/test_conftest_cuda_gating.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_conftest_cuda_gating.py`:

```python
"""Tests for the CUDA-gating conftest helper."""

from pathlib import Path
from unittest import mock


def test_cuda_available_returns_false_when_torch_missing(monkeypatch):
    """If torch import fails, _cuda_available returns False (no exception)."""
    import sys
    # Hide torch from the import path so the helper's try/except triggers.
    monkeypatch.setitem(sys.modules, "torch", None)

    from tests.conftest import _cuda_available

    assert _cuda_available() is False


def test_cuda_available_reflects_torch_cuda_is_available():
    """When torch imports, _cuda_available mirrors torch.cuda.is_available()."""
    from tests.conftest import _cuda_available

    with mock.patch("torch.cuda.is_available", return_value=True):
        assert _cuda_available() is True
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert _cuda_available() is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
python3 -m pytest tests/test_conftest_cuda_gating.py -v
```

Expected: collection error or ImportError for `tests.conftest._cuda_available` (function not yet defined).

- [ ] **Step 3: Create the conftest**

Write `tests/conftest.py`:

```python
"""Pytest collection-time CUDA gating.

Skips tests under CUDA-required directories when torch is unavailable or
torch.cuda.is_available() returns False, with an informative reason.

Pure-Python tests (tests/data, tests/tokenizer, tests/decoding) continue
to run without CUDA.
"""

from pathlib import Path

import pytest


CUDA_REQUIRED_DIRS = {"inference", "pipeline", "cli", "models", "train"}
SKIP_REASON = (
    "CUDA required (this project requires a CUDA-capable GPU; "
    "see docs/HARDWARE.md)"
)


def _cuda_available() -> bool:
    """Return True if torch is importable and reports an available CUDA device.

    Returns False (without raising) if torch is missing, torch import fails,
    or no CUDA device is visible.
    """
    try:
        import torch
        if torch is None:
            return False
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        parts = Path(str(item.fspath)).parts
        if any(p in CUDA_REQUIRED_DIRS for p in parts):
            item.add_marker(skip)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python3 -m pytest tests/test_conftest_cuda_gating.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Add the `cuda` marker to pytest.ini**

Modify `pytest.ini` (currently 2 lines):

```ini
[pytest]
testpaths = tests src/tests eval/tests
markers =
    cuda: tests that require a CUDA-capable GPU
```

- [ ] **Step 6: Verify the gating works against real CUDA-required tests**

```bash
python3 -m pytest tests/inference/ tests/models/ -v 2>&1 | tail -20
```

Expected (on CPU box): every test reports `SKIPPED` with reason `"CUDA required (this project requires a CUDA-capable GPU; see docs/HARDWARE.md)"`. No `RuntimeError: operator torchvision::nms does not exist`.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_conftest_cuda_gating.py pytest.ini
git commit -m "test(conftest): add CUDA-gating for tests requiring a GPU

Resolves codex finding #1. Tests under tests/inference, tests/pipeline,
tests/cli, tests/models, tests/train auto-skip with an informative reason
when torch.cuda.is_available() is False or torch import fails. Pure-Python
tests under tests/data, tests/tokenizer, tests/decoding continue to run
on any environment.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Code Fix #2 — Stage A Weights Default + Validation

**Why:** Codex finding #2. The current default `~/Clarity-OMR/info/yolo.pt` points at a sibling repo's per-staff YOLO; running the lieder eval without `--stage-a-weights` could silently use the wrong model and reintroduce the per-staff/system mismatch Subproject 4 was designed to remove.

**Files:**
- Modify: `eval/run_lieder_eval.py:69` (default path constant)
- Modify: `eval/run_lieder_eval.py:321-323` (max-decode-steps — Task 3, but adjacent)
- Modify: `eval/run_lieder_eval.py:329-334` (--stage-a-weights argparse line)
- Modify: `eval/run_lieder_eval.py` `main()` (add validation after argparse)
- Test: `eval/tests/test_run_lieder_eval.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `eval/tests/test_run_lieder_eval.py`:

```python
def test_default_stage_a_weights_points_to_system_yolo():
    """Default --stage-a-weights is the in-repo system YOLO, not the sibling repo."""
    from eval.run_lieder_eval import _DEFAULT_STAGE_A_YOLO
    assert str(_DEFAULT_STAGE_A_YOLO) == "runs/detect/runs/yolo26m_systems/weights/best.pt"


def test_main_errors_when_stage_a_weights_missing(tmp_path, capsys, monkeypatch):
    """When --stage-a-weights points to a non-existent file, main exits with a clear error."""
    import sys
    from eval import run_lieder_eval

    missing = tmp_path / "does_not_exist.pt"
    fake_argv = [
        "run_lieder_eval",
        "--checkpoint", str(tmp_path / "ckpt.pt"),
        "--config", str(tmp_path / "cfg.yaml"),
        "--name", "test",
        "--stage-a-weights", str(missing),
    ]
    # The checkpoint and config don't need to exist for this test — argparse
    # validation of --stage-a-weights happens early.
    monkeypatch.setattr(sys, "argv", fake_argv)

    with pytest.raises(SystemExit):
        run_lieder_eval.main()

    captured = capsys.readouterr()
    assert "Stage A weights not found" in captured.err
    assert str(missing) in captured.err
    assert "docs/TRAINING.md" in captured.err
```

(`pytest` is imported at the top of the existing test file. If not, add `import pytest`.)

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py::test_default_stage_a_weights_points_to_system_yolo eval/tests/test_run_lieder_eval.py::test_main_errors_when_stage_a_weights_missing -v
```

Expected: 2 failed (current default is `~/Clarity-OMR/info/yolo.pt`; no validation logic exists).

- [ ] **Step 3: Update the default constant**

In `eval/run_lieder_eval.py` around line 68-69:

```python
# OLD:
# Override via --stage-a-weights.
_DEFAULT_STAGE_A_YOLO = Path.home() / "Clarity-OMR" / "info" / "yolo.pt"

# NEW:
# Override via --stage-a-weights. Path is repo-relative; resolved against
# the current working directory (typically the repo root).
_DEFAULT_STAGE_A_YOLO = Path("runs/detect/runs/yolo26m_systems/weights/best.pt")
```

- [ ] **Step 4: Update the argparse help text**

In `eval/run_lieder_eval.py` around lines 328-334:

```python
# OLD:
p.add_argument(
    "--stage-a-weights", type=Path, default=_DEFAULT_STAGE_A_YOLO,
    help=(
        "Path to Stage-A YOLO weights (default: ~/Clarity-OMR/info/yolo.pt). "
        "Download by running sibling Clarity-OMR's omr.py once on any PDF."
    ),
)

# NEW:
p.add_argument(
    "--stage-a-weights", type=Path, default=_DEFAULT_STAGE_A_YOLO,
    help=(
        f"Path to Stage A YOLO weights (default: {_DEFAULT_STAGE_A_YOLO}). "
        "Train Stage A first via scripts/train_yolo.py — see docs/TRAINING.md."
    ),
)
```

- [ ] **Step 5: Add startup validation in main()**

In `eval/run_lieder_eval.py` `main()` (starts at line 294, parser is `p = argparse.ArgumentParser(...)` at line 295), find `args = p.parse_args()`. Immediately after that line, **before any other path validation**, insert:

```python
if not args.stage_a_weights.is_file():
    p.error(
        f"Stage A weights not found at {args.stage_a_weights}. "
        f"Train Stage A first (see docs/TRAINING.md) or pass --stage-a-weights."
    )
```

`p.error()` writes to stderr, prefixes with the program name, and exits with status 2 — standard argparse error path. Place this validation **first** so the test (which doesn't create checkpoint/config files) doesn't trip over a different validation first.

If the test in Step 1 fails because checkpoint or config validation runs ahead of stage_a_weights, also create dummy files for those in the test:

```python
ckpt = tmp_path / "ckpt.pt"; ckpt.write_bytes(b"")
cfg = tmp_path / "cfg.yaml"; cfg.write_text("")
```

and pass those paths in `fake_argv`.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py -v
```

Expected: all tests in the file pass (the new ones plus all previously passing ones).

- [ ] **Step 7: Verify --help still works**

```bash
python3 -m eval.run_lieder_eval --help 2>&1 | grep -A1 stage-a-weights
```

Expected: help text mentions `runs/detect/runs/yolo26m_systems/weights/best.pt` and `docs/TRAINING.md`.

- [ ] **Step 8: Commit**

```bash
git add eval/run_lieder_eval.py eval/tests/test_run_lieder_eval.py
git commit -m "fix(eval): default --stage-a-weights to system YOLO; validate at startup

Resolves codex finding #2. The previous default pointed at the sibling
Clarity-OMR repo's per-staff YOLO, risking a silent fallback to wrong-format
weights and reintroducing the per-staff/system mismatch Subproject 4 was
built to remove. New default is the in-repo system YOLO checkpoint produced
by Stage A training. Validates the path exists at startup; on miss,
parser.error gives an actionable message pointing at docs/TRAINING.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Code Fix #3 — Decode Steps Default Alignment

**Why:** Codex finding #3. The corpus eval defaults `--max-decode-steps` to 256, but `SystemInferencePipeline` (the underlying engine) defaults to 2048. Help text still says "per staff" — stale terminology. The Subproject 4 shipped handoff already noted decoder truncation at 2048 for at least one smoke system; running corpus eval at 256 truncates almost everything.

**Files:**
- Modify: `eval/run_lieder_eval.py:320-323` (--max-decode-steps argparse line)
- Modify: `eval/run_lieder_eval.py:25-29` (module docstring mentioning the default)
- Test: `eval/tests/test_run_lieder_eval.py` (extend)

- [ ] **Step 1: Refactor the parser into a builder function**

The parser is currently built inline in `main()` (line 295). To make the default testable without invoking the full pipeline, extract a module-level builder. In `eval/run_lieder_eval.py`:

```python
def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the run_lieder_eval CLI parser. Module-level for testability."""
    p = argparse.ArgumentParser(
        # ... same description as the inline one ...
    )
    # ... move all p.add_argument(...) calls here, exactly as currently in main() ...
    return p


def main() -> None:
    p = build_argument_parser()
    args = p.parse_args()
    # ... rest of main() unchanged ...
```

This is a mechanical move: cut the parser construction (everything between `p = argparse.ArgumentParser(...)` and the line just before `args = p.parse_args()` inside `main()`), paste into the new function, and replace with a single call.

- [ ] **Step 2: Write the failing test**

Append to `eval/tests/test_run_lieder_eval.py`:

```python
def test_default_max_decode_steps_aligns_with_pipeline():
    """Default --max-decode-steps matches SystemInferencePipeline's 2048."""
    from eval.run_lieder_eval import build_argument_parser
    parser = build_argument_parser()
    namespace = parser.parse_args([
        "--checkpoint", "x.pt",
        "--config", "x.yaml",
        "--name", "test",
        "--stage-a-weights", "x.pt",
    ])
    assert namespace.max_decode_steps == 2048
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py::test_default_max_decode_steps_aligns_with_pipeline -v
```

Expected: fails (default is 256, or `build_argument_parser` doesn't exist yet).

- [ ] **Step 4: Update the default and help text**

In `eval/run_lieder_eval.py` around lines 320-323:

```python
# OLD:
p.add_argument(
    "--max-decode-steps", type=int, default=256,
    help="Stage-B max decode steps per staff (default 256; full quality is 512)",
)

# NEW:
p.add_argument(
    "--max-decode-steps", type=int, default=2048,
    help="Stage-B max decode steps per system crop (default 2048; aligns with "
         "SystemInferencePipeline default).",
)
```

- [ ] **Step 5: Update the module docstring**

In `eval/run_lieder_eval.py` around lines 25-30, find the docstring text:

```
Defaults to greedy decode (beam_width=1, max_decode_steps=256) — fast enough
for the MVP gate where the goal is just "any non-NaN F1 means inference works."
For real eval against a Stage 3 checkpoint, override with --beam-width 5
--max-decode-steps 512.
```

Replace with:

```
Defaults to greedy decode (beam_width=1, max_decode_steps=2048) — aligned
with SystemInferencePipeline's per-system-crop default. Override --beam-width
to 5 for higher-quality (slower) decoding.
```

- [ ] **Step 6: Run the test**

```bash
python3 -m pytest eval/tests/test_run_lieder_eval.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Verify --help reflects the new default**

```bash
python3 -m eval.run_lieder_eval --help 2>&1 | grep -B0 -A1 max-decode-steps
```

Expected: shows `default 2048` and `per system crop`.

- [ ] **Step 8: Commit**

```bash
git add eval/run_lieder_eval.py eval/tests/test_run_lieder_eval.py
git commit -m "fix(eval): align run_lieder_eval --max-decode-steps default to 2048

Resolves codex finding #3. The corpus eval default was 256 (per-staff
era), but SystemInferencePipeline's per-system-crop default is 2048.
Running corpus eval at 256 truncated decode for system crops with dense
notation. Help text updated from 'per staff' to 'per system crop'.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Code Fix #6 — Add `stage_d_skipped_systems` to Scoring CSV

**Why:** Codex finding #6. `StageDExportDiagnostics` writes `skipped_systems` to the diagnostics sidecar (`<pred>.diagnostics.json`), but `eval/_scoring_utils.py` neither parses it nor includes it in `CSV_HEADER`. This blocks the triage path from low `onset_f1` pieces back to dropped/truncated systems.

**Files:**
- Modify: `eval/_scoring_utils.py:46-53` (CSV_HEADER list)
- Modify: `eval/_scoring_utils.py:56-81` (`_read_stage_d_diag` function)
- Modify: `eval/_scoring_utils.py` module docstring (says "8-tuple" — must say "9-tuple")
- Modify: `eval/README.md` (if it documents the tuple shape)
- Test: `eval/tests/test_score_lieder_eval.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `eval/tests/test_score_lieder_eval.py`:

```python
def test_csv_header_includes_stage_d_skipped_systems():
    """CSV_HEADER includes the new stage_d_skipped_systems column."""
    from eval._scoring_utils import CSV_HEADER
    assert "stage_d_skipped_systems" in CSV_HEADER


def test_read_stage_d_diag_extracts_skipped_systems(tmp_path):
    """_read_stage_d_diag returns the skipped_systems count from the sidecar."""
    import json
    from eval._scoring_utils import _read_stage_d_diag

    pred = tmp_path / "piece.musicxml"
    pred.write_text("<score-partwise/>", encoding="utf-8")
    diag = pred.with_suffix(pred.suffix + ".diagnostics.json")
    diag.write_text(json.dumps({
        "skipped_notes": 1,
        "skipped_chords": 2,
        "missing_durations": 3,
        "malformed_spans": 4,
        "unknown_tokens": 5,
        "fallback_rests": 6,
        "raised_during_part_append": [],
        "skipped_systems": [
            {"system_index": 7, "reason": "decode_truncated"},
            {"system_index": 11, "reason": "malformed"},
        ],
    }), encoding="utf-8")

    result = _read_stage_d_diag(pred)

    assert len(result) == 9, f"expected 9-tuple, got {len(result)}: {result}"
    # The new field is the count, not the list — append at end of tuple.
    assert result[-1] == 2


def test_read_stage_d_diag_missing_skipped_systems_field(tmp_path):
    """If the sidecar lacks skipped_systems (older outputs), value defaults to 0."""
    import json
    from eval._scoring_utils import _read_stage_d_diag

    pred = tmp_path / "piece.musicxml"
    pred.write_text("<score-partwise/>", encoding="utf-8")
    diag = pred.with_suffix(pred.suffix + ".diagnostics.json")
    diag.write_text(json.dumps({
        "skipped_notes": 0, "skipped_chords": 0, "missing_durations": 0,
        "malformed_spans": 0, "unknown_tokens": 0, "fallback_rests": 0,
        "raised_during_part_append": [],
    }), encoding="utf-8")

    result = _read_stage_d_diag(pred)

    assert len(result) == 9
    assert result[-1] == 0


def test_read_stage_d_diag_returns_none_tuple_on_missing_sidecar(tmp_path):
    """When the sidecar is absent, return all-None 9-tuple (existing contract extended)."""
    from eval._scoring_utils import _read_stage_d_diag

    pred = tmp_path / "no_sidecar.musicxml"
    pred.write_text("<score-partwise/>", encoding="utf-8")

    result = _read_stage_d_diag(pred)

    assert result == (None, None, None, None, None, None, None, None, None)
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
python3 -m pytest eval/tests/test_score_lieder_eval.py::test_csv_header_includes_stage_d_skipped_systems eval/tests/test_score_lieder_eval.py::test_read_stage_d_diag_extracts_skipped_systems eval/tests/test_score_lieder_eval.py::test_read_stage_d_diag_missing_skipped_systems_field eval/tests/test_score_lieder_eval.py::test_read_stage_d_diag_returns_none_tuple_on_missing_sidecar -v
```

Expected: 4 failed (header missing column; tuple is 8-wide).

- [ ] **Step 3: Update CSV_HEADER**

In `eval/_scoring_utils.py:46-53`:

```python
# OLD:
CSV_HEADER = [
    "piece", "onset_f1", "tedn", "linearized_ser",
    "stage_d_skipped_notes", "stage_d_skipped_chords",
    "stage_d_missing_durations", "stage_d_malformed_spans",
    "stage_d_unknown_tokens", "stage_d_fallback_rests",
    "stage_d_raised_count", "stage_d_first_error",
    "score_failure_reason",
]

# NEW:
CSV_HEADER = [
    "piece", "onset_f1", "tedn", "linearized_ser",
    "stage_d_skipped_notes", "stage_d_skipped_chords",
    "stage_d_missing_durations", "stage_d_malformed_spans",
    "stage_d_unknown_tokens", "stage_d_fallback_rests",
    "stage_d_raised_count", "stage_d_first_error",
    "stage_d_skipped_systems",
    "score_failure_reason",
]
```

- [ ] **Step 4: Update `_read_stage_d_diag` to return a 9-tuple**

In `eval/_scoring_utils.py:56-81`:

```python
# OLD:
def _read_stage_d_diag(pred_path: Path) -> tuple:
    """Return the 8 Stage-D diagnostic CSV values for *pred_path*.

    Looks for <pred_path>.diagnostics.json alongside the MusicXML output.
    Returns a tuple of 8 values (all None if the sidecar is absent or unreadable).
    Logs a warning to stderr when parsing fails (but still returns all-None so
    the row pipeline is not interrupted).
    """
    diag_path = pred_path.with_suffix(pred_path.suffix + ".diagnostics.json")
    try:
        raw = json.loads(diag_path.read_text(encoding="utf-8"))
        raised = raw.get("raised_during_part_append", [])
        first_error = raised[0].get("error_message", "") if raised else ""
        return (
            raw.get("skipped_notes"),
            raw.get("skipped_chords"),
            raw.get("missing_durations"),
            raw.get("malformed_spans"),
            raw.get("unknown_tokens"),
            raw.get("fallback_rests"),
            len(raised),
            first_error,
        )
    except Exception as exc:
        print(f"[warn] Stage D sidecar parse failed for {pred_path}: {exc}", file=sys.stderr)
        return (None, None, None, None, None, None, None, None)

# NEW:
def _read_stage_d_diag(pred_path: Path) -> tuple:
    """Return the 9 Stage-D diagnostic CSV values for *pred_path*.

    Looks for <pred_path>.diagnostics.json alongside the MusicXML output.
    Returns a tuple of 9 values (all None if the sidecar is absent or unreadable).
    Logs a warning to stderr when parsing fails (but still returns all-None so
    the row pipeline is not interrupted).

    The 9 values, in CSV column order:
        skipped_notes, skipped_chords, missing_durations, malformed_spans,
        unknown_tokens, fallback_rests, raised_count, first_error, skipped_systems_count
    """
    diag_path = pred_path.with_suffix(pred_path.suffix + ".diagnostics.json")
    try:
        raw = json.loads(diag_path.read_text(encoding="utf-8"))
        raised = raw.get("raised_during_part_append", [])
        first_error = raised[0].get("error_message", "") if raised else ""
        skipped_systems_count = len(raw.get("skipped_systems", []))
        return (
            raw.get("skipped_notes"),
            raw.get("skipped_chords"),
            raw.get("missing_durations"),
            raw.get("malformed_spans"),
            raw.get("unknown_tokens"),
            raw.get("fallback_rests"),
            len(raised),
            first_error,
            skipped_systems_count,
        )
    except Exception as exc:
        print(f"[warn] Stage D sidecar parse failed for {pred_path}: {exc}", file=sys.stderr)
        return (None, None, None, None, None, None, None, None, None)
```

- [ ] **Step 5: Update module docstring**

Near the top of `eval/_scoring_utils.py` find any reference to "8-tuple" and change to "9-tuple". The module-docstring comment line `_read_stage_d_diag()   — reads <pred>.diagnostics.json sidecar, returns 8-tuple` becomes `9-tuple`.

- [ ] **Step 6: Update the call site in score_lieder_eval.py**

`eval/score_lieder_eval.py:763` does `stage_d_cols = _read_stage_d_diag(pred)` and likely splats the tuple into a CSV row. The CSV writer must place `skipped_systems_count` in the column matching the new `stage_d_skipped_systems` header position (between `stage_d_first_error` and `score_failure_reason`). Read the surrounding ~30 lines around line 763 to confirm the column-order assumption holds; if the tuple is unpacked positionally and the row appends `score_failure_reason` separately, the new value at tuple position 8 (zero-indexed) lands in the right column.

If the call site needs adjustment (e.g., tuple positions are unpacked into named variables before being assembled into the CSV row), update accordingly so the new column lands between `stage_d_first_error` and `score_failure_reason`.

- [ ] **Step 7: Update existing test that asserts 8-tuple shape**

Search `eval/tests/test_score_lieder_eval.py` for assertions like `len(result) == 8` or `assert result == (None,) * 8` and update to 9. Use grep:

```bash
grep -n "len(result) == 8\|result\[7\]\|(None, None, None, None, None, None, None, None)" eval/tests/test_score_lieder_eval.py
```

Update each match to the 9-tuple version.

- [ ] **Step 8: Update eval/README.md**

If `eval/README.md` documents the function as returning an 8-tuple, update to 9-tuple. Check:

```bash
grep -n "8-tuple\|8 values\|stage_d_first_error" eval/README.md
```

Replace any "8-tuple" / "8 values" with "9-tuple" / "9 values". After the `stage_d_first_error` mention in the field list, append `, skipped_systems_count`.

- [ ] **Step 9: Run all eval tests**

```bash
python3 -m pytest eval/tests/ -v
```

Expected: all tests pass (new ones plus existing ones).

- [ ] **Step 10: Commit**

```bash
git add eval/_scoring_utils.py eval/score_lieder_eval.py eval/tests/test_score_lieder_eval.py eval/README.md
git commit -m "fix(eval): add stage_d_skipped_systems to scoring CSV

Resolves codex finding #6. StageDExportDiagnostics already writes
skipped_systems to the per-piece diagnostics sidecar, but the scoring
CSV ignored it. Adds stage_d_skipped_systems column to CSV_HEADER and
extends _read_stage_d_diag from an 8-tuple to a 9-tuple. Missing
sidecar fields default to 0 (or None when the whole sidecar is absent),
preserving the existing contract for older outputs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Subproject 4 Plan — Stale Flag Snippets

**Why:** Codex finding #4. The Subproject 4 implementation plan references `--stage-a-yolo`, `--predictions-dir`, and `--status-jsonl` flags that don't exist on the actual driver. A future reviewer following the plan hits argparse errors.

**Files:**
- Modify: `docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`

- [ ] **Step 1: Locate stale flag references**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
grep -n -- "--stage-a-yolo\|--predictions-dir\|--status-jsonl" docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md
```

- [ ] **Step 2: Replace each flag**

For each line returned in step 1:

- `--stage-a-yolo <path>` → `--stage-a-weights <path>`
- `--predictions-dir <path>` → `--output-dir <path>`
- `--status-jsonl <path>` → drop entirely. The driver writes to `eval/results/lieder_<name>_inference_status.jsonl` automatically; document the output path inline if the surrounding context expected the override.

- [ ] **Step 3: Verify no stale flags remain**

```bash
grep -n -- "--stage-a-yolo\|--predictions-dir\|--status-jsonl" docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md && echo "STILL HAS HITS" || echo "CLEAN"
```

Expected: `CLEAN`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md
git commit -m "docs(plan): fix stale flag names in subproject4 implementation plan

Resolves codex finding #4. The plan referenced --stage-a-yolo,
--predictions-dir, and --status-jsonl, none of which exist on
eval.run_lieder_eval. Updated to --stage-a-weights and --output-dir;
removed --status-jsonl (no override flag exists, status path is auto-
generated as eval/results/lieder_<name>_inference_status.jsonl).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Move Personal PowerShell Scripts to `archive/scripts/`

**Why:** Spec Section 4. These are one-off launchers, perf benchmarks, and homelab-specific utilities tied to the author's setup. They survive in git history; archived in-tree for reference.

**Files:**
- Create: `archive/scripts/`
- Move: 32 files from `scripts/` to `archive/scripts/`

- [ ] **Step 1: Create the archive subdirectory**

```bash
mkdir -p archive/scripts
```

- [ ] **Step 2: Move the personal scripts**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git mv scripts/cu132_phase3_bench_inner.ps1 archive/scripts/
git mv scripts/cu132_phase3_bench_launch.ps1 archive/scripts/
git mv scripts/cu132_phase4_compile_inner.ps1 archive/scripts/
git mv scripts/cu132_phase4_compile_launch.ps1 archive/scripts/
git mv scripts/cu132_phase6_dataloader_inner.ps1 archive/scripts/
git mv scripts/cu132_phase6_dataloader_launch.ps1 archive/scripts/
git mv scripts/mvp_inner.ps1 archive/scripts/
git mv scripts/launch_mvp.ps1 archive/scripts/
git mv scripts/backup_data_to_nas.ps1 archive/scripts/
git mv scripts/restore_data_inner.ps1 archive/scripts/
git mv scripts/restore_data_launch.ps1 archive/scripts/
git mv scripts/lieder_eval_stage3_inner.ps1 archive/scripts/
git mv scripts/lieder_eval_stage3_launch.ps1 archive/scripts/
git mv scripts/lieder_full_inner.ps1 archive/scripts/
git mv scripts/index_full_inner.ps1 archive/scripts/
git mv scripts/index_full_launch.ps1 archive/scripts/
git mv scripts/multi_dpi_render_inner.ps1 archive/scripts/
git mv scripts/multi_dpi_render_launch.ps1 archive/scripts/
git mv scripts/synthetic_inner.ps1 archive/scripts/
git mv scripts/synthetic_launch.ps1 archive/scripts/
git mv scripts/synthetic_regen_inner.ps1 archive/scripts/
git mv scripts/synthetic_regen_launch.ps1 archive/scripts/
git mv scripts/full_radio_stage1_inner.ps1 archive/scripts/
git mv scripts/full_radio_stage1_launch.ps1 archive/scripts/
git mv scripts/full_radio_stage2_inner.ps1 archive/scripts/
git mv scripts/full_radio_stage2_launch.ps1 archive/scripts/
git mv scripts/full_radio_stage3_inner.ps1 archive/scripts/
git mv scripts/full_radio_stage3_launch.ps1 archive/scripts/
git mv scripts/train_yolo26m_phase1_launch.ps1 archive/scripts/
git mv scripts/train_yolo26m_phase2_launch.ps1 archive/scripts/
git mv scripts/train_yolo_inner.ps1 archive/scripts/
git mv scripts/build_full_manifest.cmd archive/scripts/
git mv scripts/debug_burleigh_synthetic.py archive/scripts/
```

- [ ] **Step 3: Read-verify the three READ-VERIFY scripts**

For `compare_step_logs.py`, `enumerate_radio_modules.py`, `breakdown_audit.py`, and `analyze_data.py`, open each and decide whether to keep or archive based on this rule: if the file references specific run names, hardcoded personal paths, or one-off audit context (not a generic CLI a stranger could invoke), it goes to `archive/scripts/`.

```bash
head -40 scripts/compare_step_logs.py
head -40 scripts/enumerate_radio_modules.py
head -40 scripts/breakdown_audit.py
head -40 analyze_data.py
```

For each that fails the canonical bar:

```bash
git mv <file> archive/scripts/
```

- [ ] **Step 4: Verify scripts/ now contains only canonical entrypoints**

```bash
ls scripts/
```

Expected: only `setup_venv_cu132.ps1`, `cu132_venv_sitecustomize.py`, and the canonical `*.py` data-prep / audit / training entrypoints (build_*, derive_*, audit_kern_*, audit_per_*, audit_token_*, train_yolo.py, retokenize_with_staff_markers.py, build_encoder_cache.py, check_encoder_resume.py, smoketest_bracket_detector.py, visualize_audiolabs_systems.py, convert_omr_layout.py, build_sparse_augment_manifest.py, rederive_*).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: archive personal PowerShell launchers and one-off scripts

Moves cu132_phase* perf benchmarks, mvp launchers, homelab backup/restore
scripts, lieder/synthetic/multi_dpi/index_full launchers, full_radio_stage{1,2,3}
launchers, train_yolo26m phase launchers, debug_burleigh_synthetic, and
build_full_manifest.cmd to archive/scripts/. Plus any of compare_step_logs,
enumerate_radio_modules, breakdown_audit, analyze_data that contained personal
context per read-verify.

Canonical scripts/ now contains only generic entrypoints (setup_venv_cu132.ps1,
cu132_venv_sitecustomize.py, build_*.py, derive_*.py, audit_*.py,
train_yolo.py, and other reproducible utilities).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Move Handoffs to `archive/handoffs/`

**Why:** Spec Section 4. All session-state handoffs go to archive; `docs/superpowers/handoffs/` becomes empty.

**Files:**
- Create: `archive/handoffs/`
- Move: 8 files from `docs/superpowers/handoffs/` to `archive/handoffs/`

- [ ] **Step 1: Create the archive directory and move all handoffs**

```bash
mkdir -p archive/handoffs
cd /home/ari/work/Clarity-OMR-Train-RADIO
git mv docs/superpowers/handoffs/2026-05-09-radio-stage3-phase1-complete-handoff.md archive/handoffs/
git mv docs/superpowers/handoffs/2026-05-09-radio-stage3-phase1-launch-handoff.md archive/handoffs/
git mv docs/superpowers/handoffs/2026-05-09-radio-stage3-phase2-launch-handoff.md archive/handoffs/
git mv docs/superpowers/handoffs/2026-05-10-per-system-cleanup-wrap.md archive/handoffs/
git mv docs/superpowers/handoffs/2026-05-10-radio-stage3-phase2-mid-handoff.md archive/handoffs/
git mv docs/superpowers/handoffs/2026-05-10-subproject4-plan-ready.md archive/handoffs/
git mv docs/superpowers/handoffs/2026-05-10-subproject4-shipped.md archive/handoffs/
git mv docs/superpowers/handoffs/2026-05-10-subproject4-tasks-1-11-shipped.md archive/handoffs/
```

- [ ] **Step 2: Verify the source directory is empty**

```bash
ls docs/superpowers/handoffs/
```

Expected: no output (empty directory). Git won't track an empty directory; that's fine — `docs/superpowers/handoffs/` will be auto-removed from the working tree on commit, and any future handoff would re-create it.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: archive all docs/superpowers/handoffs/ to archive/handoffs/

Per the de-personalization spec: handoffs are session-state captures
tied to specific work sessions. None survive at the canonical path.
If a release-note artifact is later wanted, an impersonal one is
written fresh.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Move Eval Results to `archive/results/`

**Why:** Spec Section 4. Per-run results data has no place in the canonical tree. Future runs will write to a re-created (gitignored) `eval/results/`.

**Files:**
- Create: `archive/results/`
- Move: all of `eval/results/` to `archive/results/`

- [ ] **Step 1: Create the archive directory and move results**

```bash
mkdir -p archive/results
cd /home/ari/work/Clarity-OMR-Train-RADIO
git mv eval/results/baseline_davit_lieder.csv archive/results/
git mv eval/results/lieder_subproject4_scores.csv archive/results/
git mv eval/results/lieder_subproject4_inference_status.jsonl archive/results/
git mv eval/results/lieder_stage3_v2_smoke.csv archive/results/
git mv eval/results/lieder_mvp.csv archive/results/
git mv eval/results/per_dataset_stage2_v2_singlestaff.json archive/results/
git mv eval/results/per_dataset_stage2_v2_systems.json archive/results/
git mv eval/results/per_dataset_stage3_v2_singlestaff.json archive/results/
git mv eval/results/per_dataset_stage3_v2_systems.json archive/results/
git mv eval/results/baseline_pre_rebuild archive/results/
```

- [ ] **Step 2: Verify eval/results/ is now empty**

```bash
ls eval/results/
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: archive eval/results/ to archive/results/

Per-run eval CSVs, JSONLs, per-dataset JSONs, and the baseline_pre_rebuild
directory all leave the canonical tree. Future eval outputs will be written
to a re-created (gitignored) eval/results/. Canonical published results
will live at the link in docs/RESULTS.md (HuggingFace release pending).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Move Training Step Logs and Personal Manifest

**Why:** Spec Section 4. `src/train/training_steps*.jsonl` are personal training-run telemetry that landed in the source tree; `synthetic_systems_eval_fresh.jsonl` is a one-off eval slice manifest.

**Files:**
- Create: `archive/manifests/`
- Move: 5 files from `src/train/`, 1 file from `src/data/manifests/`

- [ ] **Step 1: Create archive subdirectory and move files**

```bash
mkdir -p archive/manifests
cd /home/ari/work/Clarity-OMR-Train-RADIO
git mv src/train/training_steps.jsonl archive/results/
git mv src/train/training_steps_observation_fix.jsonl archive/results/
git mv src/train/training_steps_repair.jsonl archive/results/
git mv src/train/training_steps_repair_from71k.jsonl archive/results/
git mv src/train/training_steps_repair_from71k_focusonly.jsonl archive/results/
git mv src/data/manifests/synthetic_systems_eval_fresh.jsonl archive/manifests/
```

- [ ] **Step 2: Verify token_manifest_stage3.jsonl is still in place**

The spec keeps `src/data/manifests/token_manifest_stage3.jsonl` as a canonical format reference example.

```bash
ls src/data/manifests/
```

Expected: `token_manifest_stage3.jsonl` present (or whatever else is documented as canonical).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: archive personal training-step logs and one-off eval manifest

Moves src/train/training_steps*.jsonl (5 files, personal training-run
telemetry) to archive/results/ and src/data/manifests/synthetic_systems_eval_fresh.jsonl
to archive/manifests/. token_manifest_stage3.jsonl stays in place as a
canonical format-reference example.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: Move NOTES.md and Read-Verified Personal Items

**Why:** Spec Section 4. `NOTES.md` is top-level personal env notes. `logs/` may contain personal log captures — read-verify and archive if so.

**Files:**
- Create: `archive/notes/`
- Move: `NOTES.md` and any verified-personal items

- [ ] **Step 1: Create archive subdirectory and move NOTES.md**

```bash
mkdir -p archive/notes
cd /home/ari/work/Clarity-OMR-Train-RADIO
git mv NOTES.md archive/notes/
```

- [ ] **Step 2: Inspect logs/**

```bash
ls -la logs/
```

If empty, `git rm -r logs/` (it's a placeholder dir). If it contains personal log captures (training run logs, debug captures), move them:

```bash
git mv logs archive/notes/logs
```

If it contains only an `__init__.py` or similar canonical placeholder, leave it.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: archive top-level NOTES.md and personal logs/

NOTES.md was the author's environment notes (Python version, gh credential
helper, RTX 5090 verification) — none of which an external user needs.
Moves it to archive/notes/.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Update .gitignore

**Why:** Spec Section 4. Future eval outputs, runs/, checkpoints/, and training step logs are gitignored to prevent them from re-accumulating in the canonical tree.

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add gitignore rules**

Append to `.gitignore`:

```
# Eval outputs (canonical published results live at docs/RESULTS.md link)
eval/results/*
!eval/results/.gitkeep
!eval/results/README.md

# YOLO training output trees (Stage A weights live on the GPU box)
runs/

# RADIO checkpoints (live on the GPU box; do not commit)
checkpoints/

# Personal training-run telemetry
src/train/training_steps*.jsonl
```

- [ ] **Step 2: Verify the rules are correct**

```bash
git check-ignore -v eval/results/foo.csv runs/detect/runs/yolo26m_systems/weights/best.pt checkpoints/full_radio_stage3_v2/_best.pt src/train/training_steps_demo.jsonl
```

Expected: each path matches a rule.

```bash
git check-ignore -v eval/results/.gitkeep eval/results/README.md
```

Expected: NOT ignored (the `!` exceptions match).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore(gitignore): exclude eval/results/, runs/, checkpoints/, training_steps logs

Keeps the canonical tree free of per-run artifacts going forward.
.gitkeep and README.md inside eval/results/ are excepted so the
empty-directory placeholder + pointer file land in git.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: Recreate `eval/results/` with `.gitkeep` and Pointer README

**Why:** Spec Section 4. The directory must exist in the tree (the eval driver writes there). Empty placeholder + a README pointing to canonical results.

**Files:**
- Create: `eval/results/.gitkeep` (zero-byte)
- Create: `eval/results/README.md`

- [ ] **Step 1: Create the placeholder file**

```bash
touch eval/results/.gitkeep
```

- [ ] **Step 2: Create the pointer README**

Write `eval/results/README.md`:

```markdown
# eval/results/

This directory holds eval outputs from local runs. Files written here by
`eval.run_lieder_eval` are gitignored.

Canonical published eval results (and any released model artifacts) live at
the link in [`docs/RESULTS.md`](../docs/RESULTS.md). To regenerate locally,
see [`docs/EVALUATION.md`](../docs/EVALUATION.md).
```

- [ ] **Step 3: Commit**

```bash
git add eval/results/.gitkeep eval/results/README.md
git commit -m "chore(eval): recreate eval/results/ with .gitkeep and pointer README

The directory is required by eval.run_lieder_eval but its contents are
gitignored. A short README points users to docs/RESULTS.md for canonical
published results and docs/EVALUATION.md for regenerating locally.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 13: Create `docs/HARDWARE.md`

**Why:** Spec Section 1 and Goal 2. The CUDA-only stance needs a dedicated, prominent doc.

**Files:**
- Create: `docs/HARDWARE.md`

- [ ] **Step 1: Write the doc**

Create `docs/HARDWARE.md`:

```markdown
# Hardware Requirements

This project requires a CUDA-capable NVIDIA GPU for both training and
inference. CPU-only execution is not supported.

## Minimum

- **GPU:** CUDA-capable NVIDIA GPU
- **VRAM:** 24 GB for inference; 48+ GB recommended for Stage B training
- **Driver:** NVIDIA 596.21 or later (CUDA 13.x compatible)
- **CUDA toolkit:** Not required — PyTorch ships its own CUDA runtime via the
  cu132 nightly wheel.
- **System RAM:** 32 GB minimum, 64 GB recommended for the lieder corpus eval
  (memory profile dominated by music21 score graphs).

## Tested Configurations

| GPU | VRAM | OS | Python | PyTorch | Notes |
|---|---|---|---|---|---|
| RTX 5090 | 32 GB | Windows 11 | 3.13 | 2.13 dev cu132 | Production reference |

The training and inference paths are CUDA-mandatory:

- Stage A YOLO training (`scripts/train_yolo.py`) calls into ultralytics, which
  uses torchvision NMS ops that resolve only against a CUDA-built torch.
- Stage B RADIO encoder (`src/models/radio_stage_b.py`) loads C-RADIOv4-H at
  bf16 on a CUDA device; the encoder cache pipeline (`scripts/build_encoder_cache.py`)
  is similarly device-bound.
- The unit test suite gates CUDA-required tests via `tests/conftest.py`. On a
  CPU-only environment, those tests skip with a clear "CUDA required" reason
  rather than failing with a cryptic torchvision import error.

## CPU-Only Tests

A small subset of pure-Python tests run anywhere — token vocabulary,
decoding utilities, and data-layer helpers under `tests/data/`,
`tests/tokenizer/`, and `tests/decoding/`. Everything else (inference,
pipeline, models, training, eval) requires a CUDA box.
```

- [ ] **Step 2: Commit**

```bash
git add docs/HARDWARE.md
git commit -m "docs: add HARDWARE.md declaring CUDA-only requirement

Linked from the slim README hardware banner. Documents minimum GPU/VRAM,
tested configuration, and what subset of the test suite runs on a CPU box.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 14: Create `docs/INSTALL.md`

**Why:** Spec Section 1. The Installation section moves out of the README into a dedicated install doc covering Linux + Windows.

**Files:**
- Create: `docs/INSTALL.md`

- [ ] **Step 1: Write the doc**

Create `docs/INSTALL.md`:

```markdown
# Installation

This project requires a CUDA-capable GPU. See [HARDWARE.md](HARDWARE.md).

## 1. Clone

```bash
git clone https://github.com/weselyj/Clarity-OMR-Train-RADIO.git
cd Clarity-OMR-Train-RADIO
```

## 2. Create the cu132 environment

### Windows (PowerShell)

```powershell
.\scripts\setup_venv_cu132.ps1
```

This creates `venv-cu132/`, installs PyTorch nightly cu132 + cuDNN 9.21.01 +
project requirements, and drops a `sitecustomize.py` to resolve the cu132
DLL search path. Idempotent — re-run to refresh.

Activate:

```powershell
.\venv-cu132\Scripts\Activate.ps1
```

### Linux (bash)

```bash
./scripts/setup_venv_cu132.sh
```

This creates `venv-cu132/`, installs PyTorch nightly cu132 + cuDNN 13 +
project requirements. The Windows DLL workaround does not apply on Linux.

Activate:

```bash
source venv-cu132/bin/activate
```

### Manual install (any platform)

```bash
python -m venv venv-cu132
source venv-cu132/bin/activate  # or .\venv-cu132\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
pip install nvidia-cudnn-cu13
pip install -r requirements.txt
pip install pytest
```

## 3. Verify

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"
```

Expected: `CUDA OK: <your GPU name>`.

## 4. Run the test suite

```bash
pytest tests/data tests/tokenizer -q
```

Expected: pure-Python tests pass. CUDA-required tests under `tests/inference/`,
`tests/pipeline/`, `tests/cli/`, `tests/models/`, `tests/train/`, and
`eval/tests/` skip cleanly on a CPU-only environment with reason
`"CUDA required (this project requires a CUDA-capable GPU; see docs/HARDWARE.md)"`.

To run the full suite (requires CUDA):

```bash
pytest -q
```

## Rollback (cu128)

If cu132 nightly is unstable, the previous cu128 environment can be restored:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

This produces `venv/` (the rollback path), distinct from `venv-cu132/`
(production).
```

- [ ] **Step 2: Commit**

```bash
git add docs/INSTALL.md
git commit -m "docs: add INSTALL.md with Linux + Windows cu132 paths

Replaces the inline Installation section in README. Covers
scripts/setup_venv_cu132.{ps1,sh}, manual venv install, GPU smoke test,
and the cu128 rollback escape hatch.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 15: Create `docs/RESULTS.md`

**Why:** Spec Section 1 + Hugging Face follow-up hook. A stable canonical pointer for whoever picks this up later.

**Files:**
- Create: `docs/RESULTS.md`

- [ ] **Step 1: Write the doc**

Create `docs/RESULTS.md`:

```markdown
# Results

Per-run eval outputs are not committed to this repository (they are
gitignored under `eval/results/`). Canonical published results — model
weights, scoring artifacts, and the lieder eval CSV bundle — will be
released to Hugging Face once Stage 3 inference reaches the ship gate.

**Hugging Face:** _Release pending. Link will be published here when
artifacts are uploaded._

To regenerate results locally on a CUDA-capable GPU box, see
[EVALUATION.md](EVALUATION.md).

## Reference baselines

Historical baselines (DaViT pre-rebuild, MVP smokes, pre-Subproject 4 lieder
runs) are preserved in [`archive/results/`](../archive/results/) for
reproducibility audits but should not be used as targets for current work.
```

- [ ] **Step 2: Commit**

```bash
git add docs/RESULTS.md
git commit -m "docs: add RESULTS.md as a stable pointer for the HF release

Per-run results live outside the repo (gitignored eval/results/);
canonical published results will land on Hugging Face. RESULTS.md is
the link target both the README and EVALUATION.md point to.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 16: Create `docs/paths.md` (replaces `docs/locations.md`)

**Why:** Spec Section 1 + 7. `locations.md` is full of personal paths (`/home/ari/work/...`, `C:\Users\Jonathan Wesely\...`, GPU box IP). Replacement is repo-relative only.

**Files:**
- Create: `docs/paths.md`

- [ ] **Step 1: Write the doc**

Create `docs/paths.md`:

```markdown
# Repository-Relative Paths

Canonical paths for build artifacts, training outputs, and corpus data
that aren't committed to the repo. Paths are relative to the repo root
unless noted.

## Stage A (system YOLO) weights

| Path | Source |
|---|---|
| `runs/detect/runs/yolo26m_systems/weights/best.pt` | Produced by `scripts/train_yolo.py` (see [TRAINING.md](TRAINING.md)) |
| `runs/detect/runs/yolo26m_systems/weights/last.pt` | Sibling final-epoch weights |

This is the default `--stage-a-weights` for `eval.run_lieder_eval`. The
`runs/` tree is gitignored.

## Stage B (RADIO) checkpoints

| Path | Notes |
|---|---|
| `checkpoints/full_radio_stage3_v2/_best.pt` | Production Stage 3 v2 checkpoint (val_loss 0.148 at step 4000) |
| `checkpoints/full_radio_stage3_v2/_final.pt` | Last step of the run |
| `checkpoints/full_radio_stage3_v2/_step_<N>.pt` | Every 500 steps for ablations |

The `checkpoints/` tree is gitignored. Other Stage B variant directories
(`full_radio_stage1_v2`, `full_radio_stage2_systems_v2`, `baseline_davit`,
…) follow the same pattern.

## Lieder corpus (eval data)

| Path | Contents |
|---|---|
| `data/openscore_lieder/scores/` | `.mxl` ground-truth files, nested `<Composer>/<Opus>/<Song>/<id>.mxl` |
| `data/openscore_lieder/eval_pdfs/` | PDF render of the deterministic 10% eval split |

Source: [OpenScore Lieder](https://github.com/OpenScore/Lieder). The 50-piece
subset used for Subproject 4 evals is a deterministic sub-slice of `eval_pdfs/`
(the eval driver enforces the slice).

## Synthetic data

| Path | Notes |
|---|---|
| `data/synthetic_v2/` | Verovio-rendered grand-staff pages + per-system labels |

Built by `src/data/generate_synthetic.py`.

## Encoder cache (Stage 3 training)

| Path | Notes |
|---|---|
| `data/encoder_cache/` | Pre-encoded RADIO features keyed by sample hash |

Built by `scripts/build_encoder_cache.py`.

## Eval outputs (per-run, gitignored)

| Path | Producer |
|---|---|
| `eval/results/lieder_<name>/<piece_id>.musicxml` | `eval.run_lieder_eval` |
| `eval/results/lieder_<name>/<piece_id>.musicxml.diagnostics.json` | `eval.run_lieder_eval` |
| `eval/results/lieder_<name>_inference_status.jsonl` | `eval.run_lieder_eval` |
| `eval/results/lieder_<name>_scores.csv` | `eval.score_lieder_eval` (run via `--run-scoring`) |

See [`eval/results/README.md`](../eval/results/README.md) and
[`docs/RESULTS.md`](RESULTS.md).

## Token manifest reference

`src/data/manifests/token_manifest_stage3.jsonl` — first line is the canonical
example of expected per-system manifest format (`staves_in_system`,
`staff_indices` show multi-staff stacked layout).

## Decision-gate config

`eval/decision_gate.py` — gates 3 static system floors
(`synthetic_systems` ≥ 90, `grandstaff_systems` ≥ 95, `primus_systems` ≥ 80)
and 2 dynamic regression tripwires (`cameraprimus_systems`, `cameraprimus`,
`primus`).

`grandstaff` (single-staff split) is intentionally NOT gated — it's invalid
eval data per the product rule (single-staff inputs must come from naturally
single-staff sources).
```

- [ ] **Step 2: Commit**

```bash
git add docs/paths.md
git commit -m "docs: add paths.md (repo-relative canonical paths) replacing locations.md

locations.md mixed repo-relative paths with the author's local laptop
and GPU box machine paths. paths.md is repo-relative only — describes
artifacts produced by the build, where they land, and how to regenerate
them. Per-run results are documented as gitignored.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 17: Create `docs/ARCHITECTURE.md` (relocate dense content from README)

**Why:** Spec Section 2. The README currently contains ~250 lines of architecture detail. That's the right content but the wrong place for a slim landing.

**Files:**
- Create: `docs/ARCHITECTURE.md`
- Source: `README.md` (current 533-line version, before Task 21 slims it)

- [ ] **Step 1: Build ARCHITECTURE.md from README sections**

Open `README.md` and identify the following sections (still present pre-slim):

- "What this project does" → "Why system-level (vs per-staff)" through "Per-staff archival" (lines 14-54 in current README)
- "Stage A — System detection" through end of "Final training metrics" (lines 65-111)
- "Stage B — System-level recognition" through "DoRA adaptation" inclusive (lines 113-178)
- "Token vocabulary (~495 tokens)" through "Encoding example" (lines 180-224)
- "Grammar FSA (constrained decoding)" (lines 226-247)
- "Loss function" (lines 249-254)
- "Training stability" (lines 256-263)
- "Data augmentation" (lines 265-277)
- "Architecture rationale (selected)" (lines 511-519)
- "References" (lines 521-529)

Create `docs/ARCHITECTURE.md` with the following structure (paste each section in order, adjusting heading levels):

```markdown
# Architecture

## What this project does

[paste from README]

## Why system-level (vs per-staff)

[paste from README]

## Per-staff archival (2026-05-10)

[paste from README]

## Stage A — System detection

[paste from README, including the data table, label derivation pipeline,
and training command]

## Stage B — System-level recognition

[paste from README, including encoder/positional bridge/decoder/DoRA]

## Token vocabulary

[paste from README, including the table and encoding example]

## Grammar FSA

[paste from README]

## Loss function

[paste from README]

## Training stability

[paste from README]

## Data augmentation

[paste from README]

## Architecture rationale

[paste from README]

## References

[paste from README]
```

After paste, fix any heading levels: ARCHITECTURE.md is the document root, so the original README `##` headings become `##` here (same level), and `###` stays `###`.

Fix one personal reference: the README line 495 fragment `(in user repo)` does not appear in any of the relocated sections — it's in the Training section being moved to TRAINING.md. No fix needed in ARCHITECTURE.md.

- [ ] **Step 2: De-personalization grep on the new file**

```bash
grep -nE "/home/ari|C:\\\\Users\\\\Jonathan|seder|10\\.10\\.1\\.29|ari-homelab|in user repo" docs/ARCHITECTURE.md
```

Expected: no hits. If any appear, replace with repo-relative or generic equivalents.

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: add ARCHITECTURE.md with the dense architecture content from README

Relocates Stage A/B detail, token vocabulary, grammar FSA, loss function,
training stability, data augmentation, architecture rationale, and
references out of the README. README slim happens in a later task.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 18: Create `docs/EVALUATION.md`

**Why:** Spec Section 1. The Evaluation section moves out of README; new content covers running the lieder eval end-to-end.

**Files:**
- Create: `docs/EVALUATION.md`
- Source: existing README "Evaluation" section + `eval/run_lieder_eval.py --help` semantics

- [ ] **Step 1: Write the doc**

Create `docs/EVALUATION.md`:

```markdown
# Evaluation

This page covers running the lieder corpus evaluation end-to-end.
For metric definitions and decision-gate semantics, see the
"Evaluation" section of [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

- A CUDA-capable GPU (see [HARDWARE.md](HARDWARE.md))
- Activated cu132 venv (see [INSTALL.md](INSTALL.md))
- Stage A YOLO weights at `runs/detect/runs/yolo26m_systems/weights/best.pt`
  (or pass `--stage-a-weights`)
- A trained Stage B checkpoint (e.g. `checkpoints/full_radio_stage3_v2/_best.pt`)
- Lieder corpus data at `data/openscore_lieder/{scores,eval_pdfs}/`
  (see [paths.md](paths.md))

## Two-pass design

Lieder eval runs in two passes to keep music21/zss memory bounded:

1. **Inference pass.** `eval.run_lieder_eval` runs the RADIO/YOLO pipeline and
   writes `<piece_id>.musicxml` + `<piece_id>.musicxml.diagnostics.json` per
   piece. No metrics computed.
2. **Scoring pass.** `eval.score_lieder_eval` runs metric computation in a
   subprocess per piece (so music21 state is fully reclaimed between pieces)
   and writes `lieder_<name>_scores.csv`.

The `--run-scoring` flag on `run_lieder_eval` invokes the scorer subprocess
automatically after the inference pass completes.

## Running the corpus eval

```bash
python -m eval.run_lieder_eval \
    --checkpoint checkpoints/full_radio_stage3_v2/_best.pt \
    --config configs/train_stage2_radio_polyphonic.yaml \
    --name stage3v2_corpus \
    --max-pieces 50 \
    --run-scoring
```

Outputs (gitignored under `eval/results/`):

- `eval/results/lieder_stage3v2_corpus/<piece_id>.musicxml`
- `eval/results/lieder_stage3v2_corpus/<piece_id>.musicxml.diagnostics.json`
- `eval/results/lieder_stage3v2_corpus_inference_status.jsonl`
- `eval/results/lieder_stage3v2_corpus_scores.csv`

`--max-pieces 50` runs the deterministic 50-piece subset used for the
Subproject 4 corpus gate. Drop the flag for the full 145-piece eval split.

## Smoke run (single piece)

```bash
python -m src.cli.run_system_inference \
    --pdf data/openscore_lieder/eval_pdfs/lc6623145.pdf \
    --out smoke_lc6623145.musicxml \
    --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt \
    --stage-b-ckpt checkpoints/full_radio_stage3_v2/_best.pt
```

## Stratified scoring

`lieder_<name>_scores.csv` includes per-piece `onset_f1` and `linearized_ser`.
The `stage_d_skipped_systems` column flags pieces where one or more system
crops were dropped during MusicXML export — useful for triaging low scores.

The `tedn` column is omitted by default; pass `--tedn` to
`eval.score_lieder_eval` to enable (slow, memory-heavy).

## Metrics and decision gates

See ["Evaluation"](ARCHITECTURE.md#evaluation) in ARCHITECTURE.md for metric
definitions, the per-staff-count stratification, and the strong/mixed/flat
decision-gate boundaries.

## Canonical published results

See [RESULTS.md](RESULTS.md). Per-run CSVs from local boxes stay local
(gitignored).
```

- [ ] **Step 2: Commit**

```bash
git add docs/EVALUATION.md
git commit -m "docs: add EVALUATION.md covering the two-pass lieder eval

Documents prerequisites, the inference/scoring split, the
--run-scoring shortcut, smoke-run vs corpus-run commands, and the
stage_d_skipped_systems triage column. Metric definitions and
decision-gate semantics live in ARCHITECTURE.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 19: Create `docs/TRAINING.md` (merge TRAINING_COMMANDS*.md)

**Why:** Spec Section 1. Two existing training docs (`docs/TRAINING_COMMANDS.md` + `docs/TRAINING_COMMANDS_UBUNTU.md`) merge into one impersonal `docs/TRAINING.md`.

**Files:**
- Create: `docs/TRAINING.md`
- Delete: `docs/TRAINING_COMMANDS.md`
- Delete: `docs/TRAINING_COMMANDS_UBUNTU.md`

- [ ] **Step 1: Read both source files**

```bash
cat docs/TRAINING_COMMANDS.md
cat docs/TRAINING_COMMANDS_UBUNTU.md
```

Identify:
- Stage A YOLO training command
- Stage B (Stage 1 / Stage 2 / Stage 3) commands
- Any platform-specific notes worth preserving
- Any personal references (`/home/ari`, `seder`, `C:\Users\...`) that need removing

- [ ] **Step 2: Write the merged doc**

Create `docs/TRAINING.md`. Required structure:

```markdown
# Training

This page covers training commands for Stage A (YOLO system detection) and
Stage B (RADIO token recognition). Both stages require a CUDA-capable GPU
(see [HARDWARE.md](HARDWARE.md)).

## Stage A — System-level YOLO

[Lift the existing canonical training command from TRAINING_COMMANDS.md
and the README's Training section. Strip personal paths.]

```bash
python scripts/train_yolo.py \
  --model yolo26m.pt \
  --data data/processed/mixed_systems_v1/data.yaml \
  --epochs 100 --imgsz 1920 --batch 4 --workers 6 \
  --amp --nan-guard --noise --noise-warmup-steps 2000 \
  --project runs/detect/runs --name yolo26m_systems --patience 30
```

Wall time: ~10–12h on a single 5090. Gate: val mAP50 ≥ 0.95.

[Notable flags explanation lifted from the README — --noise-warmup-steps 2000,
--nan-guard, scan-noise pipeline.]

## Stage B — System-level RADIO

[Lift the three commands from TRAINING_COMMANDS.md / README:
Stage 1 v2, Stage 2 v2, Stage 3 (with reference to its design spec).]

```bash
# Stage 1 v2 — per-staff RADIO (completed; for re-runs only)
python src/train/train.py --config configs/train_stage1_radio.yaml

# Stage 2 v2 — polyphonic vocab-extension warmup
python src/train/train.py --config configs/train_stage2_radio_polyphonic.yaml

# Stage 3 — full system-level retrain with encoder-cache hybrid
# Requires Phase 0 encoder cache pre-built via scripts/build_encoder_cache.py.
# Stage 3 design: docs/superpowers/specs/2026-05-07-radio-stage3-design.md (in-repo).
```

[Add any platform-specific notes from TRAINING_COMMANDS_UBUNTU.md, generalized.
Drop any "/home/ari" or "seder" references.]

## Training data preparation

For data preparation steps (downloading PrIMuS / Camera-PrIMuS / GrandStaff /
OpenScore Lieder, rendering synthetic_v2, building the mixed Stage A dataset),
see [QUICKSTART.md](QUICKSTART.md).

## Encoder cache (Stage 3)

Stage 3 training uses a pre-built encoder cache that must be regenerated
when the encoder configuration changes:

```bash
python scripts/build_encoder_cache.py \
  --manifest src/data/manifests/token_manifest_stage3.jsonl \
  --output data/encoder_cache/
```

See `scripts/check_encoder_resume.py` for resume safety checks.
```

The actual content is constructed by reading both source files and the
README's "Training" section, removing personal references, and
consolidating into the structure above.

- [ ] **Step 3: Delete the old training docs**

```bash
git rm docs/TRAINING_COMMANDS.md
git rm docs/TRAINING_COMMANDS_UBUNTU.md
```

- [ ] **Step 4: De-personalization grep**

```bash
grep -nE "/home/ari|C:\\\\Users\\\\Jonathan|seder|10\\.10\\.1\\.29|in user repo" docs/TRAINING.md
```

Expected: no hits.

- [ ] **Step 5: Commit**

```bash
git add docs/TRAINING.md docs/TRAINING_COMMANDS.md docs/TRAINING_COMMANDS_UBUNTU.md
git commit -m "docs: merge TRAINING_COMMANDS{,_UBUNTU}.md into impersonal TRAINING.md

Single training doc covering Stage A YOLO + Stage B (Stage 1 v2,
Stage 2 v2, Stage 3) commands. Encoder cache build referenced.
References to specific machines and personal paths removed.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 20: Create `docs/QUICKSTART.md`

**Why:** Spec Section 1. The "what do I do first" doc for an external user.

**Files:**
- Create: `docs/QUICKSTART.md`

- [ ] **Step 1: Write the doc**

Create `docs/QUICKSTART.md`:

```markdown
# Quickstart

This guide takes you from a fresh clone to running a single-piece smoke
inference. Full training and corpus evaluation are covered in
[TRAINING.md](TRAINING.md) and [EVALUATION.md](EVALUATION.md).

## Prerequisites

- CUDA-capable GPU and driver. See [HARDWARE.md](HARDWARE.md).
- Python 3.13+.

## 1. Install

Follow [INSTALL.md](INSTALL.md) to clone the repo, create the cu132 venv,
and verify that `torch.cuda.is_available()` returns `True`.

## 2. Download Stage A YOLO weights

Stage A weights are not committed to the repo (the `runs/` tree is
gitignored). Either:

- **Train Stage A** — see [TRAINING.md](TRAINING.md). Requires the mixed
  systems dataset built by `scripts/build_mixed_v2_systems.py`.
- **Download released weights** — see [RESULTS.md](RESULTS.md) for the
  Hugging Face release link (pending).

Place the weights at `runs/detect/runs/yolo26m_systems/weights/best.pt`
(the path the eval driver expects by default).

## 3. Download a Stage B checkpoint

Same options as Stage A. Place the checkpoint at
`checkpoints/full_radio_stage3_v2/_best.pt` (or pass an explicit path).

## 4. Run inference on one PDF

```bash
python -m src.cli.run_system_inference \
    --pdf path/to/score.pdf \
    --out output.musicxml \
    --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt \
    --stage-b-ckpt checkpoints/full_radio_stage3_v2/_best.pt
```

A diagnostics sidecar is also written to `output.musicxml.diagnostics.json`.

## 5. Verify the test suite

Pure-Python tests (no GPU needed):

```bash
pytest tests/data tests/tokenizer -q
```

Full suite (CUDA required):

```bash
pytest -q
```

## Next steps

- For corpus-level eval: [EVALUATION.md](EVALUATION.md)
- For training: [TRAINING.md](TRAINING.md)
- For the deep architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- For repo-relative artifact paths: [paths.md](paths.md)
```

- [ ] **Step 2: Commit**

```bash
git add docs/QUICKSTART.md
git commit -m "docs: add QUICKSTART.md for the install -> smoke-inference path

Routes external users from a fresh clone through INSTALL, weights
acquisition, and a single-piece run_system_inference smoke test.
Full training and corpus eval defer to TRAINING.md / EVALUATION.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 21: Slim README

**Why:** Spec Section 2. README becomes a thin landing page that points into the docs/.

**Files:**
- Modify: `README.md` (full rewrite)

- [ ] **Step 1: Write the new slim README**

Replace the entire contents of `README.md` with:

```markdown
# Clarity-OMR-Train-RADIO

Training pipeline for an optical music recognition model that turns printed-score images into MusicXML.

This repository is a fork of [**clquwu/Clarity-OMR-Train**](https://github.com/clquwu/Clarity-OMR-Train) — the original training pipeline for [Clarity-OMR](https://github.com/clquwu/Clarity-OMR) (the inference repo). The fork extends the upstream project in two directions:

1. **Encoder swap (DaViT → C-RADIOv4-H).** Replaces the 86M-param ImageNet-pretrained DaViT encoder with NVIDIA's ~700M-param RADIO foundation encoder.
2. **System-level architectural rebuild.** Stage A detects full multi-staff systems; Stage B decodes whole systems in one pass with `<staff_idx_N>` marker tokens.

For inference only (PDF → MusicXML), see the upstream [Clarity-OMR](https://github.com/clquwu/Clarity-OMR) repo.

## ⚠️ Hardware Requirement

**This project requires a CUDA-capable GPU. CPU-only execution is not supported.**

See [docs/HARDWARE.md](docs/HARDWARE.md) for tested configurations.

## What it does

```
INPUT: Full-page score image (scan or PDF render)
  │
  ▼
STAGE A — System Detection (YOLO26m)
  │  Detect: full multi-staff systems on the page
  │  Output: ordered list of system bounding boxes, each tagged with its staff count
  │
  ▼
STAGE B — System-Level Recognition (C-RADIOv4-H encoder + RoPE decoder)
  │  Input: cropped system image (multi-staff, all voices in one pass)
  │  Output: token sequence with <staff_idx_N> markers identifying which staff each
  │          note/rest belongs to
  │
  ▼
STAGE C/D — Assembly + MusicXML Serialization
  │  Cross-staff attributes resolved (shared time/key signatures, barline alignment)
  │  Token stream → music21 stream objects → MusicXML export
  │
  ▼
OUTPUT: Valid MusicXML file
```

The earlier per-staff design lost cross-staff coordination signal (ties spanning systems, voice-piano alignment). System-level inputs preserve that context. Single-staff scores are still supported — they're treated as 1-staff systems at inference time.

The legacy per-staff inference pipeline is archived under [archive/per_staff/](archive/per_staff/).

For the full architecture (encoder choice, decoder, grammar FSA, loss, training stability, data augmentation, references), see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project status

| Subproject | Component | Status |
|---|---|---|
| 1 | Stage A system-level YOLO retrain | **Complete** — mAP50 0.995, recall 0.996, precision 0.998 on `mixed_systems_v1` |
| 2 | Kern converter rebuild + Stage 2 v2 trainer optimization | **Complete** — 96.9% kern→OMR token-fidelity audit; Stage 2 v2 val_loss 0.148 at step 4000 |
| 3 | Stage 3 RADIO retrain on system crops | **Training complete** — val_loss 0.148 at step 4000; 5.59h wall on a 5090 |
| 4 | Per-system inference + lieder corpus eval | **Shipped 2026-05-10** — see [docs/EVALUATION.md](docs/EVALUATION.md) |

## Documentation

| Doc | Purpose |
|---|---|
| [QUICKSTART.md](docs/QUICKSTART.md) | Clone → install → smoke inference |
| [HARDWARE.md](docs/HARDWARE.md) | GPU/VRAM/OS requirements |
| [INSTALL.md](docs/INSTALL.md) | cu132 venv setup (Linux + Windows) |
| [TRAINING.md](docs/TRAINING.md) | Stage A + Stage B training commands |
| [EVALUATION.md](docs/EVALUATION.md) | Lieder corpus eval (two-pass) |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full architecture, vocab, FSA, references |
| [paths.md](docs/paths.md) | Repo-relative paths for artifacts |
| [RESULTS.md](docs/RESULTS.md) | Canonical published results (HF release pending) |

## Repository layout

```
configs/                  # training YAML configs
src/
  data/                   # dataset preparation, label derivation
  models/                 # YOLO + RADIO encoder/decoder
  train/                  # training loops, model factory
  tokenizer/              # 495-token music vocabulary
  decoding/               # grammar FSA + beam search
  pipeline/               # cross-system assembly + MusicXML export
  inference/              # SystemInferencePipeline
  eval/                   # checkpoint eval, MusicXML comparison
  cli/                    # one-off CLI entrypoints
scripts/                  # canonical entrypoints (train_yolo, build_*, derive_*, audit_*)
tests/                    # pytest suite (CUDA-gated where required)
eval/                     # corpus eval drivers
docs/                     # documentation (see table above)
archive/                  # per-staff legacy code, archived scripts/results/handoffs
```

For per-file detail, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#repository-structure).

## License

GPL-3.0 — see [LICENSE](LICENSE).
```

- [ ] **Step 2: De-personalization grep on the new README**

```bash
grep -nE "/home/ari|C:\\\\Users\\\\Jonathan|seder|10\\.10\\.1\\.29|ari-homelab|in user repo|~/Clarity-OMR/" README.md
```

Expected: no hits.

- [ ] **Step 3: Verify all internal links resolve**

```bash
for link in $(grep -oE "\(docs/[^)]+\)" README.md | tr -d '()'); do
  test -e "$link" && echo "OK: $link" || echo "MISSING: $link"
done
test -e archive/per_staff && echo "OK: archive/per_staff" || echo "MISSING: archive/per_staff"
test -e LICENSE && echo "OK: LICENSE" || echo "MISSING: LICENSE"
```

Expected: every line says `OK:`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: slim README to a thin landing page

Hardware-required banner up top, 3-stage architecture diagram,
project status table, documentation index, slim repo layout, license.
Dense architecture/vocab/FSA/training/eval content moved to
docs/ARCHITECTURE.md, docs/TRAINING.md, docs/EVALUATION.md.
Installation moved to docs/INSTALL.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 22: Delete `docs/locations.md`

**Why:** Replaced by `docs/paths.md` (Task 16).

**Files:**
- Delete: `docs/locations.md`

- [ ] **Step 1: Verify nothing in the repo references `docs/locations.md`**

```bash
grep -rn "locations\.md\|locations.md" docs/ README.md src/ scripts/ eval/ tests/ requirements.txt 2>/dev/null
```

If any references remain (excluding `docs/locations.md` itself), update them to `docs/paths.md`.

- [ ] **Step 2: Delete the file**

```bash
git rm docs/locations.md
```

- [ ] **Step 3: Commit**

```bash
git commit -m "docs: remove docs/locations.md (replaced by docs/paths.md)

locations.md mixed repo-relative paths with personal Linux/Windows
machine paths. paths.md is the impersonal replacement.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 23: Update `requirements.txt` Comments for CUDA-Only

**Why:** Spec Goal 2. Current comments offer a CPU install path that doesn't actually work; stance is now strict CUDA-only.

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Replace the header comment block**

Open `requirements.txt`. Replace lines 1-13 (the comment block at top) with:

```
# Clarity-OMR-Train-RADIO dependencies
#
# This project requires a CUDA-capable GPU (see docs/HARDWARE.md).
# CPU-only torch is NOT a supported install path — ultralytics' YOLO
# kernels and the RADIO encoder both require CUDA-built torchvision.
#
# Install (production cu132 nightly path):
#   Windows: scripts/setup_venv_cu132.ps1
#   Linux:   scripts/setup_venv_cu132.sh
#   Manual:  see docs/INSTALL.md
```

Replace the next torch / torchvision section (currently a CPU-default install with a comment about CUDA override) with a note that torch/torchvision are installed by setup scripts:

```
# torch and torchvision are installed by scripts/setup_venv_cu132.{ps1,sh}
# from the PyTorch nightly cu132 index — DO NOT add them to this file
# (would override the cu132 selection with the latest stable wheel).
```

Remove the bare `torch` and `torchvision` lines.

- [ ] **Step 2: Verify Pillow comment still references the correct perf doc**

The existing Pillow~=12.1.0 comment references `docs/perf/2026-04-27-cu132-rollout.md`. Verify that file still exists:

```bash
test -e docs/perf/2026-04-27-cu132-rollout.md && echo "OK" || echo "MISSING"
```

If missing, generalize the comment to reference `docs/perf/` without naming a specific file.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore(requirements): rewrite header for strict CUDA-only stance

Drops the CPU install path (which never actually worked — ultralytics'
torchvision NMS op requires a CUDA-built torchvision). Points users at
scripts/setup_venv_cu132.{ps1,sh} or docs/INSTALL.md. torch/torchvision
removed from the requirements list because the setup scripts install
them from the cu132 nightly index.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 24: Create `scripts/setup_venv_cu132.sh`

**Why:** Spec Section 4. The Windows setup_venv script has no Linux equivalent; an external Linux user needs one to match.

**Files:**
- Create: `scripts/setup_venv_cu132.sh`

- [ ] **Step 1: Write the script**

Create `scripts/setup_venv_cu132.sh`:

```bash
#!/usr/bin/env bash
# Recreate the venv-cu132 nightly torch + CUDA 13.2 environment (Linux).
#
# Idempotent: if venv-cu132 already exists, the script appends to it.
# Tested on Ubuntu 24.04+ with NVIDIA driver 596+, CUDA Version 13.2.
#
# The Windows-only DLL workaround (scripts/cu132_venv_sitecustomize.py)
# does not apply here.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$repo/venv-cu132"
py="$venv/bin/python"

if [ ! -d "$venv" ]; then
    echo "[setup_venv_cu132] creating $venv"
    python3 -m venv "$venv"
fi

echo "[setup_venv_cu132] upgrading pip + setuptools + wheel"
"$py" -m pip install --upgrade pip setuptools wheel

echo "[setup_venv_cu132] installing nightly torch + torchvision (cu132)"
"$py" -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132

echo "[setup_venv_cu132] installing nvidia-cudnn-cu13"
"$py" -m pip install nvidia-cudnn-cu13

echo "[setup_venv_cu132] installing project requirements"
"$py" -m pip install -r "$repo/requirements.txt"

echo "[setup_venv_cu132] installing pytest"
"$py" -m pip install pytest

echo "[setup_venv_cu132] verifying CUDA"
"$py" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "[setup_venv_cu132] done. Activate with: source $venv/bin/activate"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/setup_venv_cu132.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_venv_cu132.sh
git commit -m "scripts: add Linux setup_venv_cu132.sh

Mirrors the Windows PowerShell setup script: creates venv-cu132,
installs torch nightly cu132 + torchvision + cudnn-cu13 + project
requirements + pytest, then runs a CUDA verification smoke test.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 25: De-personalization Grep Audit + Remaining Fixes

**Why:** Spec Section 7 + Verification criterion #1. The earlier tasks targeted known offenders; this task sweeps the canonical surface for stragglers.

**Files:**
- Modify: any file in canonical paths surfaced by the grep

- [ ] **Step 1: Run the audit grep**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
grep -rnE "/home/ari|C:\\\\Users\\\\Jonathan|Jonathan Wesely|seder|10\\.10\\.1\\.29|ari-homelab|in user repo|~/Clarity-OMR/" \
  README.md docs/ scripts/ src/ requirements.txt eval/ tests/ \
  --exclude-dir=archive 2>/dev/null
```

- [ ] **Step 2: Fix each hit**

For every hit returned in step 1:

- If it's in code/docs that should be canonical: replace with the impersonal equivalent.
  - `/home/ari/work/Clarity-OMR-Train-RADIO/<path>` → `<path>` (repo-relative) or `the repo root`
  - `C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\<path>` → `<path>` (repo-relative)
  - `seder` / `10.10.1.29` → "the GPU box" or remove entirely
  - `ari-homelab` → remove
  - `in user repo` → resolve to in-repo path or remove
  - `~/Clarity-OMR/info/yolo.pt` (sibling repo) → handled in Task 2; if any reference remains in a doc, replace with `runs/detect/runs/yolo26m_systems/weights/best.pt`
- If the file is itself archive material that wasn't moved: re-evaluate; should it be in `archive/`?

- [ ] **Step 3: Re-run the grep until clean**

```bash
grep -rnE "/home/ari|C:\\\\Users\\\\Jonathan|Jonathan Wesely|seder|10\\.10\\.1\\.29|ari-homelab|in user repo|~/Clarity-OMR/" \
  README.md docs/ scripts/ src/ requirements.txt eval/ tests/ \
  --exclude-dir=archive 2>/dev/null \
  && echo "STILL HAS HITS" || echo "CLEAN"
```

Expected: `CLEAN`.

- [ ] **Step 4: Voice scan on retained docs**

Skim `README.md`, `docs/QUICKSTART.md`, `docs/HARDWARE.md`, `docs/INSTALL.md`,
`docs/TRAINING.md`, `docs/EVALUATION.md`, `docs/ARCHITECTURE.md`,
`docs/RESULTS.md`, `docs/paths.md` for first-person voice ("I", "my", "we
got burned", "my GPU"). Rewrite to impersonal voice ("this project", "the
trainer", "users").

```bash
grep -rnE "\\bI \\b|\\bI'm\\b|\\bmy\\b|\\bwe got\\b" README.md docs/QUICKSTART.md docs/HARDWARE.md docs/INSTALL.md docs/TRAINING.md docs/EVALUATION.md docs/ARCHITECTURE.md docs/RESULTS.md docs/paths.md 2>/dev/null
```

Review each hit; rewrite where the voice is genuinely first-person (avoid false positives like "I/O" or "via").

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: de-personalization grep sweep across canonical paths

Final pass to replace hardcoded /home/ari, C:\\Users\\Jonathan, seder,
10.10.1.29, ari-homelab, ~/Clarity-OMR/, and 'in user repo' references
in canonical files. archive/ retains its content unchanged. First-person
voice in retained docs rewritten to impersonal.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

If step 5 finds nothing to commit (no diffs from steps 2-4 because all the work was caught upstream), skip the commit and move on.

---

## Task 26: Create External Tree Stub README

**Why:** Spec Section 5. The external `/home/ari/docs/superpowers/` tree gets a one-line pointer to canonical so future sessions don't get confused.

**Files:**
- Create: `/home/ari/docs/superpowers/README.md` (outside the repo)

- [ ] **Step 1: Verify the external tree was sync'd during brainstorming**

```bash
ls /home/ari/docs/superpowers/specs/ | grep subproject4
ls /home/ari/docs/superpowers/handoffs/ | grep subproject4
```

Expected: `2026-05-10-radio-subproject4-design.md`, `2026-05-10-radio-subproject4-overview.md` and three subproject4 handoffs are present.

- [ ] **Step 2: Write the stub README**

```bash
cat > /home/ari/docs/superpowers/README.md <<'EOF'
# /home/ari/docs/superpowers/

The canonical project specs, plans, audits, and handoffs for
Clarity-OMR-Train-RADIO live in the repo at
[`docs/superpowers/`](https://github.com/weselyj/Clarity-OMR-Train-RADIO/tree/main/docs/superpowers).

This local tree is a personal session-notes archive and may contain stale
material; treat the in-repo tree as authoritative.
EOF
```

- [ ] **Step 3: Verify**

```bash
cat /home/ari/docs/superpowers/README.md
```

Expected: the contents above.

This task does not touch the repo so there is no commit. The stub README lives only on the local machine.

---

## Task 27: Final Verification

**Why:** Spec Verification (Success Criteria) §1-10. Confirms the implementation is complete before opening a PR.

**Files:**
- None modified — verification only.

- [ ] **Step 1: Verification 1 — De-personalization grep is clean on canonical paths**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
grep -rnE "/home/ari|C:\\\\Users\\\\Jonathan|Jonathan Wesely|seder|10\\.10\\.1\\.29|ari-homelab" \
  README.md docs/ scripts/ src/ requirements.txt eval/ tests/ \
  --exclude-dir=archive 2>/dev/null \
  && echo "FAIL — hits remain" || echo "PASS"
```

Expected: `PASS`.

- [ ] **Step 2: Verification 2 — CUDA-only declaration consistency**

```bash
grep -l "CUDA" README.md docs/HARDWARE.md docs/INSTALL.md tests/conftest.py requirements.txt
```

Expected: all five files listed.

- [ ] **Step 3: Verification 3 — Pure-Python tests run on CPU**

```bash
python3 -m pytest tests/data tests/tokenizer tests/decoding -q 2>&1 | tail -10
```

Expected: pure-Python tests pass (or report only pre-existing failures unrelated to this effort).

- [ ] **Step 4: Verification 4 — CUDA-required tests skip cleanly on CPU**

```bash
python3 -m pytest tests/inference tests/models tests/cli tests/pipeline tests/train -q 2>&1 | tail -20
```

Expected: tests reported as `s` (skipped), not `E` (error). Skip reason matches `"CUDA required (this project requires a CUDA-capable GPU; see docs/HARDWARE.md)"`.

- [ ] **Step 5: Verification 5 — `--help` works (run on a CUDA box)**

If a CUDA box is available:

```bash
python3 -m src.cli.run_system_inference --help 2>&1 | head -5
python3 -m eval.run_lieder_eval --help 2>&1 | head -5
```

Expected: both print without traceback. The eval help text shows `runs/detect/runs/yolo26m_systems/weights/best.pt` as the default `--stage-a-weights` and `2048` as the default `--max-decode-steps`.

If no CUDA box is available, skip this step and flag for manual verification before merge.

- [ ] **Step 6: Verification 6 — README link integrity**

```bash
for link in $(grep -oE "\(docs/[^)]+\)" README.md | tr -d '()'); do
  test -e "$link" && echo "OK: $link" || echo "MISSING: $link"
done
test -e archive/per_staff && echo "OK: archive/per_staff" || echo "MISSING: archive/per_staff"
test -e LICENSE && echo "OK: LICENSE" || echo "MISSING: LICENSE"
```

Expected: every line `OK:`.

- [ ] **Step 7: Verification 7 — Codex code findings verifiable in source**

```bash
grep -n "_DEFAULT_STAGE_A_YOLO" eval/run_lieder_eval.py
# Expected: line shows runs/detect/runs/yolo26m_systems/weights/best.pt

grep -n "default=2048" eval/run_lieder_eval.py
# Expected: at least one match on the --max-decode-steps line

grep -n "stage_d_skipped_systems" eval/_scoring_utils.py
# Expected: appears in CSV_HEADER list

grep -n -- "--stage-a-yolo\|--predictions-dir\|--status-jsonl" docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md
# Expected: no output (clean)
```

- [ ] **Step 8: Verification 8 — External tree stub README**

```bash
test -e /home/ari/docs/superpowers/README.md && echo "PASS" || echo "FAIL"
```

Expected: `PASS`.

- [ ] **Step 9: Verification 9 — `docs/superpowers/handoffs/` is empty in the repo**

```bash
ls docs/superpowers/handoffs/ 2>/dev/null | wc -l
```

Expected: `0`.

- [ ] **Step 10: Verification 10 — `eval/results/` only contains placeholders**

```bash
ls eval/results/
```

Expected: only `.gitkeep` and `README.md`.

- [ ] **Step 11: If all verifications pass, push the branch and open a PR**

```bash
git push -u origin feat/repo-depersonalization
gh pr create --title "Repo de-personalization + codex findings" --body "$(cat <<'EOF'
## Summary

- Resolves all six findings from the 2026-05-10 codex gap review.
- Declares the project CUDA-only consistently across README, requirements, install docs, and the test suite.
- Slims the README into a thin landing page; relocates dense architecture, training, evaluation, install, and paths content into dedicated `docs/` files.
- Moves personal artifacts (PowerShell launchers, perf benchmarks, eval results, training logs, NOTES.md, session handoffs) under `archive/`.
- Fixes the eval driver defaults so the corpus eval doesn't silently use the wrong Stage A weights or truncate decoding.
- Adds `stage_d_skipped_systems` to the scoring CSV.
- Mirrors recent `docs/superpowers/` content into the external personal tree and stubs a pointer README there.

## Test plan

- [ ] `pytest tests/data tests/tokenizer tests/decoding -q` passes on a CPU box
- [ ] `pytest tests/inference tests/models tests/cli tests/pipeline tests/train -q` cleanly skips on a CPU box (reason: CUDA required)
- [ ] Full `pytest -q` passes on the GPU box
- [ ] `python -m eval.run_lieder_eval --help` shows new defaults
- [ ] De-personalization grep returns zero hits on canonical paths

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

If verifications fail, address the failures before opening the PR. The PR opens only after all 10 success criteria pass.
