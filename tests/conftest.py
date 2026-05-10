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
