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
