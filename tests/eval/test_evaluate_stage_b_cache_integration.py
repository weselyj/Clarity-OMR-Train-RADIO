# tests/eval/test_evaluate_stage_b_cache_integration.py
"""TDD tests for encoder-cache integration in evaluate_stage_b_checkpoint.py.

Tests exercise _resolve_cache_memory — the helper extracted from the per-sample
loop in _run_stage_b_inference_with_progress — via mocking.  All heavy model
and I/O operations are mocked so these tests run CPU-only without PyTorch.

Torch is imported lazily inside implementation code (not at module level here)
so tests can run on machines where torch is not installed by patching sys.modules
at collection time.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Stub out torch so tests collect on machines without PyTorch
# ---------------------------------------------------------------------------

def _make_torch_stub():
    """Return a minimal torch stub sufficient for the cache integration tests."""
    stub = types.ModuleType("torch")

    class _FakeTensor:
        def __init__(self, shape=(8, 1280), dtype=None):
            self.shape = shape
            self.dtype = dtype or _FakeDtype("bfloat16")

        def to(self, **kwargs):
            return self

        def float(self):
            return self

        def half(self):
            return self

    class _FakeDtype:
        def __init__(self, name):
            self._name = name

        def __repr__(self):
            return f"torch.{self._name}"

    stub.Tensor = _FakeTensor
    stub.bfloat16 = _FakeDtype("bfloat16")
    stub.float32 = _FakeDtype("float32")
    stub.float16 = _FakeDtype("float16")
    stub.device = lambda s: MagicMock(type=s.split(":")[0] if ":" not in s else s)
    stub.inference_mode = lambda: MagicMock(__enter__=lambda *a: None, __exit__=lambda *a: None)
    stub.load = MagicMock(return_value={})
    stub.no_grad = MagicMock(return_value=MagicMock(__enter__=lambda *a: None, __exit__=lambda *a: None))
    stub.cuda = MagicMock()
    stub.cuda.is_available = MagicMock(return_value=False)
    return stub


if "torch" not in sys.modules:
    sys.modules["torch"] = _make_torch_stub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CACHE_MOD = "src.data.encoder_cache"
_MOD = "src.eval.evaluate_stage_b_checkpoint"
# _encode_staff_image is imported from src.inference.decoder_runtime inside
# _resolve_cache_memory's body (post Phase A archival cleanup), so we patch it
# there — that's where the name is bound at call time.
_DECODER_MOD = "src.inference.decoder_runtime"


def _fake_memory():
    """Return a fake memory object (stands in for the bf16 tensor)."""
    return sys.modules["torch"].Tensor(shape=(8, 1280))


def _fake_cached_entry():
    """Return what read_cache_entry returns: (tensor, h16, w16)."""
    return (_fake_memory(), 2, 4)


def _fake_device(type_str="cpu"):
    dev = MagicMock()
    dev.type = type_str
    dev.__str__ = lambda self: type_str
    return dev


# ---------------------------------------------------------------------------
# Test 1: cached dataset uses cache when root is provided
# ---------------------------------------------------------------------------

class TestCachedDatasetUsesCacheWhenRootProvided:
    """read_cache_entry is called for synthetic_systems; _encode_staff_image is NOT."""

    def test_cached_dataset_uses_cache_when_root_provided(self, tmp_path: Path) -> None:
        from src.eval.evaluate_stage_b_checkpoint import _resolve_cache_memory

        device = _fake_device("cpu")
        sentinel_memory = MagicMock(name="post_bridge_memory")

        with patch(f"{_CACHE_MOD}.read_cache_entry", return_value=_fake_cached_entry()) as mock_read, \
             patch(f"{_DECODER_MOD}._encode_staff_image") as mock_encode, \
             patch(f"{_MOD}._encode_from_cached_features", return_value=sentinel_memory) as mock_bridge:

            result = _resolve_cache_memory(
                decode_model=MagicMock(),
                pixel_values=MagicMock(),
                dataset="synthetic_systems",
                sample_id="synthetic_systems:page001__sys00",
                cache_root=tmp_path / "cache",
                cache_hash="abc123",
                device=device,
                use_fp16=False,
            )

        mock_read.assert_called_once()
        mock_bridge.assert_called_once()  # cache hit → bridge runs
        mock_encode.assert_not_called()  # encoder skipped
        assert result is sentinel_memory


# ---------------------------------------------------------------------------
# Test 2: uncached dataset runs encoder forward
# ---------------------------------------------------------------------------

class TestUncachedDatasetRunsEncoderForward:
    """For cameraprimus_systems, cache is NOT consulted; encoder forward IS called."""

    def test_uncached_dataset_runs_encoder_forward(self, tmp_path: Path) -> None:
        from src.eval.evaluate_stage_b_checkpoint import _resolve_cache_memory

        fake_memory = _fake_memory()
        device = _fake_device("cpu")

        with patch(f"{_CACHE_MOD}.read_cache_entry") as mock_read, \
             patch(f"{_DECODER_MOD}._encode_staff_image", return_value=fake_memory) as mock_encode:

            result = _resolve_cache_memory(
                decode_model=MagicMock(),
                pixel_values=MagicMock(),
                dataset="cameraprimus_systems",
                sample_id="cameraprimus_systems:sample001",
                cache_root=tmp_path / "cache",
                cache_hash="abc123",
                device=device,
                use_fp16=False,
            )

        mock_read.assert_not_called()
        mock_encode.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# Test 3: no cache root means encoder forward for all datasets
# ---------------------------------------------------------------------------

class TestNoCacheRootMeansEncoderForwardForAll:
    """When cache_root=None, all samples (including cached datasets) hit encoder path."""

    def test_no_cache_root_means_encoder_forward_for_all(self, tmp_path: Path) -> None:
        from src.eval.evaluate_stage_b_checkpoint import _resolve_cache_memory

        fake_memory = _fake_memory()
        device = _fake_device("cpu")

        with patch(f"{_CACHE_MOD}.read_cache_entry") as mock_read, \
             patch(f"{_DECODER_MOD}._encode_staff_image", return_value=fake_memory) as mock_encode:

            result = _resolve_cache_memory(
                decode_model=MagicMock(),
                pixel_values=MagicMock(),
                dataset="synthetic_systems",   # normally a cached dataset
                sample_id="synthetic_systems:page001__sys00",
                cache_root=None,               # no cache root provided
                cache_hash="abc123",
                device=device,
                use_fp16=False,
            )

        mock_read.assert_not_called()
        mock_encode.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: cache miss (both new + legacy key) falls back to encoder forward
# ---------------------------------------------------------------------------

class TestCacheMissFallsBackToEncoderForward:
    """If read_cache_entry raises FileNotFoundError on both new+legacy keys,
    the helper falls back to _encode_staff_image."""

    def test_cache_miss_falls_back_to_encoder_forward(self, tmp_path: Path) -> None:
        from src.eval.evaluate_stage_b_checkpoint import _resolve_cache_memory

        fake_memory = _fake_memory()
        device = _fake_device("cpu")

        # Both new-scheme and legacy-scheme raise FileNotFoundError
        with patch(f"{_CACHE_MOD}.read_cache_entry", side_effect=FileNotFoundError("miss")) as mock_read, \
             patch(f"{_DECODER_MOD}._encode_staff_image", return_value=fake_memory) as mock_encode:

            result = _resolve_cache_memory(
                decode_model=MagicMock(),
                pixel_values=MagicMock(),
                dataset="synthetic_systems",
                sample_id="synthetic_systems:page001__sys00",
                cache_root=tmp_path / "cache",
                cache_hash="abc123",
                device=device,
                use_fp16=False,
            )

        # read_cache_entry called at least once (new + legacy = 2 total)
        assert mock_read.call_count >= 1
        mock_encode.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# Test 5: legacy key fallback on FileNotFoundError from new scheme
# ---------------------------------------------------------------------------

class TestLegacyKeyFallbackOnFileNotFound:
    """New-scheme key raises FileNotFoundError; legacy key succeeds.
    Verifies the two-attempt pattern mirrored from train.py ~line 670."""

    def test_legacy_key_fallback_on_filenotfound(self, tmp_path: Path) -> None:
        from src.eval.evaluate_stage_b_checkpoint import _resolve_cache_memory
        from src.data.encoder_cache import (
            _sanitize_sample_key,
            _sanitize_sample_key_legacy,
        )

        fake_entry = _fake_cached_entry()
        device = _fake_device("cpu")

        # '/' in the sample ID makes new and legacy keys differ:
        # new  → page001_SLASH_sys00
        # legacy → page001__sys00
        sample_id = "synthetic_systems:page001/sys00"
        new_key = _sanitize_sample_key(sample_id)
        legacy_key = _sanitize_sample_key_legacy(sample_id)
        assert new_key != legacy_key, "Test requires keys to differ (sanity check)"

        def side_effect(cache_root, hash16, tier, key):
            if key == new_key:
                raise FileNotFoundError(f"miss new key: {key}")
            if key == legacy_key:
                return fake_entry
            raise FileNotFoundError(f"unexpected key: {key}")

        sentinel_memory = MagicMock(name="post_bridge_memory")
        with patch(f"{_CACHE_MOD}.read_cache_entry", side_effect=side_effect) as mock_read, \
             patch(f"{_DECODER_MOD}._encode_staff_image") as mock_encode, \
             patch(f"{_MOD}._encode_from_cached_features", return_value=sentinel_memory) as mock_bridge:

            result = _resolve_cache_memory(
                decode_model=MagicMock(),
                pixel_values=MagicMock(),
                dataset="synthetic_systems",
                sample_id=sample_id,
                cache_root=tmp_path / "cache",
                cache_hash="abc123",
                device=device,
                use_fp16=False,
            )

        # read_cache_entry called exactly twice (new first, legacy on miss)
        assert mock_read.call_count == 2
        # encoder NOT called because legacy key succeeded
        mock_encode.assert_not_called()
        assert result is not None
