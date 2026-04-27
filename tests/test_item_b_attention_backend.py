"""Tests for Item B: attention backend resolved once at import time (Tier 2 #6).

Runs on CPU only — no CUDA required.  Tests verify:
  1. Two calls to _run_attention on identical inputs produce identical outputs
     (numerical equivalence, pre- and post-change).
  2. _maybe_flash_attn is NOT called per attention invocation (backend is
     cached at module-level constants _FLASH_ATTN_FUNC and _SDPA_* flags).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="torch required")


class TestAttentionBackendCached:
    """Verify that _run_attention dispatches on cached constants, not per-call probes."""

    def _make_qkv(self, batch=2, heads=4, seq=8, head_dim=16, dtype=None):
        """Build deterministic Q/K/V tensors on CPU."""
        if dtype is None:
            dtype = torch.float32
        gen = torch.Generator()
        gen.manual_seed(42)
        q = torch.randn(batch, heads, seq, head_dim, generator=gen, dtype=dtype)
        k = torch.randn(batch, heads, seq, head_dim, generator=gen, dtype=dtype)
        v = torch.randn(batch, heads, seq, head_dim, generator=gen, dtype=dtype)
        return q, k, v

    def test_identical_outputs_on_repeated_call(self):
        """Two calls with the same input produce identical outputs (deterministic)."""
        from src.models.davit_stage_b import _run_attention

        q, k, v = self._make_qkv()
        out1 = _run_attention(q, k, v, causal=False)
        out2 = _run_attention(q, k, v, causal=False)
        assert torch.allclose(out1, out2, atol=0.0, rtol=0.0), (
            "Expected identical outputs on repeated call — non-determinism detected."
        )

    def test_identical_outputs_causal(self):
        """Causal=True also produces identical outputs on repeated call."""
        from src.models.davit_stage_b import _run_attention

        q, k, v = self._make_qkv()
        out1 = _run_attention(q, k, v, causal=True)
        out2 = _run_attention(q, k, v, causal=True)
        assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)

    def test_maybe_flash_attn_not_called_per_invocation(self):
        """_maybe_flash_attn must NOT be called during _run_attention.

        The backend is resolved at import time into _FLASH_ATTN_FUNC.
        If _maybe_flash_attn is called inside _run_attention, it would still
        read env vars and probe flash_attn on every forward pass — defeating
        the purpose of the refactor.
        """
        import src.models.davit_stage_b as module

        q, k, v = self._make_qkv()

        call_count = 0
        original_fn = module._maybe_flash_attn

        def counting_maybe_flash_attn():
            nonlocal call_count
            call_count += 1
            return original_fn()

        with patch.object(module, "_maybe_flash_attn", side_effect=counting_maybe_flash_attn):
            module._run_attention(q, k, v, causal=False)
            module._run_attention(q, k, v, causal=False)

        assert call_count == 0, (
            f"_maybe_flash_attn was called {call_count} time(s) inside _run_attention; "
            "expected 0 — backend must be cached at module level."
        )

    def test_module_level_constants_exist(self):
        """The cached backend constants must be present at module level."""
        import src.models.davit_stage_b as module

        # _FLASH_ATTN_FUNC is None (not installed) or a callable.
        assert hasattr(module, "_FLASH_ATTN_FUNC"), "missing _FLASH_ATTN_FUNC"
        assert module._FLASH_ATTN_FUNC is None or callable(module._FLASH_ATTN_FUNC)

        # SDPA control flags must exist and be bool.
        for attr in ("_SDPA_DISABLE_FLASH", "_SDPA_DISABLE_MEM_EFFICIENT", "_SDPA_FORCE_MATH"):
            assert hasattr(module, attr), f"missing {attr}"
            assert isinstance(getattr(module, attr), bool), f"{attr} must be bool"

    def test_output_shape_matches_input(self):
        """Output tensor shape must equal Q shape."""
        from src.models.davit_stage_b import _run_attention

        q, k, v = self._make_qkv(batch=3, heads=8, seq=16, head_dim=32)
        out = _run_attention(q, k, v, causal=False)
        assert out.shape == q.shape, f"expected {q.shape}, got {out.shape}"
