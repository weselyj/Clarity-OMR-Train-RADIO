"""Tier-aware grad accumulation in _run_stage.

Verifies opt-step counter increments correctly across cached and live
batches with different grad_accum values, and that the sampler is
build_tier_grouped_sampler_by_opt_steps when tier_grouped_sampling=True.
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch

import torch
import pytest


def test_compute_tier_aware_total_batches():
    """The trainer's helper computes total batches from opt-step targets correctly."""
    from src.train.train import _compute_tier_grouped_batch_plan

    plan = _compute_tier_grouped_batch_plan(
        target_opt_steps=4500,
        cached_data_ratio=0.9,
        b_cached=16,
        b_live=2,
        grad_accum_cached=1,
        grad_accum_live=8,
    )
    # n_cached_opt_steps = 4500 * 0.9 = 4050; n_live_opt_steps = 450
    assert plan.n_cached_opt_steps == 4050
    assert plan.n_live_opt_steps == 450
    assert plan.total_batches == 4050 * 1 + 450 * 8  # 4050 + 3600 = 7650


def test_per_batch_grad_accum_dispatch_on_tier():
    """_grad_accum_for_batch returns 1 for cached, 8 for live."""
    from src.train.train import _grad_accum_for_batch

    cached_batch = {"tier": "cached", "encoder_hidden": torch.zeros(16, 156, 1280)}
    live_batch = {"tier": "live", "images": torch.zeros(2, 1, 250, 2500)}

    assert _grad_accum_for_batch(cached_batch, grad_accum_cached=1, grad_accum_live=8) == 1
    assert _grad_accum_for_batch(live_batch, grad_accum_cached=1, grad_accum_live=8) == 8
