"""Tests for src/train/tier_sampler.py::build_tier_grouped_sampler."""
from __future__ import annotations

import random
from collections import Counter

import pytest


from src.train.train import _CACHED_DATASETS as CACHED_DATASETS

LIVE_DATASETS = {"cameraprimus_systems"}


def _make_mock_entries(n_cached: int, n_live: int) -> list[dict]:
    entries = []
    for i in range(n_cached):
        ds = list(CACHED_DATASETS)[i % len(CACHED_DATASETS)]
        entries.append({"dataset": ds, "sample_id": f"cached_{i}"})
    for i in range(n_live):
        entries.append({"dataset": "cameraprimus_systems", "sample_id": f"live_{i}"})
    return entries


def test_all_batches_are_tier_pure() -> None:
    """Every batch returned by the sampler must contain samples from exactly one tier."""
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=900, n_live=100)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=200,
        b_cached=8,
        b_live=2,
        seed=42,
    )

    for batch_idx, batch in enumerate(batches):
        tiers = set()
        for sample_idx in batch:
            ds = entries[sample_idx]["dataset"]
            if ds in CACHED_DATASETS:
                tiers.add("cached")
            else:
                tiers.add("live")
        assert len(tiers) == 1, (
            f"Batch {batch_idx} mixes tiers: {tiers}. "
            f"sample datasets: {[entries[i]['dataset'] for i in batch]}"
        )


def test_cached_batch_size_is_b_cached() -> None:
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=900, n_live=100)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=200,
        b_cached=8,
        b_live=2,
        seed=0,
    )
    for batch in batches:
        ds = entries[batch[0]]["dataset"]
        tier = "cached" if ds in CACHED_DATASETS else "live"
        expected_bs = 8 if tier == "cached" else 2
        assert len(batch) == expected_bs, (
            f"Batch has {len(batch)} samples but expected {expected_bs} for tier={tier}"
        )


def test_ratio_approximately_90_10() -> None:
    """Long-run cached batch fraction should be 90% ±5%."""
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=9000, n_live=1000)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=1000,
        b_cached=8,
        b_live=2,
        seed=7,
    )
    n_cached_batches = sum(
        1 for b in batches if entries[b[0]]["dataset"] in CACHED_DATASETS
    )
    frac = n_cached_batches / len(batches)
    assert 0.85 <= frac <= 0.95, f"Cached batch fraction {frac:.3f} outside 85–95% band"


def test_returns_list_of_lists() -> None:
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=90, n_live=10)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=50,
        b_cached=4,
        b_live=2,
        seed=1,
    )
    assert isinstance(batches, list)
    assert all(isinstance(b, list) for b in batches)
    assert all(isinstance(idx, int) for b in batches for idx in b)


def test_indices_are_valid() -> None:
    """All returned indices must be in [0, len(entries))."""
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=100, n_live=20)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=50,
        b_cached=4,
        b_live=2,
        seed=99,
    )
    n = len(entries)
    for batch in batches:
        for idx in batch:
            assert 0 <= idx < n, f"Index {idx} out of range [0, {n})"
