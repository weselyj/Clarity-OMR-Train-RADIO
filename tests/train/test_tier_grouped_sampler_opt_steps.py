"""Tier-grouped sampler opt-step semantics — live batches in contiguous blocks."""
from __future__ import annotations

import pytest


CACHED_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}
LIVE_DATASETS = {"cameraprimus_systems"}


def _make_entries(n_cached: int = 100, n_live: int = 100):
    out = []
    for i in range(n_cached):
        out.append({"dataset": "synthetic_systems", "split": "train", "sample_id": f"c{i}"})
    for i in range(n_live):
        out.append({"dataset": "cameraprimus_systems", "split": "train", "sample_id": f"l{i}"})
    return out


def test_opt_step_api_emits_correct_batch_counts():
    from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

    batches = build_tier_grouped_sampler_by_opt_steps(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        n_cached_opt_steps=10,
        n_live_opt_steps=2,
        b_cached=16,
        b_live=2,
        grad_accum_cached=1,
        grad_accum_live=8,
        seed=42,
    )

    # 10 cached opt-steps × 1 batch each = 10 cached batches
    # 2 live opt-steps × 8 batches each = 16 live batches
    # Total = 26 batches
    assert len(batches) == 26


def test_live_batches_are_contiguous_8_blocks():
    """Each live opt-step's 8 batches must be emitted contiguously."""
    from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

    batches = build_tier_grouped_sampler_by_opt_steps(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        n_cached_opt_steps=20,
        n_live_opt_steps=3,
        b_cached=16,
        b_live=2,
        grad_accum_cached=1,
        grad_accum_live=8,
        seed=42,
    )
    entries = _make_entries()

    def _tier_of(batch):
        idx = batch[0]
        return "cached" if entries[idx]["dataset"] in CACHED_DATASETS else "live"

    # Walk batch sequence; live runs must be exactly 8 long.
    i = 0
    while i < len(batches):
        if _tier_of(batches[i]) == "live":
            run_len = 0
            while i < len(batches) and _tier_of(batches[i]) == "live":
                run_len += 1
                i += 1
            assert run_len == 8, f"expected live run of 8 batches, got {run_len}"
        else:
            i += 1


def test_all_batches_tier_pure():
    """Same invariant as Phase 0 tier sampler — every batch is single-tier."""
    from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

    batches = build_tier_grouped_sampler_by_opt_steps(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        n_cached_opt_steps=50,
        n_live_opt_steps=5,
        b_cached=16,
        b_live=2,
        grad_accum_cached=1,
        grad_accum_live=8,
        seed=7,
    )
    entries = _make_entries()
    for batch in batches:
        tiers = set()
        for idx in batch:
            ds = entries[idx]["dataset"]
            tiers.add("cached" if ds in CACHED_DATASETS else "live")
        assert len(tiers) == 1, f"mixed-tier batch: {tiers}"


def test_grad_accum_cached_greater_than_one():
    """If grad_accum_cached > 1, cached batches also emit in contiguous blocks."""
    from src.train.tier_sampler import build_tier_grouped_sampler_by_opt_steps

    batches = build_tier_grouped_sampler_by_opt_steps(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        n_cached_opt_steps=5,
        n_live_opt_steps=2,
        b_cached=8,
        b_live=2,
        grad_accum_cached=2,  # 2 micro-batches per cached opt-step
        grad_accum_live=8,
        seed=0,
    )

    # 5 cached opt-steps × 2 batches + 2 live opt-steps × 8 batches = 26
    assert len(batches) == 26
    # First batch's tier dictates the run; verify cached runs are exactly 2.
    entries = _make_entries()

    def _tier_of(batch):
        return "cached" if entries[batch[0]]["dataset"] in CACHED_DATASETS else "live"

    i = 0
    while i < len(batches):
        tier = _tier_of(batches[i])
        run_len = 0
        while i < len(batches) and _tier_of(batches[i]) == tier:
            run_len += 1
            i += 1
        expected = 2 if tier == "cached" else 8
        assert run_len == expected, f"{tier} run length {run_len} != {expected}"


def test_legacy_api_still_works():
    """Phase 0's build_tier_grouped_sampler API stays callable."""
    from src.train.tier_sampler import build_tier_grouped_sampler

    batches = build_tier_grouped_sampler(
        entries=_make_entries(),
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.9,
        total_batches=100,
        b_cached=8,
        b_live=2,
        seed=0,
    )
    assert len(batches) == 100
