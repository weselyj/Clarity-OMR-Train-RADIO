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
    """Each live opt-step's 8 batches must be emitted contiguously.

    Verified by partitioning the batch sequence into per-opt-step blocks
    (size = grad_accum_live for live, grad_accum_cached for cached) and
    asserting each block is tier-pure: tier transitions only happen on
    opt-step boundaries.
    """
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

    i = 0
    live_blocks_seen = 0
    while i < len(batches):
        tier = _tier_of(batches[i])
        block_size = 1 if tier == "cached" else 8
        assert i + block_size <= len(batches)
        for j in range(block_size):
            assert _tier_of(batches[i + j]) == tier, (
                f"tier transition mid-block at i={i+j}: expected {tier}"
            )
        if tier == "live":
            live_blocks_seen += 1
        i += block_size

    assert live_blocks_seen == 3, f"saw {live_blocks_seen} live blocks, expected 3"


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
    """If grad_accum_cached > 1, cached batches also emit in contiguous blocks.

    Opt-step boundary alignment means: walking the batch sequence in
    per-opt-step blocks (size depends on the tier of the block's first
    batch), each block must be tier-pure. Two adjacent same-tier opt-step
    blocks are allowed (random interleave can land them next to each other);
    what's NOT allowed is a tier transition mid-block.
    """
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
    entries = _make_entries()

    def _tier_of(batch):
        return "cached" if entries[batch[0]]["dataset"] in CACHED_DATASETS else "live"

    # Walk the sequence one opt-step block at a time. The block size is
    # determined by the tier of the first batch. Every batch in the block
    # must be the same tier (i.e. transitions only happen on block boundaries).
    i = 0
    cached_blocks_seen = 0
    live_blocks_seen = 0
    while i < len(batches):
        tier = _tier_of(batches[i])
        block_size = 2 if tier == "cached" else 8
        assert i + block_size <= len(batches), (
            f"truncated block at i={i}: tier={tier} block_size={block_size} but only "
            f"{len(batches) - i} batches remain"
        )
        for j in range(block_size):
            assert _tier_of(batches[i + j]) == tier, (
                f"tier transition mid-block at i={i+j}: expected {tier}, got "
                f"{_tier_of(batches[i + j])}"
            )
        if tier == "cached":
            cached_blocks_seen += 1
        else:
            live_blocks_seen += 1
        i += block_size

    assert cached_blocks_seen == 5, f"saw {cached_blocks_seen} cached blocks, expected 5"
    assert live_blocks_seen == 2, f"saw {live_blocks_seen} live blocks, expected 2"


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
