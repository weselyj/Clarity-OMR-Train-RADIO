"""Tier-grouped batch sampler for Stage 3 two-tier dataloader.

Guarantees that each batch is 100% from one tier (cached or live). This is
required because cached batches use b_cached (8 or 16) while live batches
use b_live=2, and the model forward path dispatches on the tier key.

The sampler pre-computes a list of batched index lists, interleaved at the
specified cached_ratio. Indices are drawn with replacement within each tier.
"""
from __future__ import annotations

import random


def build_tier_grouped_sampler(
    entries: list[dict],
    cached_datasets: set[str],
    live_datasets: set[str],
    cached_ratio: float,
    total_batches: int,
    b_cached: int,
    b_live: int,
    seed: int = 0,
) -> list[list[int]]:
    """Build a list of tier-pure batched index lists.

    Args:
        entries: Full dataset entries list (same order as dataset.entries).
        cached_datasets: Set of dataset names that are in the cached tier.
        live_datasets: Set of dataset names that are in the live tier.
        cached_ratio: Fraction of batches that should be from the cached tier
            (e.g. 0.90 for 90% cached / 10% live).
        total_batches: Total number of batches to generate.
        b_cached: Batch size for cached-tier batches.
        b_live: Batch size for live-tier batches.
        seed: Random seed for reproducibility.

    Returns:
        A list of total_batches lists. Each inner list contains integer indices
        into `entries`. All indices in a given inner list come from the same tier.
    """
    rng = random.Random(seed)

    # Partition entry indices by tier
    cached_indices = [
        i for i, e in enumerate(entries)
        if e.get("dataset") in cached_datasets
    ]
    live_indices = [
        i for i, e in enumerate(entries)
        if e.get("dataset") in live_datasets
    ]

    if not cached_indices:
        raise ValueError(
            f"build_tier_grouped_sampler: no entries found for cached tier. "
            f"cached_datasets={cached_datasets}"
        )
    if not live_indices:
        raise ValueError(
            f"build_tier_grouped_sampler: no entries found for live tier. "
            f"live_datasets={live_datasets}"
        )

    # Determine how many cached vs live batches to generate
    n_cached_batches = round(total_batches * cached_ratio)
    n_live_batches = total_batches - n_cached_batches

    # Generate per-tier batches (with replacement)
    def _draw_batches(indices: list[int], batch_size: int, n_batches: int) -> list[list[int]]:
        batches = []
        for _ in range(n_batches):
            batch = [rng.choice(indices) for _ in range(batch_size)]
            batches.append(batch)
        return batches

    cached_batches = _draw_batches(cached_indices, b_cached, n_cached_batches)
    live_batches = _draw_batches(live_indices, b_live, n_live_batches)

    # Interleave cached and live batches in proportion (shuffle by tier assignment)
    # Deterministic shuffle: alternate with occasional live batch
    tier_sequence: list[str] = (["cached"] * n_cached_batches) + (["live"] * n_live_batches)
    rng.shuffle(tier_sequence)

    cached_iter = iter(cached_batches)
    live_iter = iter(live_batches)
    result: list[list[int]] = []
    for tier in tier_sequence:
        if tier == "cached":
            result.append(next(cached_iter))
        else:
            result.append(next(live_iter))

    return result


def build_tier_grouped_sampler_by_opt_steps(
    entries: list[dict],
    cached_datasets: set[str],
    live_datasets: set[str],
    *,
    n_cached_opt_steps: int,
    n_live_opt_steps: int,
    b_cached: int,
    b_live: int,
    grad_accum_cached: int,
    grad_accum_live: int,
    seed: int = 0,
) -> list[list[int]]:
    """Build a list of tier-pure batched index lists where opt-step boundaries
    coincide with tier transitions.

    The sampler emits per-opt-step blocks: each cached opt-step is
    ``grad_accum_cached`` consecutive cached batches; each live opt-step is
    ``grad_accum_live`` consecutive live batches. Opt-step blocks are then
    randomly interleaved so the trainer sees an unpredictable cached/live
    mix at the opt-step level — but a cached batch never interrupts a live
    accumulation window.

    Args:
        entries: Full dataset entries list (same order as dataset.entries).
        cached_datasets: Set of dataset names that are in the cached tier.
        live_datasets: Set of dataset names that are in the live tier.
        n_cached_opt_steps: Number of cached-tier opt-steps to emit.
        n_live_opt_steps: Number of live-tier opt-steps to emit.
        b_cached: Batch size for cached batches.
        b_live: Batch size for live batches.
        grad_accum_cached: Number of cached batches per cached opt-step.
        grad_accum_live: Number of live batches per live opt-step.
        seed: Random seed for reproducibility.

    Returns:
        A flat list of batches, total length
        ``n_cached_opt_steps * grad_accum_cached + n_live_opt_steps * grad_accum_live``.
    """
    rng = random.Random(seed)

    cached_indices = [i for i, e in enumerate(entries) if e.get("dataset") in cached_datasets]
    live_indices = [i for i, e in enumerate(entries) if e.get("dataset") in live_datasets]

    if not cached_indices and n_cached_opt_steps > 0:
        raise ValueError(
            f"build_tier_grouped_sampler_by_opt_steps: n_cached_opt_steps={n_cached_opt_steps} "
            f"but no entries match cached_datasets={cached_datasets}"
        )
    if not live_indices and n_live_opt_steps > 0:
        raise ValueError(
            f"build_tier_grouped_sampler_by_opt_steps: n_live_opt_steps={n_live_opt_steps} "
            f"but no entries match live_datasets={live_datasets}"
        )

    def _draw_batch(pool: list[int], size: int) -> list[int]:
        return [rng.choice(pool) for _ in range(size)]

    # Build per-opt-step blocks.
    cached_blocks: list[list[list[int]]] = [
        [_draw_batch(cached_indices, b_cached) for _ in range(grad_accum_cached)]
        for _ in range(n_cached_opt_steps)
    ]
    live_blocks: list[list[list[int]]] = [
        [_draw_batch(live_indices, b_live) for _ in range(grad_accum_live)]
        for _ in range(n_live_opt_steps)
    ]

    # Interleave at the opt-step block level.
    block_order: list[str] = (["cached"] * n_cached_opt_steps) + (["live"] * n_live_opt_steps)
    rng.shuffle(block_order)

    cached_iter = iter(cached_blocks)
    live_iter = iter(live_blocks)
    result: list[list[int]] = []
    for tier in block_order:
        block = next(cached_iter) if tier == "cached" else next(live_iter)
        result.extend(block)

    return result
