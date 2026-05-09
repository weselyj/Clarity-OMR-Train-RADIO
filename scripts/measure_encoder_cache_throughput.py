#!/usr/bin/env python3
"""Throughput and VRAM sweep for Stage 3 two-tier training shape.

Runs the cached dataloader at batch sizes 4, 8, 16, 32 and measures:
  - GPU VRAM usage (peak, reset between each batch size)
  - Step time (forward + backward, no optimizer step)
  - Samples/second

Phase 0 exit criteria #4 (dataloader throughput) and #5 (VRAM sweep).
Recommends b_cached for Phase 1: largest batch size with VRAM <= 80% of total.

Usage:
    python scripts/measure_encoder_cache_throughput.py \\
        --manifest src/data/manifests/token_manifest_stage3.jsonl \\
        --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --cache-root data/cache/encoder \\
        --hash16 <hash16> \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

CACHED_TIER_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}
BATCH_SIZES_TO_TEST = [4, 8, 16, 32]
N_WARMUP_STEPS = 10
N_MEASURE_STEPS = 50
MAX_SEQ_LEN = 512


def _load_cached_entries(manifest: Path) -> list[dict]:
    entries = []
    with manifest.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("dataset") in CACHED_TIER_DATASETS and e.get("image_path"):
                entries.append(e)
    return entries


def _measure_cached_forward(
    model,
    cache_root: Path,
    hash16: str,
    entries: list[dict],
    batch_size: int,
    device: torch.device,
    n_warmup: int,
    n_measure: int,
) -> tuple[float, float]:
    """Run cached forward + backward for n_warmup + n_measure steps.

    Returns:
        (avg_step_sec, peak_vram_gb) where avg is over measure steps only.
        VRAM peak is reset at entry and measured over the full run.
    """
    import random

    from src.data.encoder_cache import _sanitize_sample_key, read_cache_entry

    rng = random.Random(42)
    # Reset peak stats so each batch-size sweep is independent
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    times: list[float] = []
    n_total = n_warmup + n_measure

    for step in range(n_total):
        batch_entries = rng.choices(entries, k=batch_size)
        tensors, h16s, w16s = [], [], []
        for e in batch_entries:
            key = _sanitize_sample_key(e["sample_id"])
            t, h16, w16 = read_cache_entry(cache_root, hash16, e["dataset"], key)
            tensors.append(t)
            h16s.append(h16)
            w16s.append(w16)

        # Assert that all samples in the batch share the same spatial dimensions.
        # Mixed h16/w16 would silently apply wrong positional encoding to all
        # but the first sample, since the bridge is applied uniformly.
        h16s_set = set(h16s)
        w16s_set = set(w16s)
        if len(h16s_set) > 1 or len(w16s_set) > 1:
            raise ValueError(
                f"cached batch has mixed spatial shapes: h16={h16s_set} w16={w16s_set}; "
                f"all cached features in a batch must share spatial dims for the "
                f"positional bridge to be applied correctly"
            )
        h16 = h16s_set.pop()
        w16 = w16s_set.pop()

        # Stack to (B, seq_tokens, 1280) on device
        encoder_hidden = torch.stack(tensors, dim=0).to(device)
        tgt = torch.zeros(batch_size, MAX_SEQ_LEN - 1, dtype=torch.long, device=device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.forward(
                cached_features=encoder_hidden,
                tgt=tgt,
                _h16=h16,
                _w16=w16,
            )
            loss = F.cross_entropy(
                out["logits"].reshape(-1, out["logits"].shape[-1]),
                tgt.reshape(-1).clamp(0),
            )

        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - t0
        if step >= n_warmup:
            times.append(elapsed)

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    avg_sec = sum(times) / len(times) if times else float("nan")
    return avg_sec, peak_gb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("src/data/manifests/token_manifest_stage3.jsonl"),
        help="Path to token_manifest_stage3.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/full_radio_stage2_systems_v2/"
            "stage2-radio-systems-polyphonic_best.pt"
        ),
        help="Stage 2 checkpoint (DoRA-aware loader used automatically)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/cache/encoder"),
        help="Root directory of the encoder cache",
    )
    parser.add_argument(
        "--hash16",
        type=str,
        required=True,
        help="16-char cache hash from build_encoder_cache.py output",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=N_WARMUP_STEPS,
        help=f"Warm-up steps per batch size (default: {N_WARMUP_STEPS})",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=N_MEASURE_STEPS,
        help=f"Measurement steps per batch size (default: {N_MEASURE_STEPS})",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- DoRA-aware checkpoint loading (same pattern as build_encoder_cache.py) ---
    from src.checkpoint_io import load_stage_b_checkpoint
    from src.tokenizer.vocab import build_default_vocabulary
    from src.train.model_factory import (
        ModelFactoryConfig,
        build_stage_b_components,
        model_factory_config_from_checkpoint_payload,
    )

    print(f"[sweep] loading checkpoint: {args.checkpoint}", flush=True)
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = build_default_vocabulary()
    fallback_factory_cfg = ModelFactoryConfig(stage_b_vocab_size=vocab.size)
    factory_cfg = model_factory_config_from_checkpoint_payload(
        checkpoint_payload,
        vocab_size=vocab.size,
        fallback=fallback_factory_cfg,
    )
    del checkpoint_payload  # free memory before DoRA wrapping allocates

    components = build_stage_b_components(factory_cfg)
    model = components["model"]
    dora_config = components.get("dora_config")

    ckpt_result = load_stage_b_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        model=model,
        device=device,
        dora_config=dora_config,
        min_coverage=0.95,
    )
    model = ckpt_result["_model"]
    print(f"[sweep] checkpoint format: {ckpt_result['checkpoint_format']}", flush=True)
    print(
        f"[sweep] coverage: {ckpt_result['load_ratio']:.1%}"
        f" ({ckpt_result['loaded_keys']}/{ckpt_result['total_keys']})",
        flush=True,
    )
    print(
        f"[sweep] missing={ckpt_result['missing_keys']}"
        f" unexpected={ckpt_result['unexpected_keys']}",
        flush=True,
    )

    # Freeze encoder — throughput sweep exercises decoder+bridge only
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.to(device)
    model.train()

    # --- Load manifest entries ---
    cached_entries = _load_cached_entries(args.manifest)
    print(f"[sweep] cached-tier entries: {len(cached_entries)}", flush=True)
    if not cached_entries:
        print("[sweep] ERROR: no cached entries found; check manifest path and datasets", flush=True)
        return 1

    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"[sweep] device total VRAM: {total_vram_gb:.2f} GB", flush=True)
    print(
        f"[sweep] warmup_steps={args.warmup_steps} measure_steps={args.measure_steps}"
        f" (total per bs: {args.warmup_steps + args.measure_steps})",
        flush=True,
    )

    # --- Sweep ---
    results: list[dict] = []
    hdr = f"{'batch':>5} {'avg_step_sec':>14} {'samples/sec':>12} {'peak_vram_gb':>13} {'vram%':>7}  status"
    print(f"\n{hdr}", flush=True)
    print("-" * len(hdr), flush=True)

    for bs in BATCH_SIZES_TO_TEST:
        try:
            avg_sec, peak_gb = _measure_cached_forward(
                model=model,
                cache_root=args.cache_root,
                hash16=args.hash16,
                entries=cached_entries,
                batch_size=bs,
                device=device,
                n_warmup=args.warmup_steps,
                n_measure=args.measure_steps,
            )
            vram_pct = peak_gb / total_vram_gb * 100
            samples_per_sec = bs / avg_sec
            status = "OK" if vram_pct <= 80.0 else "OOM_RISK"
            results.append(
                {
                    "batch_size": bs,
                    "avg_step_sec": avg_sec,
                    "samples_per_sec": samples_per_sec,
                    "peak_vram_gb": peak_gb,
                    "vram_pct": vram_pct,
                    "status": status,
                }
            )
            print(
                f"{bs:>5} {avg_sec:>14.3f} {samples_per_sec:>12.1f}"
                f" {peak_gb:>13.2f} {vram_pct:>6.1f}%  {status}",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": bs, "status": "OOM"})
            print(f"{bs:>5} {'OOM':>14}", flush=True)

    # --- Recommendation ---
    passing = [r for r in results if r.get("status") == "OK"]
    if passing:
        b_cached = max(r["batch_size"] for r in passing)
        best = next(r for r in passing if r["batch_size"] == b_cached)
        grad_accum_cached = max(1, 16 // b_cached)
        print(
            f"\n[sweep] RECOMMENDATION: b_cached={b_cached}"
            f" (VRAM {best['vram_pct']:.1f}%, {best['samples_per_sec']:.1f} samples/sec)",
            flush=True,
        )
        print(
            f"[sweep] grad_accum_cached = 16 / {b_cached} = {grad_accum_cached}"
            f"  (effective batch = 16 for Phase 1 config)",
            flush=True,
        )
    else:
        b_cached = None
        grad_accum_cached = None
        print(
            "\n[sweep] WARNING: no batch size passes VRAM <= 80% constraint."
            " Consider reducing MAX_SEQ_LEN or switching to gradient checkpointing.",
            flush=True,
        )

    # --- Phase 0 exit criteria summary ---
    print("\n[sweep] Phase 0 exit criteria (Phase 0d gate):", flush=True)
    print(
        f"[sweep]   #4 dataloader throughput: step time dominated by backward pass, not I/O?",
        flush=True,
    )
    print(
        f"[sweep]   #5 VRAM sweep:            b_cached recommendation = {b_cached}",
        flush=True,
    )
    print(
        f"[sweep]   Record b_cached and grad_accum_cached in Task 14 handoff doc.",
        flush=True,
    )

    return 0 if b_cached is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
