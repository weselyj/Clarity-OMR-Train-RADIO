#!/usr/bin/env python3
"""Offline encoder feature cache builder for Stage 3.

Iterates the combined Stage 3 manifest, filters to the 90% cached tier
(synthetic_systems, grandstaff_systems, primus_systems), runs RadioEncoder
under torch.no_grad() + bf16 autocast, and writes per-sample .pt files to
data/cache/encoder/<hash16>/<tier>/<sample_key>.pt.

Usage (dry-run for Phase 0a sizing):
    python scripts/build_encoder_cache.py \\
        --manifest src/data/manifests/token_manifest_stage3.jsonl \\
        --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --cache-root data/cache/encoder \\
        --batch-size 8 \\
        --device cuda \\
        --dry-run

Usage (full build with resume):
    python scripts/build_encoder_cache.py \\
        --manifest src/data/manifests/token_manifest_stage3.jsonl \\
        --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --cache-root data/cache/encoder \\
        --batch-size 8 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.checkpoint_io import load_stage_b_checkpoint
from src.data.encoder_cache import (
    _sanitize_sample_key,
    cache_entry_exists,
    compute_cache_hash,
    write_cache_entry,
    write_cache_metadata,
)

CACHED_TIER_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}
DRY_RUN_SAMPLE_LIMIT = 1_000


def _load_manifest_entries(manifest_path: Path, cached_only: bool = True) -> list[dict]:
    """Load JSONL manifest, optionally filtering to cached-tier datasets."""
    entries = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if cached_only and entry.get("dataset") not in CACHED_TIER_DATASETS:
                continue
            # Skip entries with null image_path (filtered staves from alignment fix)
            if entry.get("image_path") is None:
                continue
            entries.append(entry)
    return entries


def _load_image_tensor(
    image_path: Path,
    project_root: Path,
    image_height: int,
    image_width: int,
) -> Optional[torch.Tensor]:
    """Load and resize a single image to (1, H, W) float32 tensor in [0, 1]."""
    import torchvision.transforms.functional as TF
    from PIL import Image

    full_path = project_root / image_path if not Path(image_path).is_absolute() else Path(image_path)
    if not full_path.exists():
        return None
    try:
        img = Image.open(full_path).convert("L")
        # Resize to target height, pad width to image_width
        scale = image_height / img.height
        new_w = min(int(img.width * scale), image_width)
        img = img.resize((new_w, image_height), Image.LANCZOS)
        # Create white canvas and paste
        canvas = Image.new("L", (image_width, image_height), color=255)
        canvas.paste(img, (0, 0))
        tensor = TF.to_tensor(canvas)  # (1, H, W) float32 in [0, 1]
        return tensor
    except Exception as exc:
        print(f"[builder] WARNING: failed to load {full_path}: {exc}", file=sys.stderr)
        return None


def _build_cache_for_entries(
    entries: list[dict],
    cache_root: Path,
    hash16: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    project_root: Path,
    image_height: int,
    image_width: int,
    batch_size: int,
    dry_run: bool,
) -> dict:
    """Core builder loop. Returns stats dict.

    Args:
        entries: Manifest entries to process (already filtered to cached tier).
        cache_root: Root directory for cache.
        hash16: Cache identity hash (16 hex chars).
        encode_fn: Callable[image_batch_cpu] -> (B, 1280, H/16, W/16) tensor.
            image_batch_cpu is (B, 1, H, W) float32 on CPU. Output must be
            CPU bf16 with shape (B, 1280, h16, w16).
        project_root: Repo root for resolving relative image paths.
        image_height: Target image height in pixels.
        image_width: Target image width in pixels.
        batch_size: Number of samples per encoder forward pass.
        dry_run: If True, limit to DRY_RUN_SAMPLE_LIMIT entries and don't write.

    Returns:
        Dict with keys: written, skipped_cached, skipped_load_fail, oom_count,
        total_bytes, samples_processed.
    """
    oom_log_path = Path(cache_root) / hash16 / "oom_log.jsonl"

    stats = {
        "written": 0,
        "skipped_cached": 0,
        "skipped_load_fail": 0,
        "oom_count": 0,
        "total_bytes": 0,
        "samples_processed": 0,
    }

    limit = DRY_RUN_SAMPLE_LIMIT if dry_run else len(entries)
    entries_to_process = entries[:limit]

    # Sort by image_path for filesystem locality
    entries_to_process = sorted(
        entries_to_process,
        key=lambda e: str(e.get("image_path", "")),
    )

    # Batch iteration
    i = 0
    while i < len(entries_to_process):
        batch_entries = entries_to_process[i: i + batch_size]
        i += batch_size

        # Check which entries in this batch still need caching
        pending = []
        for entry in batch_entries:
            ds = str(entry.get("dataset", ""))
            sid = str(entry.get("sample_id", ""))
            key = _sanitize_sample_key(sid)
            if cache_entry_exists(cache_root, hash16, ds, key):
                stats["skipped_cached"] += 1
                continue
            pending.append((entry, ds, key))

        if not pending:
            continue

        # Load images for pending entries
        images = []
        valid_pending = []
        for entry, ds, key in pending:
            img_path = entry.get("image_path")
            tensor = _load_image_tensor(
                Path(str(img_path)), project_root, image_height, image_width
            )
            if tensor is None:
                stats["skipped_load_fail"] += 1
                continue
            images.append(tensor)
            valid_pending.append((entry, ds, key))

        if not images:
            continue

        # Stack into batch
        image_batch = torch.stack(images, dim=0)  # (B, 1, H, W)

        if dry_run:
            # In dry-run mode: run encoder on first real batch to measure sizes,
            # then count remaining samples for projection
            try:
                feature_map = encode_fn(image_batch)  # (B, 1280, h16, w16)
                h16 = feature_map.shape[2]
                w16 = feature_map.shape[3]
                seq_tokens = h16 * w16
                bytes_per_sample = seq_tokens * 1280 * 2  # bf16
                stats["written"] += len(valid_pending)
                stats["total_bytes"] += bytes_per_sample * len(valid_pending)
                stats["samples_processed"] += len(valid_pending)
            except torch.cuda.OutOfMemoryError:
                stats["oom_count"] += len(valid_pending)
            continue

        # Run encoder forward with OOM protection
        try:
            feature_map = encode_fn(image_batch)  # (B, 1280, h16, w16)
        except torch.cuda.OutOfMemoryError:
            stats["oom_count"] += len(valid_pending)
            oom_log_path.parent.mkdir(parents=True, exist_ok=True)
            with oom_log_path.open("a") as fh:
                for entry, ds, key in valid_pending:
                    fh.write(json.dumps({"sample_id": entry.get("sample_id"), "oom": True}) + "\n")
            continue

        h16 = feature_map.shape[2]
        w16 = feature_map.shape[3]

        # Write per-sample files
        for b_idx, (entry, ds, key) in enumerate(valid_pending):
            tensor = feature_map[b_idx].cpu().to(torch.bfloat16)
            flat = tensor.flatten(1).transpose(0, 1)  # (h16*w16, 1280) = (seq_tokens, 1280)
            p = write_cache_entry(cache_root, hash16, ds, key, flat, h16=h16, w16=w16)
            stats["written"] += 1
            stats["total_bytes"] += p.stat().st_size
            stats["samples_processed"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("src/data/manifests/token_manifest_stage3.jsonl"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/full_radio_stage2_systems_v2/"
                                     "stage2-radio-systems-polyphonic_best.pt"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/encoder"))
    parser.add_argument("--preproc-cfg", type=Path, default=Path("configs/preproc_stage3.yaml"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help=f"Process only first {DRY_RUN_SAMPLE_LIMIT} samples; print disk projection.")
    parser.add_argument("--ignore-git-sha", action="store_true",
                        help="Omit git HEAD SHA from cache hash (for CI environments).")
    args = parser.parse_args()

    t0 = time.time()

    # Load preprocessing config
    with args.preproc_cfg.open() as fh:
        preproc_cfg = yaml.safe_load(fh)
    print(f"[builder] preproc_cfg: {preproc_cfg}", flush=True)

    # Get git HEAD SHA
    git_head_sha: Optional[str] = None
    if not args.ignore_git_sha:
        try:
            git_head_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            print(f"[builder] git HEAD SHA: {git_head_sha}", flush=True)
        except Exception as exc:
            print(f"[builder] WARNING: could not get git HEAD SHA: {exc}. Use --ignore-git-sha to suppress.", flush=True)
            raise

    # Compute cache hash
    print(f"[builder] computing cache hash from {args.checkpoint}...", flush=True)
    hash16 = compute_cache_hash(
        args.checkpoint, preproc_cfg, "c-radio_v4-h", git_head_sha=git_head_sha
    )
    print(f"[builder] cache hash: {hash16}", flush=True)
    print(f"[builder] cache directory: {args.cache_root / hash16}", flush=True)

    # Pre-flight disk check — walk up to the nearest existing ancestor
    # so we still get a meaningful free-space reading even when the cache
    # root and its parent haven't been created yet.
    _probe = args.cache_root
    while not _probe.exists():
        if _probe == _probe.parent:
            _probe = Path(".").resolve()
            break
        _probe = _probe.parent
    free_bytes = shutil.disk_usage(_probe).free
    print(f"[builder] free disk: {free_bytes / 1e9:.1f} GB", flush=True)

    # Load manifest
    entries = _load_manifest_entries(args.manifest, cached_only=True)
    print(f"[builder] cached-tier entries: {len(entries)}", flush=True)

    if args.dry_run:
        print(f"[builder] DRY RUN: processing first {DRY_RUN_SAMPLE_LIMIT} entries", flush=True)

    # Load model
    from src.tokenizer.vocab import build_default_vocabulary
    from src.train.model_factory import (
        ModelFactoryConfig,
        build_stage_b_components,
        model_factory_config_from_checkpoint_payload,
    )
    print(f"[builder] loading checkpoint: {args.checkpoint}", flush=True)

    # Reconstruct factory config from checkpoint metadata (encoder type, dora_rank, etc.)
    # so that build_stage_b_components produces the correct model architecture and dora_config.
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = build_default_vocabulary()
    fallback_factory_cfg = ModelFactoryConfig(stage_b_vocab_size=vocab.size)
    factory_cfg = model_factory_config_from_checkpoint_payload(
        checkpoint_payload,
        vocab_size=vocab.size,
        fallback=fallback_factory_cfg,
    )
    del checkpoint_payload  # free memory before DoRA wrapping allocates

    device = torch.device(args.device)
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
    print(f"[builder] checkpoint format: {ckpt_result['checkpoint_format']}", flush=True)
    print(
        f"[builder] coverage: {ckpt_result['load_ratio']:.1%}"
        f" ({ckpt_result['loaded_keys']}/{ckpt_result['total_keys']})",
        flush=True,
    )
    print(
        f"[builder] missing={ckpt_result['missing_keys']}"
        f" unexpected={ckpt_result['unexpected_keys']}",
        flush=True,
    )
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.encoder.to(device)
    print(f"[builder] encoder on {device}", flush=True)

    def encode_fn(image_batch_cpu: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Expand grayscale to 3-channel
            if image_batch_cpu.shape[1] == 1:
                image_batch_cpu = image_batch_cpu.repeat(1, 3, 1, 1)
            batch_gpu = image_batch_cpu.to(device)
            feat = model.encoder(batch_gpu)  # (B, 1280, h16, w16)
            return feat.cpu().to(torch.bfloat16)

    stats = _build_cache_for_entries(
        entries=entries,
        cache_root=args.cache_root,
        hash16=hash16,
        encode_fn=encode_fn,
        project_root=ROOT,
        image_height=preproc_cfg.get("image_height", 250),
        image_width=preproc_cfg.get("image_width", 2500),
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    elapsed = time.time() - t0
    print(f"\n[builder] === {'DRY RUN ' if args.dry_run else ''}COMPLETE ===", flush=True)
    print(f"[builder] entries_total:       {len(entries)}", flush=True)
    print(f"[builder] samples_processed:   {stats['samples_processed']}", flush=True)
    print(f"[builder] written:             {stats['written']}", flush=True)
    print(f"[builder] skipped_cached:      {stats['skipped_cached']}", flush=True)
    print(f"[builder] skipped_load_fail:   {stats['skipped_load_fail']}", flush=True)
    print(f"[builder] oom_count:           {stats['oom_count']}", flush=True)
    print(f"[builder] total_bytes_sampled: {stats['total_bytes'] / 1e9:.3f} GB", flush=True)
    print(f"[builder] elapsed_sec:         {elapsed:.1f}", flush=True)

    if args.dry_run and stats["samples_processed"] > 0:
        per_sample_bytes = stats["total_bytes"] / stats["samples_processed"]
        projected_total = per_sample_bytes * len(entries)
        with_overhead = projected_total * 1.5
        print(f"\n[builder] === DISK PROJECTION ===", flush=True)
        print(f"[builder] per_sample_bytes:    {per_sample_bytes / 1e6:.2f} MB", flush=True)
        print(f"[builder] projected_total:     {projected_total / 1e12:.3f} TB ({projected_total / 1e9:.1f} GB)", flush=True)
        print(f"[builder] with_1.5x_overhead:  {with_overhead / 1e12:.3f} TB ({with_overhead / 1e9:.1f} GB)", flush=True)
        print(f"[builder] free_disk:           {free_bytes / 1e9:.1f} GB", flush=True)
        if with_overhead > free_bytes:
            print(f"[builder] WARNING: projected size EXCEEDS free disk. Stop and reassess.", flush=True)
        elif projected_total > 2e12:
            print(f"[builder] WARNING: projected total > 2 TB. Review spec §0a sizing table.", flush=True)
        elif projected_total > 1e12:
            print(f"[builder] CAUTION: 1 TB - 2 TB band. Consider dropping primus from cache.", flush=True)
        elif projected_total > 5e11:
            print(f"[builder] INFO: 500 GB - 1 TB. Verify free disk; proceed if headroom exists.", flush=True)
        else:
            print(f"[builder] INFO: <= 500 GB. Proceed with full cache build.", flush=True)
        return 0

    # Write metadata for full build
    if not args.dry_run:
        write_cache_metadata(args.cache_root, hash16, {
            "encoder_weights_path": str(args.checkpoint),
            "preproc_cfg": preproc_cfg,
            "radio_arch_version": "c-radio_v4-h",
            "git_head_sha": git_head_sha,
            "hash16": hash16,
            "hidden_dim": 1280,
            "dtype": "bfloat16",
            "sample_count": stats["written"],
            "total_bytes": stats["total_bytes"],
            "oom_count": stats["oom_count"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"[builder] metadata written to {args.cache_root / hash16 / 'metadata.json'}", flush=True)

    return 0 if stats["oom_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
