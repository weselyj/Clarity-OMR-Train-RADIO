#!/usr/bin/env python3
"""Validate encoder cache correctness against live encoder forward.

For N random cached samples:
  1. Load the cached encoder features from disk.
  2. Run live encoder forward on the same image.
  3. Compare the two tensors element-wise.
  4. Report max absolute diff, mean absolute diff, and pass/fail.

Phase 0 exit criterion: max abs diff <= 1e-3 on 100 samples.

Usage:
    python scripts/validate_cache_correctness.py \\
        --manifest src/data/manifests/token_manifest_stage3.jsonl \\
        --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --cache-root data/cache/encoder \\
        --hash16 <16-char-hash> \\
        --n-samples 100 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.checkpoint_io import load_stage_b_checkpoint
from src.data.encoder_cache import _sanitize_sample_key, read_cache_entry

CACHED_TIER_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}


def _load_image(image_path: Path, project_root: Path, image_height: int, image_width: int):
    import torchvision.transforms.functional as TF
    from PIL import Image

    full_path = project_root / image_path if not Path(image_path).is_absolute() else Path(image_path)
    img = Image.open(full_path).convert("L")
    scale = image_height / img.height
    new_w = min(int(img.width * scale), image_width)
    img = img.resize((new_w, image_height), Image.LANCZOS)
    canvas = Image.new("L", (image_width, image_height), color=255)
    canvas.paste(img, (0, 0))
    return TF.to_tensor(canvas)  # (1, H, W) float32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("src/data/manifests/token_manifest_stage3.jsonl"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/full_radio_stage2_systems_v2/"
                                     "stage2-radio-systems-polyphonic_best.pt"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/encoder"))
    parser.add_argument("--hash16", type=str, required=True,
                        help="16-char cache hash from build_encoder_cache.py output")
    parser.add_argument("--preproc-cfg", type=Path, default=Path("configs/preproc_stage3.yaml"))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    with args.preproc_cfg.open() as fh:
        preproc_cfg = yaml.safe_load(fh)
    image_height = preproc_cfg.get("image_height", 250)
    image_width = preproc_cfg.get("image_width", 2500)

    # Load manifest, filter to cached tier with valid image_path
    entries = []
    with args.manifest.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("dataset") in CACHED_TIER_DATASETS and e.get("image_path"):
                entries.append(e)
    print(f"[validate] cached entries with images: {len(entries)}", flush=True)

    rng = random.Random(args.seed)
    sample_entries = rng.sample(entries, min(args.n_samples, len(entries)))
    print(f"[validate] sampling {len(sample_entries)} entries", flush=True)

    # Load model using DoRA-aware checkpoint loader (same pattern as build_encoder_cache.py)
    from src.tokenizer.vocab import build_default_vocabulary
    from src.train.model_factory import (
        ModelFactoryConfig,
        build_stage_b_components,
        model_factory_config_from_checkpoint_payload,
    )
    print(f"[validate] loading checkpoint: {args.checkpoint}", flush=True)

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
    print(f"[validate] checkpoint format: {ckpt_result['checkpoint_format']}", flush=True)
    print(
        f"[validate] coverage: {ckpt_result['load_ratio']:.1%}"
        f" ({ckpt_result['loaded_keys']}/{ckpt_result['total_keys']})",
        flush=True,
    )
    print(
        f"[validate] missing={ckpt_result['missing_keys']}"
        f" unexpected={ckpt_result['unexpected_keys']}",
        flush=True,
    )
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.encoder.to(device)
    print(f"[validate] encoder on {device}", flush=True)

    # Run validation
    max_diffs = []
    mean_diffs = []
    failed = []

    for i, entry in enumerate(sample_entries):
        sid = entry["sample_id"]
        ds = entry["dataset"]
        key = _sanitize_sample_key(sid)

        # Load cached tensor
        try:
            cached_tensor, h16, w16 = read_cache_entry(args.cache_root, args.hash16, ds, key)
        except Exception as exc:
            print(f"[validate] MISS {sid}: {exc}", flush=True)
            failed.append({"sample_id": sid, "reason": "cache_miss"})
            continue

        # Run live encoder on the same image
        try:
            img = _load_image(Path(entry["image_path"]), ROOT, image_height, image_width)
        except Exception as exc:
            print(f"[validate] IMG_FAIL {sid}: {exc}", flush=True)
            failed.append({"sample_id": sid, "reason": "image_load_fail"})
            continue

        img_batch = img.unsqueeze(0).to(device)
        if img_batch.shape[1] == 1:
            img_batch = img_batch.repeat(1, 3, 1, 1)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            live_feat = model.encoder(img_batch)  # (1, 1280, h16, w16)

        live_flat = live_feat[0].cpu().to(torch.bfloat16).flatten(1).transpose(0, 1)  # (seq, 1280)
        cached_cpu = cached_tensor.cpu()

        # Shape must match
        if live_flat.shape != cached_cpu.shape:
            failed.append({
                "sample_id": sid,
                "reason": f"shape_mismatch live={live_flat.shape} cached={cached_cpu.shape}",
            })
            print(
                f"[validate] SHAPE_MISMATCH {sid}: live={live_flat.shape} cached={cached_cpu.shape}",
                flush=True,
            )
            continue

        diff = (live_flat.float() - cached_cpu.float()).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        max_diffs.append(max_d)
        mean_diffs.append(mean_d)

        status = "PASS" if max_d <= args.tolerance else "FAIL"
        if status == "FAIL":
            failed.append({"sample_id": sid, "max_diff": max_d})
        if i % 10 == 0:
            print(
                f"[validate] {i + 1}/{len(sample_entries)} {status}"
                f" max_diff={max_d:.2e} mean_diff={mean_d:.2e}",
                flush=True,
            )

    print(f"\n[validate] === RESULTS ===", flush=True)
    print(f"[validate] samples_checked:  {len(max_diffs)}", flush=True)
    print(f"[validate] samples_failed:   {len(failed)}", flush=True)
    if max_diffs:
        print(f"[validate] max_diff_overall: {max(max_diffs):.4e}", flush=True)
        print(f"[validate] mean_diff_mean:   {sum(mean_diffs) / len(mean_diffs):.4e}", flush=True)
        overall_pass = max(max_diffs) <= args.tolerance
        print(
            f"[validate] PHASE 0 GATE:     {'PASS' if overall_pass else 'FAIL'}"
            f" (tolerance={args.tolerance:.0e})",
            flush=True,
        )
        return 0 if overall_pass else 1
    else:
        print(f"[validate] ERROR: no samples validated", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
