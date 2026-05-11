"""A2: Encoder output parity between cached features and live encoder.

Stage 3 v2 was trained with a frozen encoder; encoder features for every
training sample were cached to disk and loaded during training. At
inference time the encoder runs live on the input image. If these two
feature distributions differ, the decoder learned to translate one and
sees the other at inference — train/eval skew that no amount of decoder
training can fix.

Cache layout:
  The encoder cache (hash ac8948ae4b5be3e9) stores raw RadioEncoder
  output, i.e. model.encoder(image_batch) → (B, 1280, h16, w16),
  flattened to (seq_tokens, 1280) bfloat16 per sample.

  This is the output BEFORE deformable_attention and positional_bridge
  (which project 1280 → 768 and are part of the trainable model).

  For this parity check we run the same encoder on the inference-time
  preprocessing (BILINEAR resize) and compare the raw 1280-dim features
  against the cached features to quantify the LANCZOS vs BILINEAR gap.

  We also run on the training-time preprocessing (LANCZOS, via
  build_encoder_cache._load_image_tensor) as a sanity-check: this path
  should give near-zero diff against the cache (up to bfloat16 rounding).

Pass criterion: BILINEAR vs cached max abs diff <= 1e-2 (allowing for
fp32 autocast differences). Mean abs diff <= 1e-3.

Cache key note: the cache was built before 2026-05-09 using the legacy
_sanitize_sample_key (/ → __). This script tries the new scheme first,
then falls back to legacy, so it works against both old and new caches.

cameraprimus_systems has no cache tier (not written by
build_encoder_cache.py) so those samples are reported as no_cache_tier
and skipped.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.a2_encoder_parity \\
        --manifest src\\data\\manifests\\token_manifest_stage3.jsonl \\
        --cache-root data\\cache\\encoder \\
        --cache-hash16 ac8948ae4b5be3e9 \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --n-per-corpus 5 \\
        --out audit_results\\a2_encoder.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

# Datasets that have a cache tier. cameraprimus_systems is NOT cached.
_CACHED_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}


def _find_cache_entry(cache_root: Path, hash16: str, tier: str, sample_id: str):
    """Return (tensor, h16, w16, scheme) using new-scheme first, legacy fallback.

    Returns (None, None, None, None) if neither path exists.
    scheme is 'new' or 'legacy'.
    """
    from src.data.encoder_cache import (
        CacheMiss,
        _sanitize_sample_key,
        _sanitize_sample_key_legacy,
        read_cache_entry,
    )

    # New scheme first
    new_key = _sanitize_sample_key(sample_id)
    try:
        tensor, h16, w16 = read_cache_entry(cache_root, hash16, tier, new_key)
        return tensor, h16, w16, "new"
    except CacheMiss:
        pass

    # Legacy fallback (caches built before 2026-05-09)
    legacy_key = _sanitize_sample_key_legacy(sample_id)
    try:
        tensor, h16, w16 = read_cache_entry(cache_root, hash16, tier, legacy_key)
        return tensor, h16, w16, "legacy"
    except CacheMiss:
        pass

    return None, None, None, None


def _load_bilinear_tensor(img_path: Path, image_height: int, image_max_width: int, device):
    """Load and preprocess image using the INFERENCE path (BILINEAR resize).

    Returns (1, 1, H, W) float32 tensor on device, matching
    src.inference.decoder_runtime._load_stage_b_crop_tensor output.
    """
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(img_path) as image_obj:
        gray = image_obj.convert("L")
        scale = float(image_height) / float(max(1, gray.height))
        new_width = max(1, min(image_max_width, int(round(gray.width * scale))))
        resized = gray.resize((new_width, image_height), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (image_max_width, image_height), color=255)
    canvas.paste(resized, (0, 0))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


def _load_lanczos_tensor(img_path: Path, image_height: int, image_max_width: int, device):
    """Load and preprocess image using the TRAINING/CACHE-BUILD path (LANCZOS resize).

    Mirrors build_encoder_cache._load_image_tensor: resize to height with
    LANCZOS, paste onto white canvas, convert to float32 in [0, 1].
    Returns (1, 1, H, W) float32 tensor on device.
    """
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(img_path) as img:
        gray = img.convert("L")
        scale = float(image_height) / float(max(1, gray.height))
        new_width = max(1, min(image_max_width, int(round(gray.width * scale))))
        resized = gray.resize((new_width, image_height), Image.LANCZOS)

    canvas = Image.new("L", (image_max_width, image_height), color=255)
    canvas.paste(resized, (0, 0))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


def _raw_encoder_forward(encoder, pixel_values_1hw):
    """Run RadioEncoder on a (1,1,H,W) tensor → (seq_tokens, 1280) float32.

    Mirrors the cache-build encode_fn:
      - Expand grayscale to 3-channel
      - encoder forward (no_grad, bfloat16 autocast)
      - flatten spatial dims: (B,1280,h16,w16) → (seq_tokens,1280)
    Returns (flat_tensor_float32, h16, w16).
    """
    import torch

    device = pixel_values_1hw.device
    x = pixel_values_1hw
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)  # (1,3,H,W)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feat = encoder(x)  # (1, 1280, h16, w16)
    h16 = feat.shape[2]
    w16 = feat.shape[3]
    # (1, 1280, h16, w16) → (h16*w16, 1280)  same as cache builder
    flat = feat[0].detach().cpu().to(torch.float32).flatten(1).transpose(0, 1).contiguous()
    return flat, h16, w16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Root of the encoder cache (e.g. data/cache/encoder)")
    p.add_argument("--cache-hash16", type=str, required=True,
                   help="16-char cache hash (e.g. ac8948ae4b5be3e9)")
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--n-per-corpus", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    from src.inference.checkpoint_load import load_stage_b_for_inference
    from scripts.audit._sample_picker import pick_audit_samples

    device = torch.device("cuda")
    print(f"Loading Stage B checkpoint: {args.stage_b_ckpt}")
    bundle = load_stage_b_for_inference(args.stage_b_ckpt, device, use_fp16=False)
    # Extract the raw RadioEncoder (frozen; no deformable_attn, no bridge)
    encoder = bundle.decode_model.encoder
    encoder.eval()
    print("Model loaded. Encoder extracted.")

    samples = pick_audit_samples(
        args.manifest, n_per_corpus=args.n_per_corpus, seed=args.seed
    )
    print(f"Selected {len(samples)} samples; comparing cached vs live encoder features...")

    per_sample = []
    for sample in samples:
        sample_id = sample["sample_id"]
        dataset = sample["dataset"]
        img_path = _REPO / sample["image_path"]

        # cameraprimus_systems has no cache tier
        if dataset not in _CACHED_DATASETS:
            per_sample.append({
                "sample_id": sample_id,
                "dataset": dataset,
                "status": "no_cache_tier",
            })
            continue

        # Load cached features (1280-dim, bfloat16, built with LANCZOS)
        cached_tensor, cached_h16, cached_w16, scheme = _find_cache_entry(
            args.cache_root, args.cache_hash16, dataset, sample_id
        )
        if cached_tensor is None:
            per_sample.append({
                "sample_id": sample_id,
                "dataset": dataset,
                "status": "cache_miss",
            })
            continue

        if not img_path.exists():
            per_sample.append({
                "sample_id": sample_id,
                "dataset": dataset,
                "status": "missing_image",
                "image_path": str(img_path),
            })
            continue

        # Convert cached to float32 for comparison
        cached_f32 = cached_tensor.to(dtype=torch.float32)  # (seq_tokens, 1280)

        # Run encoder on BILINEAR-preprocessed image (inference path)
        pv_bilinear = _load_bilinear_tensor(
            img_path, args.image_height, args.image_max_width, device
        )
        live_bilinear, h16_b, w16_b = _raw_encoder_forward(encoder, pv_bilinear)

        # Run encoder on LANCZOS-preprocessed image (training/cache-build path)
        pv_lanczos = _load_lanczos_tensor(
            img_path, args.image_height, args.image_max_width, device
        )
        live_lanczos, h16_l, w16_l = _raw_encoder_forward(encoder, pv_lanczos)

        # Shape checks
        shapes_ok = (cached_f32.shape == live_bilinear.shape == live_lanczos.shape)
        if not shapes_ok:
            per_sample.append({
                "sample_id": sample_id,
                "dataset": dataset,
                "status": "shape_mismatch",
                "cached_shape": list(cached_f32.shape),
                "bilinear_shape": list(live_bilinear.shape),
                "lanczos_shape": list(live_lanczos.shape),
                "cache_key_scheme": scheme,
            })
            continue

        # Comparison 1: BILINEAR (inference) vs cached (trained-on)
        diff_bilinear = (cached_f32 - live_bilinear).numpy()
        max_abs_b = float(abs(diff_bilinear).max())
        mean_abs_b = float(abs(diff_bilinear).mean())
        mean_signed_b = float(diff_bilinear.mean())

        # Comparison 2: LANCZOS (training preproc) vs cached (sanity check)
        diff_lanczos = (cached_f32 - live_lanczos).numpy()
        max_abs_l = float(abs(diff_lanczos).max())
        mean_abs_l = float(abs(diff_lanczos).mean())
        mean_signed_l = float(diff_lanczos.mean())

        per_sample.append({
            "sample_id": sample_id,
            "dataset": dataset,
            "status": "compared",
            # Primary: inference path (BILINEAR) vs cache
            "bilinear_vs_cache_max_abs": max_abs_b,
            "bilinear_vs_cache_mean_abs": mean_abs_b,
            "bilinear_vs_cache_mean_signed": mean_signed_b,
            # Sanity: training path (LANCZOS) vs cache
            "lanczos_vs_cache_max_abs": max_abs_l,
            "lanczos_vs_cache_mean_abs": mean_abs_l,
            "lanczos_vs_cache_mean_signed": mean_signed_l,
            "shape": list(cached_f32.shape),
            "cache_key_scheme": scheme,
        })
        print(
            f"  [{dataset:<22}] "
            f"BILINEAR max={max_abs_b:.4e} mean={mean_abs_b:.4e}  |  "
            f"LANCZOS  max={max_abs_l:.4e} mean={mean_abs_l:.4e}  (scheme={scheme})"
        )

    by_status: dict[str, int] = {}
    for r in per_sample:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1

    compared = [r for r in per_sample if r["status"] == "compared"]

    # Primary pass criterion: BILINEAR (inference) vs cache
    overall_max_bilinear = max(
        (r["bilinear_vs_cache_max_abs"] for r in compared), default=None
    )
    overall_mean_bilinear = (
        sum(r["bilinear_vs_cache_mean_abs"] for r in compared) / len(compared)
        if compared else None
    )

    # Sanity criterion: LANCZOS vs cache (should be near fp16 noise)
    overall_max_lanczos = max(
        (r["lanczos_vs_cache_max_abs"] for r in compared), default=None
    )
    overall_mean_lanczos = (
        sum(r["lanczos_vs_cache_mean_abs"] for r in compared) / len(compared)
        if compared else None
    )

    pass_criterion = (
        overall_max_bilinear is not None
        and overall_max_bilinear <= 1e-2
        and overall_mean_bilinear is not None
        and overall_mean_bilinear <= 1e-3
    )

    results = {
        "experiment": "a2_encoder_parity",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_samples": len(per_sample),
        "by_status": by_status,
        # Primary: BILINEAR (inference path) vs cache (training path)
        "bilinear_vs_cache_overall_max_abs": overall_max_bilinear,
        "bilinear_vs_cache_overall_mean_abs": overall_mean_bilinear,
        # Sanity: LANCZOS vs cache
        "lanczos_vs_cache_overall_max_abs": overall_max_lanczos,
        "lanczos_vs_cache_overall_mean_abs": overall_mean_lanczos,
        "pass": pass_criterion,
        "per_sample": per_sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print("=== A2: Encoder output parity ===")
    print(f"Samples: {len(per_sample)}, status: {by_status}")
    print(f"BILINEAR vs cache — max abs: {overall_max_bilinear},  mean abs: {overall_mean_bilinear}")
    print(f"LANCZOS  vs cache — max abs: {overall_max_lanczos},  mean abs: {overall_mean_lanczos}")
    print(f"PASS (BILINEAR <= 1e-2 max, <= 1e-3 mean): {pass_criterion}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
