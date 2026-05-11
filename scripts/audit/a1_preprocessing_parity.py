"""A1: Image preprocessing parity between training and inference pipelines.

For each audit sample, load the image through both the training-time
preprocessing path (src.train.train._load_raster_image_tensor) and the
inference-time path (src.inference.decoder_runtime._load_stage_b_crop_tensor),
then compare the resulting tensors element-wise.

Pass criterion: tensors bit-identical OR within fp32 tolerance (max abs
diff <= 1e-6) for every sample. Any larger diff is a finding worth
reporting in the audit.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.a1_preprocessing_parity \\
        --manifest data\\processed\\<combined_manifest>.jsonl \\
        --n-per-corpus 5 \\
        --out audit_results\\a1_preprocessing.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to combined training manifest (jsonl)")
    p.add_argument("--n-per-corpus", type=int, default=5)
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from scripts.audit._sample_picker import pick_audit_samples
    samples = pick_audit_samples(args.manifest, n_per_corpus=args.n_per_corpus, seed=args.seed)
    print(f"Selected {len(samples)} audit samples")

    results = _run_parity(samples, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    _print_summary(results)


def _run_parity(samples, args):
    import numpy as np
    import torch
    from PIL import Image
    from src.train.train import _load_raster_image_tensor
    from src.inference.decoder_runtime import _load_stage_b_crop_tensor

    per_sample = []
    for sample in samples:
        img_path = _REPO / sample["image_path"]
        if not img_path.exists():
            per_sample.append({
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "status": "missing_image",
                "image_path": str(img_path),
            })
            continue

        # Training-time path. Returns (tensor[1, H, W], content_width).
        train_tensor, train_cw = _load_raster_image_tensor(
            img_path, height=args.image_height, max_width=args.image_max_width
        )

        # Inference-time path. Returns tensor[1, 1, H, W] on a torch device.
        inf_tensor = _load_stage_b_crop_tensor(
            img_path,
            image_height=args.image_height,
            image_max_width=args.image_max_width,
            device=torch.device("cpu"),
        )

        # Normalize shapes for comparison.
        if train_tensor.dim() == 3:
            train_normalized = train_tensor.unsqueeze(0)  # [1, 1, H, W]
        else:
            train_normalized = train_tensor
        if inf_tensor.dim() == 4 and inf_tensor.shape[1] == 1:
            inf_normalized = inf_tensor
        else:
            inf_normalized = inf_tensor

        train_arr = train_normalized.detach().cpu().numpy().astype(np.float32)
        inf_arr = inf_normalized.detach().cpu().numpy().astype(np.float32)

        if train_arr.shape != inf_arr.shape:
            per_sample.append({
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "status": "shape_mismatch",
                "train_shape": list(train_arr.shape),
                "inference_shape": list(inf_arr.shape),
            })
            continue

        diff = train_arr - inf_arr
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        nonzero_pixels = int(np.sum(diff != 0))
        total_pixels = int(np.prod(diff.shape))
        per_sample.append({
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "status": "compared",
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "nonzero_pixel_fraction": nonzero_pixels / total_pixels,
            "shape": list(train_arr.shape),
        })

    by_status = {}
    for r in per_sample:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1
    overall_max = max(
        (r["max_abs_diff"] for r in per_sample if r["status"] == "compared"),
        default=None,
    )
    pass_criterion = overall_max is not None and overall_max <= 1e-6
    return {
        "experiment": "a1_preprocessing_parity",
        "args": vars_serializable(args),
        "n_samples": len(per_sample),
        "by_status": by_status,
        "overall_max_abs_diff": overall_max,
        "pass": pass_criterion,
        "per_sample": per_sample,
    }


def vars_serializable(args):
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _print_summary(results):
    print()
    print(f"=== A1: Image preprocessing parity ===")
    print(f"Samples: {results['n_samples']}")
    print(f"Status:  {results['by_status']}")
    print(f"Max abs diff (overall): {results['overall_max_abs_diff']}")
    print(f"PASS: {results['pass']}")


if __name__ == "__main__":
    main()
