"""Stream 5: Count decoder parameters and compare against reference
implementations.

Loads the Stage 3 v2 checkpoint, counts trainable + total parameters
broken down by encoder vs decoder, and prints a capacity assessment.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.decoder_capacity_analysis \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --out audit_results\\decoder_capacity.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _categorize(name: str) -> str:
    if "encoder" in name:
        return "encoder"
    if "decoder" in name:
        return "decoder"
    if "positional_bridge" in name or "deformable" in name:
        return "bridge"
    if "lm_head" in name or "output" in name.lower() or "vocab" in name.lower():
        return "head"
    if "embed" in name.lower():
        return "embedding"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    ckpt = torch.load(args.stage_b_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    if not isinstance(sd, dict):
        raise RuntimeError("Could not find state_dict in checkpoint")

    by_category = {}
    for name, tensor in sd.items():
        cat = _categorize(name)
        if cat not in by_category:
            by_category[cat] = {"n_params": 0, "n_tensors": 0, "sample_keys": []}
        by_category[cat]["n_params"] += tensor.numel()
        by_category[cat]["n_tensors"] += 1
        if len(by_category[cat]["sample_keys"]) < 5:
            by_category[cat]["sample_keys"].append(name)

    total = sum(c["n_params"] for c in by_category.values())
    summary = {
        "total_params": total,
        "by_category": {
            k: {**v, "fraction": v["n_params"] / total}
            for k, v in by_category.items()
        },
    }

    summary["reference_note"] = (
        "Reference: upstream Clarity-OMR DaViT model is approximately "
        "200-300M params (encoder ~80M, decoder ~120M per published spec). "
        "Compare the decoder count above to ~120M. If significantly smaller, "
        "capacity may be the bottleneck."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "decoder_capacity",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}\n")
    print(f"Total params: {total / 1e6:.2f}M")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<12} {s['n_params']/1e6:>7.2f}M  ({s['fraction']:.1%})  {s['n_tensors']} tensors")
        print(f"    sample keys: {s['sample_keys'][:2]}")
    print(f"\n{summary['reference_note']}")


if __name__ == "__main__":
    main()
