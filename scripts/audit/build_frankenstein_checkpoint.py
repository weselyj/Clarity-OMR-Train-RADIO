"""Build a frankenstein inference checkpoint: Stage 2 v2 encoder + Stage 3 v2
non-encoder weights.

Why: the Stage 3 v2 training audit (PR #48) found the encoder DoRA adapters
were silently updated during Stage 3 v2 training. The decoder trained against
encoder features from the Stage 2 v2 cache; at inference the live encoder is
the drifted Stage 3 v2 version. This script reconstructs the train-time
encoder + Stage 3 decoder pair so the demo eval can run against the
"intended" model — a falsifiable test of whether the encoder drift is the
dominant failure mode.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.build_frankenstein_checkpoint \\
        --stage2-ckpt checkpoints\\full_radio_stage2_systems_v2\\stage2-radio-systems-polyphonic_best.pt \\
        --stage3-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --out checkpoints\\_frankenstein_s2enc_s3dec.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def _get_state_dict(checkpoint):
    """Find the model state_dict inside a checkpoint payload.

    Trainer saves often nest: {"model": <state>, "step": N, ...}. Inference
    saves often: {"state_dict": <state>, ...}. Both formats are supported.
    Returns the state_dict and a dict of any non-state metadata for logging.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint, {}
    for key in ("model", "state_dict", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            metadata = {k: v for k, v in checkpoint.items() if k != key and not isinstance(v, dict)}
            return checkpoint[key], metadata
    # Top-level is already a state_dict
    return checkpoint, {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-ckpt", type=Path, required=True,
                   help="Stage 2 v2 best.pt — source of the encoder weights that produced the cache")
    p.add_argument("--stage3-ckpt", type=Path, required=True,
                   help="Stage 3 v2 best.pt — source of the trained decoder + bridge weights")
    p.add_argument("--out", type=Path, required=True,
                   help="Output frankenstein checkpoint path")
    p.add_argument("--encoder-key-marker", type=str, default="encoder",
                   help="Substring identifying encoder params (default: 'encoder')")
    args = p.parse_args()

    import torch

    s2_raw = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    s3_raw = torch.load(args.stage3_ckpt, map_location="cpu", weights_only=False)

    s2_sd, s2_meta = _get_state_dict(s2_raw)
    s3_sd, s3_meta = _get_state_dict(s3_raw)

    print(f"Stage 2 v2 keys: {len(s2_sd)}")
    print(f"Stage 3 v2 keys: {len(s3_sd)}")
    print(f"Stage 2 v2 metadata: {s2_meta}")
    print(f"Stage 3 v2 metadata: {s3_meta}")

    # Build merged state_dict starting from Stage 3 v2 (which has the trained
    # decoder), then overlay Stage 2 v2's encoder params.
    merged = dict(s3_sd)
    n_from_s2 = 0
    n_missing_in_s2 = 0
    n_kept_from_s3 = 0
    only_in_s2 = []
    only_in_s3 = []
    for k in list(s3_sd.keys()):
        if args.encoder_key_marker in k:
            if k in s2_sd:
                merged[k] = s2_sd[k]
                n_from_s2 += 1
            else:
                # Encoder key in S3 but not S2 — keep S3's version, but flag it.
                # Probably means the encoder grew new parameters between S2 and S3
                # (unlikely but possible if architecture changed).
                n_missing_in_s2 += 1
        else:
            n_kept_from_s3 += 1
    # Keys only in S2 that aren't in S3 — for visibility
    for k in s2_sd:
        if args.encoder_key_marker in k and k not in s3_sd:
            only_in_s2.append(k)
    # Keys only in S3 that aren't in S2 (and aren't encoder) — also for visibility
    for k in s3_sd:
        if args.encoder_key_marker not in k and k not in s2_sd:
            only_in_s3.append(k)

    print()
    print(f"Merge result:")
    print(f"  encoder keys taken from S2: {n_from_s2}")
    print(f"  encoder keys missing in S2 (kept S3 version): {n_missing_in_s2}")
    print(f"  non-encoder keys taken from S3: {n_kept_from_s3}")
    print(f"  encoder keys ONLY in S2 (dropped): {len(only_in_s2)}")
    print(f"  non-encoder keys ONLY in S3 (kept): {len(only_in_s3)}")
    if only_in_s2[:5]:
        print(f"  first 5 S2-only encoder keys: {only_in_s2[:5]}")
    if only_in_s3[:5]:
        print(f"  first 5 S3-only non-encoder keys: {only_in_s3[:5]}")

    # Safety check: if more than 1% of encoder keys mismatch, the experiment
    # may be confounded — escalate before drawing a conclusion.
    n_encoder_keys = n_from_s2 + n_missing_in_s2
    if n_encoder_keys > 0:
        mismatch_pct = 100 * (n_missing_in_s2 + len(only_in_s2)) / n_encoder_keys
        print(f"  encoder-key mismatch rate: {mismatch_pct:.2f}%")
        if mismatch_pct > 1.0:
            print(f"  WARNING: encoder-key mismatch rate exceeds 1%. Diagnostic result may be confounded.")

    # Wrap in the inference-friendly format (top-level state_dict) so the
    # eval driver can load it without special handling.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": merged,
        "note": "frankenstein: stage2_v2_encoder + stage3_v2_decoder",
        "source_stage2": str(args.stage2_ckpt),
        "source_stage3": str(args.stage3_ckpt),
        "n_encoder_keys_from_s2": n_from_s2,
        "n_encoder_keys_missing_in_s2": n_missing_in_s2,
    }
    # Preserve any non-model fields from the S3 payload (DoRA config, vocab, etc.).
    if isinstance(s3_raw, dict):
        for k, v in s3_raw.items():
            if k not in payload and k not in ("model", "state_dict", "model_state_dict"):
                payload[k] = v
    torch.save(payload, args.out)
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
