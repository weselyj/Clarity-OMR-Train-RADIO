"""A3: Decoder behavior on training data.

The model has been trained on these exact samples (with whatever
augmentation was active). At inference time, on the same images, the
predicted token sequence should closely match the ground-truth labels.

Pass criteria (triage thresholds, not certification):
  - token accuracy >= 80% averaged across samples
  - exact-match sequence rate >= 30%
  - per-class accuracy for time-sig tokens >= 80%
  - per-class accuracy for key-sig tokens >= 80%

Lower numbers indicate the model didn't memorize its training data
(unusual for a typical autoregressive transformer with enough capacity),
which would suggest either training didn't converge or there's a
preprocessing skew that A1/A2 didn't catch.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.a3_decoder_on_training \\
        --manifest src\\data\\manifests\\token_manifest_stage3.jsonl \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --n-per-corpus 5 \\
        --out audit_results\\a3_decoder.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def _token_accuracy(predicted: list, target: list) -> float:
    """Per-position accuracy over min(len(predicted), len(target)).

    Conservative: short predictions get 100% on their prefix; exact-match
    catches the length-mismatch case separately.
    """
    n = min(len(predicted), len(target))
    if n == 0:
        return 0.0
    correct = sum(1 for a, b in zip(predicted[:n], target[:n]) if a == b)
    return correct / n


def _per_class_accuracy(predicted: list, target: list, prefix: str) -> tuple[int, int]:
    """Returns (correct, total) for positions where the target token starts with prefix."""
    correct, total = 0, 0
    n = min(len(predicted), len(target))
    for i in range(n):
        if target[i].startswith(prefix):
            total += 1
            if predicted[i] == target[i]:
                correct += 1
    return correct, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--max-decode-steps", type=int, default=2048)
    p.add_argument("--n-per-corpus", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    from src.inference.checkpoint_load import load_stage_b_for_inference
    from src.inference.decoder_runtime import (
        _load_stage_b_crop_tensor,
        _encode_staff_image,
        _decode_stage_b_tokens,
    )
    from scripts.audit._sample_picker import pick_audit_samples

    device = torch.device("cuda")
    bundle = load_stage_b_for_inference(args.stage_b_ckpt, device, use_fp16=False)

    samples = pick_audit_samples(args.manifest, n_per_corpus=args.n_per_corpus, seed=args.seed)
    print(f"Selected {len(samples)} samples; running decoder on each...")

    per_sample = []
    for sample in samples:
        sample_id = sample["sample_id"]
        dataset = sample["dataset"]
        img_path = _REPO / sample["image_path"]
        target = sample.get("token_sequence", [])

        if not img_path.exists():
            per_sample.append({
                "sample_id": sample_id, "dataset": dataset,
                "status": "missing_image",
            })
            continue

        pixel_values = _load_stage_b_crop_tensor(
            img_path,
            image_height=args.image_height,
            image_max_width=args.image_max_width,
            device=device,
        )
        with torch.no_grad():
            memory = _encode_staff_image(bundle.decode_model, pixel_values)
            predicted = _decode_stage_b_tokens(
                model=bundle.model,
                pixel_values=pixel_values,
                vocabulary=bundle.vocab,
                beam_width=1,
                max_decode_steps=args.max_decode_steps,
                length_penalty_alpha=0.4,
                _precomputed={
                    "decode_model": bundle.decode_model,
                    "memory": memory,
                    "token_to_idx": bundle.token_to_idx,
                    "use_fp16": False,
                },
            )

        token_acc = _token_accuracy(predicted, target)
        exact_match = predicted == target
        ts_c, ts_n = _per_class_accuracy(predicted, target, "timeSignature-")
        ks_c, ks_n = _per_class_accuracy(predicted, target, "keySignature-")
        note_c, note_n = _per_class_accuracy(predicted, target, "note-")
        rest_c, rest_n = _per_class_accuracy(predicted, target, "rest")

        per_sample.append({
            "sample_id": sample_id, "dataset": dataset,
            "status": "compared",
            "predicted_len": len(predicted),
            "target_len": len(target),
            "token_accuracy": token_acc,
            "exact_match": exact_match,
            "timeSig_correct": ts_c, "timeSig_total": ts_n,
            "keySig_correct": ks_c, "keySig_total": ks_n,
            "note_correct": note_c, "note_total": note_n,
            "rest_correct": rest_c, "rest_total": rest_n,
            "predicted_first_50": predicted[:50],
            "target_first_50": target[:50],
        })

    compared = [r for r in per_sample if r["status"] == "compared"]
    n_compared = len(compared) or 1

    mean_token_acc = sum(r["token_accuracy"] for r in compared) / n_compared
    exact_match_rate = sum(1 for r in compared if r["exact_match"]) / n_compared

    def _ratio(c, n):
        return c / n if n else None

    ts_c_total = sum(r["timeSig_correct"] for r in compared)
    ts_n_total = sum(r["timeSig_total"] for r in compared)
    ks_c_total = sum(r["keySig_correct"] for r in compared)
    ks_n_total = sum(r["keySig_total"] for r in compared)
    note_c_total = sum(r["note_correct"] for r in compared)
    note_n_total = sum(r["note_total"] for r in compared)
    rest_c_total = sum(r["rest_correct"] for r in compared)
    rest_n_total = sum(r["rest_total"] for r in compared)

    results = {
        "experiment": "a3_decoder_on_training",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_samples": len(per_sample),
        "mean_token_accuracy": mean_token_acc,
        "exact_match_rate": exact_match_rate,
        "timeSig_accuracy": _ratio(ts_c_total, ts_n_total),
        "keySig_accuracy": _ratio(ks_c_total, ks_n_total),
        "note_accuracy": _ratio(note_c_total, note_n_total),
        "rest_accuracy": _ratio(rest_c_total, rest_n_total),
        "per_sample": per_sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"=== A3: Decoder on training data ===")
    print(f"Samples compared: {n_compared}")
    print(f"Mean token accuracy: {mean_token_acc:.3f}")
    print(f"Exact match rate:    {exact_match_rate:.3f}")
    print(f"timeSig accuracy:    {results['timeSig_accuracy']}")
    print(f"keySig accuracy:     {results['keySig_accuracy']}")
    print(f"note accuracy:       {results['note_accuracy']}")
    print(f"rest accuracy:       {results['rest_accuracy']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
