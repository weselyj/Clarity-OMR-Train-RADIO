#!/usr/bin/env python3
"""Auto-tune beam search penalty weights for a given checkpoint.

Runs a small validation set with many penalty configurations, evaluates
each, and picks the weights that maximize the quality score. Saves the
best config as a JSON file alongside the checkpoint.

Usage:
    python src/eval/tune_penalties.py \
        --checkpoint src/train/checkpoints/stage2-polyphonic-finetune_step_001500.pt \
        --max-samples 30 \
        --trials 40
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import aggregate_metrics, quality_score


# ---------------------------------------------------------------------------
# Penalty configuration
# ---------------------------------------------------------------------------

@dataclass
class PenaltyConfig:
    """All tuneable penalty hyperparameters."""

    # CV note count penalty
    cv_count_tolerance: int = 2
    cv_count_weight: float = 3.0

    # CV pitch prior penalty
    cv_pitch_weight: float = 1.5
    cv_pitch_octave_weight: float = 3.0

    # Built-in penalties
    pitch_range_enabled: bool = True
    accidental_consistency_enabled: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, object]) -> PenaltyConfig:
        return PenaltyConfig(**{k: v for k, v in d.items() if k in PenaltyConfig.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# Grid values for each parameter
GRID = {
    "cv_count_tolerance": [1, 2, 3, 4],
    "cv_count_weight": [0.0, 1.5, 3.0, 5.0],
    "cv_pitch_weight": [0.0, 0.5, 1.0, 1.5, 2.5],
    "cv_pitch_octave_weight": [0.0, 1.5, 3.0, 5.0],
}


def _sample_configs(n_trials: int, seed: int) -> List[PenaltyConfig]:
    """Sample penalty configs: always include baseline + no-CV, then random."""
    configs: List[PenaltyConfig] = []

    # Always include the default config
    configs.append(PenaltyConfig())

    # No CV priors at all (ablation baseline)
    configs.append(PenaltyConfig(
        cv_count_weight=0.0,
        cv_pitch_weight=0.0,
        cv_pitch_octave_weight=0.0,
    ))

    # Count-only (no pitch prior)
    configs.append(PenaltyConfig(
        cv_pitch_weight=0.0,
        cv_pitch_octave_weight=0.0,
    ))

    # Pitch-only (no count prior)
    configs.append(PenaltyConfig(
        cv_count_weight=0.0,
    ))

    # Full grid is too large — use random sampling
    rng = random.Random(seed)
    remaining = max(0, n_trials - len(configs))
    for _ in range(remaining):
        configs.append(PenaltyConfig(
            cv_count_tolerance=rng.choice(GRID["cv_count_tolerance"]),
            cv_count_weight=rng.choice(GRID["cv_count_weight"]),
            cv_pitch_weight=rng.choice(GRID["cv_pitch_weight"]),
            cv_pitch_octave_weight=rng.choice(GRID["cv_pitch_octave_weight"]),
        ))

    return configs


# ---------------------------------------------------------------------------
# Inference with a specific penalty config
# ---------------------------------------------------------------------------

def _run_inference_with_config(
    *,
    model,
    vocab,
    crop_rows: List[Dict[str, object]],
    config: PenaltyConfig,
    beam_width: int,
    max_decode_steps: int,
    length_penalty_alpha: float,
    use_kv_cache: bool,
    image_height: int,
    image_max_width: int,
    device,
    project_root: Path,
) -> List[Dict[str, object]]:
    """Run inference on all crops with a specific penalty config."""
    from src.inference.decoder_runtime import _decode_stage_b_tokens, _load_stage_b_crop_tensor

    results: List[Dict[str, object]] = []
    for row in crop_rows:
        crop_path = Path(str(row["crop_path"]))
        if not crop_path.is_absolute():
            crop_path = (project_root / crop_path).resolve()

        pixel_values = _load_stage_b_crop_tensor(
            crop_path,
            image_height=image_height,
            image_max_width=image_max_width,
            device=device,
        )

        # Build CV priors from row data
        cv_note_count = None
        cv_pitches = None
        raw_count = row.get("cv_note_count")
        if raw_count is not None and config.cv_count_weight > 0:
            cv_note_count = int(raw_count)
        raw_pitches = row.get("cv_pitches")
        if isinstance(raw_pitches, list) and config.cv_pitch_weight > 0:
            cv_pitches = raw_pitches

        # Build custom penalty function with this config's weights
        from src.decoding.beam_search import (
            make_cv_penalty_fn,
            pitch_range_penalty,
            accidental_consistency_penalty,
            cv_note_count_penalty,
            cv_pitch_prior_penalty,
        )

        def _make_penalty(cfg: PenaltyConfig, nc=cv_note_count, cp=cv_pitches):
            def _penalty(prefix: Sequence[str], candidate: str) -> float:
                p = 0.0
                if cfg.pitch_range_enabled:
                    p += pitch_range_penalty(prefix, candidate)
                if cfg.accidental_consistency_enabled:
                    p += accidental_consistency_penalty(prefix, candidate)
                if nc is not None:
                    p += cv_note_count_penalty(
                        prefix, candidate,
                        cv_note_count=nc,
                        tolerance=cfg.cv_count_tolerance,
                        penalty_weight=cfg.cv_count_weight,
                    )
                if cp is not None:
                    p += cv_pitch_prior_penalty(
                        prefix, candidate,
                        cv_pitches=cp,
                        penalty_weight=cfg.cv_pitch_weight,
                        octave_penalty_weight=cfg.cv_pitch_octave_weight,
                    )
                return p
            return _penalty

        # We need to call _decode_stage_b_tokens but with our custom penalty.
        # Instead, replicate the core logic directly for flexibility.
        tokens = _decode_with_custom_penalty(
            model=model,
            pixel_values=pixel_values,
            vocabulary=vocab,
            beam_width=beam_width,
            max_decode_steps=max_decode_steps,
            length_penalty_alpha=length_penalty_alpha,
            use_kv_cache=use_kv_cache,
            penalty_fn=_make_penalty(config),
        )

        results.append({
            "sample_id": str(row.get("sample_id", "")),
            "pred_tokens": tokens,
            "gt_tokens": row.get("gt_tokens", []),
        })

    return results


def _decode_with_custom_penalty(
    *,
    model,
    pixel_values,
    vocabulary,
    beam_width: int,
    max_decode_steps: int,
    length_penalty_alpha: float,
    use_kv_cache: bool,
    penalty_fn,
) -> List[str]:
    """Decode tokens using a custom penalty function."""
    import torch

    from src.decoding.beam_search import BeamSearchConfig, constrained_beam_search_with_state

    from src.inference.decoder_runtime import _LazyLogitDict, _encode_staff_image, _resolve_stage_b_decode_model

    decode_model = _resolve_stage_b_decode_model(model)
    memory = _encode_staff_image(decode_model, pixel_values)
    _token_to_idx = {token: idx for idx, token in enumerate(vocabulary.tokens)}

    def _step_fn(prefix_tokens, parent_cache):
        if not use_kv_cache or parent_cache is None:
            token_ids = vocabulary.encode(prefix_tokens, strict=True)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=pixel_values.device)
            layer_cache = None
        else:
            token_ids = vocabulary.encode([prefix_tokens[-1]], strict=True)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=pixel_values.device)
            layer_cache = parent_cache

        with torch.inference_mode():
            logits, _, next_cache = decode_model.decode_tokens(
                input_ids, memory,
                past_key_values=layer_cache,
                use_cache=bool(use_kv_cache),
            )
        next_log_probs = torch.log_softmax(logits[0, -1], dim=-1).float().cpu()
        distribution = _LazyLogitDict(next_log_probs, _token_to_idx)
        return distribution, (next_cache if use_kv_cache else None)

    beams = constrained_beam_search_with_state(
        step_fn=_step_fn,
        vocabulary=vocabulary,
        config=BeamSearchConfig(
            beam_width=beam_width,
            max_steps=max_decode_steps,
            length_penalty_alpha=length_penalty_alpha,
        ),
        soft_penalty_fn=penalty_fn,
        prefix_tokens=["<bos>", "<staff_start>"],
    )
    if not beams:
        return ["<bos>", "<staff_start>", "<staff_end>", "<eos>"]
    predicted = list(beams[0].tokens)
    if not predicted or predicted[-1] != "<eos>":
        predicted.append("<eos>")
    return predicted


# ---------------------------------------------------------------------------
# CV prior generation for val samples
# ---------------------------------------------------------------------------

def _generate_cv_priors(
    crop_rows: List[Dict[str, object]],
    project_root: Path,
    quiet: bool = False,
) -> List[Dict[str, object]]:
    """Run CV analysis on each crop row and inject cv_note_count + cv_pitches."""
    try:
        from src.cv.staff_analyzer import analyze_staff
    except ImportError:
        if not quiet:
            print("[tune] CV module not available, skipping priors", file=sys.stderr)
        return crop_rows

    if not quiet:
        print(f"[tune] Running CV analysis on {len(crop_rows)} samples...", file=sys.stderr, flush=True)

    for row in crop_rows:
        crop_path = Path(str(row["crop_path"]))
        if not crop_path.is_absolute():
            crop_path = (project_root / crop_path).resolve()
        try:
            skeleton = analyze_staff(str(crop_path))
            row["cv_note_count"] = skeleton.total_note_count
            row["cv_pitches"] = [
                nh.estimated_pitch
                for nh in skeleton.noteheads
                if nh.estimated_pitch is not None
            ]
        except Exception:
            pass  # Non-fatal

    if not quiet:
        with_count = sum(1 for r in crop_rows if r.get("cv_note_count", 0) > 0)
        with_pitch = sum(1 for r in crop_rows if len(r.get("cv_pitches", [])) > 0)
        print(f"[tune] CV priors: {with_count} counts, {with_pitch} pitch lists", file=sys.stderr, flush=True)

    return crop_rows


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def tune(
    *,
    checkpoint: Path,
    project_root: Path,
    token_manifest_paths: List[Path],
    split: str = "val",
    max_samples: int = 30,
    n_trials: int = 40,
    beam_width: int = 3,
    max_decode_steps: int = 384,
    length_penalty_alpha: float = 0.6,
    use_kv_cache: bool = False,
    image_height: int = 250,
    image_max_width: int = 2500,
    device_name: Optional[str] = None,
    seed: int = 42,
    output_path: Optional[Path] = None,
    quiet: bool = False,
) -> Dict[str, object]:
    """Run the full tuning loop and return the best config."""
    import torch

    from src.tokenizer.vocab import build_default_vocabulary
    from src.train.model_factory import (
        ModelFactoryConfig,
        build_stage_b_components,
        model_factory_config_from_checkpoint_payload,
    )
    from src.checkpoint_io import load_stage_b_checkpoint

    # --- Load model ---
    device_str = str(device_name).strip() if device_name else ""
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = build_default_vocabulary()
    payload = torch.load(str(checkpoint), map_location=device)
    fallback_cfg = ModelFactoryConfig(stage_b_vocab_size=vocab.size)
    factory_cfg = model_factory_config_from_checkpoint_payload(payload, vocab_size=vocab.size, fallback=fallback_cfg)
    components = build_stage_b_components(factory_cfg)
    model = components["model"]
    ckpt_result = load_stage_b_checkpoint(
        checkpoint_path=checkpoint,
        model=model,
        device=device,
        dora_config=components.get("dora_config"),
    )
    model = ckpt_result["_model"]
    model.eval()

    if not quiet:
        print(f"[tune] Model loaded on {device}", file=sys.stderr, flush=True)

    # --- Build val set ---
    from src.eval.evaluate_stage_b_checkpoint import _read_jsonl, _build_crops_manifest_rows

    token_rows: List[Dict[str, object]] = []
    for manifest_path in token_manifest_paths:
        token_rows.extend(_read_jsonl(manifest_path))

    crop_rows = _build_crops_manifest_rows(
        token_rows, split=split, max_samples=max_samples, seed=seed,
    )
    if not crop_rows:
        raise ValueError(f"No samples found for split='{split}'")

    if not quiet:
        print(f"[tune] {len(crop_rows)} val samples", file=sys.stderr, flush=True)

    # --- Generate CV priors ---
    crop_rows = _generate_cv_priors(crop_rows, project_root, quiet=quiet)

    # --- Generate penalty configs ---
    configs = _sample_configs(n_trials, seed)
    if not quiet:
        print(f"[tune] Testing {len(configs)} penalty configurations...", file=sys.stderr, flush=True)

    # --- Run all trials ---
    results: List[Dict[str, object]] = []
    best_score = -1.0
    best_config = configs[0]
    best_idx = 0

    for trial_idx, config in enumerate(configs):
        trial_start = time.time()

        trial_results = _run_inference_with_config(
            model=model,
            vocab=vocab,
            crop_rows=crop_rows,
            config=config,
            beam_width=beam_width,
            max_decode_steps=max_decode_steps,
            length_penalty_alpha=length_penalty_alpha,
            use_kv_cache=use_kv_cache,
            image_height=image_height,
            image_max_width=image_max_width,
            device=device,
            project_root=project_root,
        )

        # Evaluate
        eval_pairs = [
            (r["pred_tokens"], r["gt_tokens"])
            for r in trial_results
            if r.get("gt_tokens")
        ]
        if not eval_pairs:
            continue

        metrics = aggregate_metrics(eval_pairs)
        q = metrics.get("quality", {})
        score = float(q.get("score", 0.0))

        trial_seconds = time.time() - trial_start

        trial_entry = {
            "trial_index": trial_idx,
            "config": config.to_dict(),
            "quality_score": score,
            "quality_rating": q.get("rating", "unknown"),
            "base_score": float(q.get("base_score", 0.0)),
            "penalty_factor": float(q.get("penalty_factor", 1.0)),
            "ser": float(metrics.get("ser", 0.0)),
            "note_event_f1": float(metrics.get("note_event_f1", 0.0)),
            "pitch_accuracy": float(metrics.get("pitch_accuracy", 0.0)),
            "measure_balance_rate": float(metrics.get("measure_balance_rate", 0.0)),
            "seconds": trial_seconds,
        }
        results.append(trial_entry)

        if score > best_score:
            best_score = score
            best_config = config
            best_idx = trial_idx

        if not quiet:
            marker = " ***BEST***" if trial_idx == best_idx else ""
            print(
                f"[tune] trial {trial_idx+1}/{len(configs)}: "
                f"quality={score:.1f} ({q.get('rating', '?')}) "
                f"note_f1={metrics.get('note_event_f1', 0):.3f} "
                f"balance={metrics.get('measure_balance_rate', 0):.3f} "
                f"({trial_seconds:.1f}s){marker}",
                file=sys.stderr,
                flush=True,
            )

    # --- Sort results ---
    results.sort(key=lambda r: r["quality_score"], reverse=True)

    summary = {
        "checkpoint": str(checkpoint),
        "split": split,
        "samples": len(crop_rows),
        "trials": len(results),
        "best_trial_index": best_idx,
        "best_quality_score": best_score,
        "best_config": best_config.to_dict(),
        "top_5": results[:5],
        "all_trials": results,
    }

    # --- Save ---
    if output_path is None:
        output_path = checkpoint.parent / f"{checkpoint.stem}_penalty_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not quiet:
        print(f"\n[tune] Best config (quality={best_score:.1f}):", file=sys.stderr)
        print(f"  {json.dumps(best_config.to_dict(), indent=2)}", file=sys.stderr)
        print(f"[tune] Saved to {output_path}", file=sys.stderr)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Auto-tune beam search penalty weights for a checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-B checkpoint (.pt)")
    parser.add_argument(
        "--token-manifest", type=str,
        default=",".join([
            str(project_root / "src" / "data" / "manifests" / "token_manifest.jsonl"),
            str(project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_token_manifest.jsonl"),
        ]),
        help="Comma-separated token manifest paths.",
    )
    parser.add_argument("--split", type=str, default="val", help="Manifest split (default: val)")
    parser.add_argument("--max-samples", type=int, default=30, help="Val samples to evaluate per trial")
    parser.add_argument("--trials", type=int, default=40, help="Number of penalty configs to try")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width (lower = faster tuning)")
    parser.add_argument("--max-decode-steps", type=int, default=384, help="Max decode steps")
    parser.add_argument("--length-penalty-alpha", type=float, default=0.6, help="Length penalty alpha")
    parser.add_argument("--image-height", type=int, default=250, help="Input image height")
    parser.add_argument("--image-max-width", type=int, default=2500, help="Input image max width")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    parser.add_argument("--project-root", type=Path, default=project_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = (project_root / checkpoint).resolve()

    manifest_paths = []
    for item in args.token_manifest.split(","):
        p = Path(item.strip())
        if not p.is_absolute():
            p = (project_root / p).resolve()
        manifest_paths.append(p)

    summary = tune(
        checkpoint=checkpoint,
        project_root=project_root,
        token_manifest_paths=manifest_paths,
        split=args.split,
        max_samples=args.max_samples,
        n_trials=args.trials,
        beam_width=args.beam_width,
        max_decode_steps=args.max_decode_steps,
        length_penalty_alpha=args.length_penalty_alpha,
        image_height=args.image_height,
        image_max_width=args.image_max_width,
        device_name=args.device,
        seed=args.seed,
        output_path=args.output,
        quiet=args.quiet,
    )

    print(json.dumps({
        "best_quality_score": summary["best_quality_score"],
        "best_config": summary["best_config"],
        "trials_run": summary["trials"],
        "output": str(args.output or ""),
    }, indent=2))


if __name__ == "__main__":
    main()
