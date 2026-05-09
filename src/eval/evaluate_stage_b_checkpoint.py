#!/usr/bin/env python3
"""Evaluate a Stage-B checkpoint directly from a token manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import default_ablation_matrix
from src.eval.run_eval import evaluate_rows

# Datasets whose encoder features may be present in the encoder cache.
# Mirrors _CACHED_DATASETS in src/train/train.py.
_CACHED_DATASETS = frozenset(
    {"synthetic_systems", "grandstaff_systems", "primus_systems"}
)


def _resolve_path(project_root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value.resolve()
    return (project_root / value).resolve()


def _resolve_manifest_paths(project_root: Path, manifest_arg: str) -> List[Path]:
    resolved: List[Path] = []
    for item in manifest_arg.split(","):
        raw = item.strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = project_root / path
        resolved.append(path.resolve())
    return resolved


def _assert_not_stale_merged_manifest(project_root: Path, manifest_paths: Sequence[Path]) -> None:
    if len(manifest_paths) != 1:
        return
    merged_path = manifest_paths[0]
    if merged_path.name.lower() != "token_manifest_train.jsonl":
        return
    if not merged_path.exists():
        return

    base_manifest = (project_root / "src" / "data" / "manifests" / "token_manifest.jsonl").resolve()
    synthetic_manifest = (
        project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_token_manifest.jsonl"
    ).resolve()
    dependencies = [path for path in (base_manifest, synthetic_manifest) if path.exists()]
    if len(dependencies) < 2:
        return

    merged_mtime = merged_path.stat().st_mtime
    newer_dependencies = [path for path in dependencies if path.stat().st_mtime > (merged_mtime + 1.0)]
    if not newer_dependencies:
        return

    newer_paths = ", ".join(str(path) for path in newer_dependencies)
    raise RuntimeError(
        "Detected stale merged token manifest "
        f"'{merged_path}'. Newer dependency manifest(s): {newer_paths}. "
        "Regenerate token_manifest_train.jsonl or pass comma-separated manifests directly "
        "(src/data/manifests/token_manifest.jsonl,data/processed/synthetic/manifests/synthetic_token_manifest.jsonl)."
    )


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _coerce_tokens(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(token) for token in value]
    if isinstance(value, str):
        return [token for token in value.replace("\t", " ").split(" ") if token]
    return []


def _build_crops_manifest_rows(
    token_rows: Sequence[Dict[str, object]],
    *,
    split: str,
    max_samples: int | None,
    seed: int,
) -> List[Dict[str, object]]:
    split_clean = split.strip().lower()
    rows: List[Dict[str, object]] = []
    for row in token_rows:
        row_split = str(row.get("split", "train")).strip().lower()
        if row_split != split_clean:
            continue

        image_path = row.get("image_path")
        gt_tokens = _coerce_tokens(row.get("token_sequence"))
        if not image_path or not gt_tokens:
            continue

        rows.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "dataset": str(row.get("dataset", "unknown")).lower(),
                "crop_path": str(image_path),
                "gt_tokens": gt_tokens,
                "reference_image_path": str(image_path),
            }
        )

    if max_samples is not None and max_samples > 0 and len(rows) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]

    return rows


def _build_eval_rows(prediction_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in prediction_rows:
        pred_tokens = _coerce_tokens(row.get("tokens"))
        gt_tokens = _coerce_tokens(row.get("gt_tokens"))
        if not pred_tokens or not gt_tokens:
            continue
        rows.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "dataset": str(row.get("dataset", "unknown")).lower(),
                "pred_tokens": pred_tokens,
                "gt_tokens": gt_tokens,
                "image_path": str(row.get("crop_path", "")),
                "reference_image_path": str(row.get("reference_image_path", row.get("crop_path", ""))),
            }
        )
    return rows


def _resolve_cache_memory(
    *,
    decode_model,
    pixel_values,
    dataset: str,
    sample_id: str,
    cache_root: Optional[Path],
    cache_hash: Optional[str],
    device,
    use_fp16: bool,
):
    """Return encoder memory for one sample, using the cache when available.

    Lookup order:
      1. If ``cache_root`` is provided AND ``dataset`` is in ``_CACHED_DATASETS``:
         a. Try the new reversible key scheme (_sanitize_sample_key).
         b. On ``FileNotFoundError``, fall back to the legacy ``__`` scheme
            (_sanitize_sample_key_legacy) for caches built before 2026-05-09.
         c. On a second ``FileNotFoundError`` (both schemes miss), fall through
            to the live encoder path so a partial cache never blocks eval.
      2. Otherwise: run ``_encode_staff_image`` (live encoder forward pass).

    Only ``FileNotFoundError`` is caught for the fallback chain.  Other errors
    (e.g. ``OSError``, corrupted .pt file) propagate immediately so cache
    corruption surfaces as a failure rather than a silent accuracy regression.

    Returns:
        memory tensor — same semantics as ``_encode_staff_image`` return value,
        moved to ``device`` and cast to fp16 if ``use_fp16`` is True.
    """
    from src.cli import _encode_staff_image

    if cache_root is not None and dataset in _CACHED_DATASETS:
        from src.data.encoder_cache import (
            _sanitize_sample_key,
            _sanitize_sample_key_legacy,
            read_cache_entry,
        )
        new_key = _sanitize_sample_key(sample_id)
        try:
            tensor, _h16, _w16 = read_cache_entry(
                cache_root, cache_hash, dataset, new_key
            )
            memory = tensor.to(device=device)
            if use_fp16:
                memory = memory.half()
            return memory
        except FileNotFoundError:
            pass  # try legacy key

        legacy_key = _sanitize_sample_key_legacy(sample_id)
        try:
            tensor, _h16, _w16 = read_cache_entry(
                cache_root, cache_hash, dataset, legacy_key
            )
            memory = tensor.to(device=device)
            if use_fp16:
                memory = memory.half()
            return memory
        except FileNotFoundError:
            pass  # both schemes missed; fall through to live encoder

    # Live path: full encoder forward pass
    return _encode_staff_image(decode_model, pixel_values)


def _run_stage_b_inference_with_progress(
    *,
    project_root: Path,
    crops_manifest: Path,
    output_predictions: Path,
    checkpoint: Path,
    beam_width: int,
    max_decode_steps: int,
    image_height: int,
    image_max_width: int,
    device_name: str | None,
    progress_every_seconds: float,
    quiet: bool,
    length_penalty_alpha: float = 0.6,
    use_kv_cache: bool = True,
    use_fp16: bool = False,
    quantize: bool = False,
    encoder_cache_root: Optional[Path] = None,
    encoder_cache_hash: Optional[str] = None,
) -> Dict[str, object]:
    import torch

    from src.cli import (
        _decode_stage_b_tokens,
        _encode_staff_image,
        _load_stage_b_crop_tensor,
        _prepare_model_for_inference,
    )
    from src.checkpoint_io import load_stage_b_checkpoint
    from src.tokenizer.vocab import build_default_vocabulary
    from src.train.model_factory import (
        ModelFactoryConfig,
        build_stage_b_components,
        model_factory_config_from_checkpoint_payload,
    )

    # Resolve encoder cache root once (may be None if not provided via CLI)
    _cache_root: Optional[Path] = (
        Path(encoder_cache_root).resolve() if encoder_cache_root is not None else None
    )
    _cache_hash: Optional[str] = encoder_cache_hash or None

    crop_rows = _read_jsonl(crops_manifest)
    if not crop_rows:
        raise ValueError(f"No crop rows found in {crops_manifest}")

    cleaned_device = str(device_name).strip() if device_name else ""
    device = torch.device(cleaned_device if cleaned_device else ("cuda" if torch.cuda.is_available() else "cpu"))

    vocab = build_default_vocabulary()
    payload = torch.load(str(checkpoint), map_location=device)
    fallback_factory_cfg = ModelFactoryConfig(stage_b_vocab_size=vocab.size)
    factory_cfg = model_factory_config_from_checkpoint_payload(
        payload,
        vocab_size=vocab.size,
        fallback=fallback_factory_cfg,
    )
    components = build_stage_b_components(factory_cfg)
    model = components["model"]
    ckpt_result = load_stage_b_checkpoint(
        checkpoint_path=checkpoint,
        model=model,
        device=device,
        dora_config=components.get("dora_config"),
        min_coverage=0.50,
    )
    model = ckpt_result["_model"]
    checkpoint_format = ckpt_result["checkpoint_format"]
    loaded_keys = ckpt_result["loaded_keys"]
    load_ratio = ckpt_result["load_ratio"]
    model.eval()

    # Prepare model once for all crops
    decode_model, use_fp16 = _prepare_model_for_inference(model, device, use_fp16=use_fp16, quantize=quantize)
    _token_to_idx = {token: idx for idx, token in enumerate(vocab.tokens)}

    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    total = len(crop_rows)
    started = time.time()
    interval_seconds = max(0.2, float(progress_every_seconds))
    next_log_at = started + interval_seconds
    if not quiet:
        print(
            f"[stage-b-eval] starting inference on {total} samples "
            f"(device={device}, beam={beam_width}, max_steps={max_decode_steps}, max_width={image_max_width}, kv_cache={use_kv_cache})",
            file=sys.stderr,
            flush=True,
        )

    with output_predictions.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(crop_rows, start=1):
            crop_raw = row.get("crop_path")
            if not crop_raw:
                raise ValueError(f"Crop row missing crop_path: {row}")
            crop_path = Path(str(crop_raw))
            if not crop_path.is_absolute():
                crop_path = (project_root / crop_path).resolve()
            if not crop_path.exists():
                raise FileNotFoundError(f"Crop image not found: {crop_path}")
            if (not quiet) and index == 1:
                print(
                    f"[stage-b-eval] running first sample ({index}/{total}): {crop_path.name}",
                    file=sys.stderr,
                    flush=True,
                )

            pixel_values = _load_stage_b_crop_tensor(
                crop_path,
                image_height=max(32, int(image_height)),
                image_max_width=max(256, int(image_max_width)),
                device=device,
            )
            if use_fp16:
                pixel_values = pixel_values.half()
            memory = _resolve_cache_memory(
                decode_model=decode_model,
                pixel_values=pixel_values,
                dataset=str(row.get("dataset", "")).lower(),
                sample_id=str(row.get("sample_id", "")),
                cache_root=_cache_root,
                cache_hash=_cache_hash,
                device=device,
                use_fp16=use_fp16,
            )

            tokens = _decode_stage_b_tokens(
                model=model,
                pixel_values=pixel_values,
                vocabulary=vocab,
                beam_width=max(1, int(beam_width)),
                max_decode_steps=max(8, int(max_decode_steps)),
                length_penalty_alpha=float(length_penalty_alpha),
                use_kv_cache=bool(use_kv_cache),
                _precomputed={
                    "decode_model": decode_model,
                    "memory": memory,
                    "token_to_idx": _token_to_idx,
                    "use_fp16": use_fp16,
                },
            )
            row_with_tokens = dict(row)
            row_with_tokens["tokens"] = tokens
            handle.write(json.dumps(row_with_tokens) + "\n")

            now = time.time()
            should_log = (index == 1) or (index == total) or (now >= next_log_at)
            if (not quiet) and should_log:
                elapsed = max(1e-9, now - started)
                rate = float(index) / elapsed
                remaining = max(0, total - index)
                eta_seconds = float(remaining) / max(rate, 1e-9)
                print(
                    f"[stage-b-eval] {index}/{total} ({(100.0 * index / total):.1f}%) "
                    f"| {rate:.2f} samples/s | elapsed {elapsed / 60.0:.1f}m | eta {eta_seconds / 60.0:.1f}m",
                    file=sys.stderr,
                    flush=True,
                )
                next_log_at = now + interval_seconds

    total_seconds = max(1e-9, time.time() - started)
    return {
        "input_crops": total,
        "predictions_written": total,
        "output_predictions": str(output_predictions),
        "checkpoint": str(checkpoint),
        "checkpoint_format": checkpoint_format,
        "device": str(device),
        "missing_keys": ckpt_result["missing_keys"],
        "unexpected_keys": ckpt_result["unexpected_keys"],
        "missing_key_sample": ckpt_result["missing_key_sample"],
        "unexpected_key_sample": ckpt_result["unexpected_key_sample"],
        "loaded_keys": loaded_keys,
        "load_ratio": load_ratio,
        "inference_seconds": float(total_seconds),
        "samples_per_second": float(total) / float(total_seconds),
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate a Stage-B checkpoint on a token-manifest split.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Repository root path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Stage-B checkpoint path (.pt).",
    )
    parser.add_argument(
        "--token-manifest",
        type=str,
        default=",".join(
            [
                str(project_root / "src" / "data" / "manifests" / "token_manifest.jsonl"),
                str(project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_token_manifest.jsonl"),
            ]
        ),
        help="Comma-separated token manifest JSONL path(s).",
    )
    parser.add_argument(
        "--allow-stale-merged-manifest",
        action="store_true",
        help="Allow using token_manifest_train.jsonl even if base/synthetic manifests are newer.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Manifest split to evaluate (default: val).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on evaluated samples (0 means all).",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for max-samples shuffling.")
    parser.add_argument("--beam-width", type=int, default=8, help="Constrained beam width.")
    parser.add_argument("--max-decode-steps", type=int, default=512, help="Maximum decode steps.")
    parser.add_argument("--length-penalty-alpha", type=float, default=0.6, help="Length normalization alpha (0=disabled).")
    parser.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KV-cache decoding for speed (default: enabled).",
    )
    parser.add_argument("--image-height", type=int, default=250, help="Stage-B input image height.")
    parser.add_argument("--image-max-width", type=int, default=2500, help="Stage-B input image max width (max 3000).")
    parser.add_argument("--device", type=str, default=None, help="Inference device (e.g. cuda, cpu).")
    parser.add_argument("--quantize", action="store_true", help="INT8 dynamic quantization on decoder (CPU: 2-3x faster, GPU: needs torchao).")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Speed-oriented eval profile: beam=1, max_decode_steps<=192, image_max_width<=1024.",
    )
    parser.add_argument(
        "--progress-every-seconds",
        type=float,
        default=10.0,
        help="Log Stage-B inference progress every N seconds.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logging during Stage-B inference.",
    )
    parser.add_argument(
        "--encoder-cache-root",
        type=Path,
        default=None,
        help=(
            "Root directory of the encoder feature cache "
            "(e.g. data/cache/encoder/). When provided together with "
            "--encoder-cache-hash, cached datasets skip the encoder forward pass "
            "for a ~10-15x throughput improvement."
        ),
    )
    parser.add_argument(
        "--encoder-cache-hash",
        type=str,
        default=None,
        help=(
            "16-character hex hash identifying the cache subdirectory "
            "(e.g. ac8948ae4b5be3e9). Required when --encoder-cache-root is set."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=project_root / "src" / "eval" / "checkpoint_eval",
        help="Directory for intermediate/output files.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional summary JSON output path.",
    )
    parser.add_argument(
        "--output-ablation-template",
        type=Path,
        default=None,
        help="Optional ablation template JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest_paths = _resolve_manifest_paths(project_root, str(args.token_manifest))
    if not args.allow_stale_merged_manifest:
        _assert_not_stale_merged_manifest(project_root, manifest_paths)
    checkpoint = _resolve_path(project_root, args.checkpoint)
    work_dir = _resolve_path(project_root, args.work_dir)

    for token_manifest in manifest_paths:
        if not token_manifest.exists():
            raise FileNotFoundError(f"Token manifest not found: {token_manifest}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    work_dir.mkdir(parents=True, exist_ok=True)
    crops_manifest_path = work_dir / "stageb_eval_crops_manifest.jsonl"
    raw_predictions_path = work_dir / "stageb_eval_predictions_raw.jsonl"
    eval_predictions_path = work_dir / "stageb_eval_predictions_eval.jsonl"
    summary_path = _resolve_path(project_root, args.output_summary) if args.output_summary else (work_dir / "stageb_eval_summary.json")
    ablation_path = (
        _resolve_path(project_root, args.output_ablation_template)
        if args.output_ablation_template
        else (work_dir / "stageb_ablation_template.json")
    )

    token_rows: List[Dict[str, object]] = []
    for token_manifest in manifest_paths:
        token_rows.extend(_read_jsonl(token_manifest))
    crops_rows = _build_crops_manifest_rows(
        token_rows,
        split=args.split,
        max_samples=(None if int(args.max_samples) <= 0 else int(args.max_samples)),
        seed=int(args.seed),
    )
    if not crops_rows:
        raise ValueError(
            f"No usable samples found for split='{args.split}' in manifest inputs: "
            + ", ".join(str(path) for path in manifest_paths)
            + ". "
            "Rows need image_path and token_sequence."
        )
    _write_jsonl(crops_manifest_path, crops_rows)

    beam_width = max(1, int(args.beam_width))
    max_decode_steps = max(8, int(args.max_decode_steps))
    image_max_width = min(3000, max(256, int(args.image_max_width)))
    if bool(args.fast):
        beam_width = 1
        max_decode_steps = min(max_decode_steps, 192)
        image_max_width = min(image_max_width, 1024)
        print(
            "[stage-b-eval] fast profile enabled "
            f"(beam={beam_width}, max_steps={max_decode_steps}, max_width={image_max_width})",
            file=sys.stderr,
            flush=True,
        )

    stage_b_result = _run_stage_b_inference_with_progress(
        project_root=project_root,
        crops_manifest=crops_manifest_path,
        output_predictions=raw_predictions_path,
        checkpoint=checkpoint,
        beam_width=beam_width,
        max_decode_steps=max_decode_steps,
        image_height=max(32, int(args.image_height)),
        image_max_width=image_max_width,
        device_name=args.device,
        progress_every_seconds=max(0.2, float(args.progress_every_seconds)),
        quiet=bool(args.quiet),
        length_penalty_alpha=float(args.length_penalty_alpha),
        use_kv_cache=bool(args.kv_cache),
        quantize=bool(getattr(args, "quantize", False)),
        encoder_cache_root=getattr(args, "encoder_cache_root", None),
        encoder_cache_hash=getattr(args, "encoder_cache_hash", None),
    )

    raw_prediction_rows = _read_jsonl(raw_predictions_path)
    eval_rows = _build_eval_rows(raw_prediction_rows)
    if not eval_rows:
        raise ValueError(
            "No evaluation rows were generated from Stage-B predictions. "
            "Check that gt_tokens were present in the crops manifest."
        )
    _write_jsonl(eval_predictions_path, eval_rows)

    summary = evaluate_rows(eval_rows)
    summary["checkpoint"] = str(checkpoint)
    summary["token_manifest"] = ",".join(str(path) for path in manifest_paths)
    summary["split"] = str(args.split)
    summary["samples_requested"] = (None if int(args.max_samples) <= 0 else int(args.max_samples))
    summary["samples_used"] = len(eval_rows)
    summary["stage_b_inference"] = stage_b_result
    summary["outputs"] = {
        "crops_manifest": str(crops_manifest_path),
        "raw_predictions": str(raw_predictions_path),
        "eval_predictions": str(eval_predictions_path),
        "summary": str(summary_path),
        "ablation_template": str(ablation_path),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_text(json.dumps(default_ablation_matrix(), indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
