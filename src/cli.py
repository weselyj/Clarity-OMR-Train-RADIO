#!/usr/bin/env python3
"""End-to-end CLI for the OMR pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.yolo_stage_a import YoloStageA, YoloStageAConfig
from src.pipeline.assemble_score import (
    StaffLocation,
    StaffRecognitionResult,
    assemble_score,
    write_assembly_manifest,
)
from src.pipeline.export_musicxml import load_assembled_score, write_musicxml


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


def run_stage_a(args: argparse.Namespace) -> Dict[str, object]:
    stage_a = YoloStageA(
        YoloStageAConfig(
            weights_path=args.weights,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
        )
    )
    detections = stage_a.detect_regions(args.image)
    crops = stage_a.crop_staff_regions(args.image, detections, args.output_crops_dir)

    crop_rows = []
    for crop in crops:
        crop_rows.append(
            {
                "sample_id": f"{Path(crop.crop_path).stem}",
                "crop_path": crop.crop_path,
                "system_index": crop.system_index,
                "staff_index": crop.staff_index,
                "page_index": args.page_index,
                "bbox": {
                    "x_min": crop.bbox.x_min,
                    "y_min": crop.bbox.y_min,
                    "x_max": crop.bbox.x_max,
                    "y_max": crop.bbox.y_max,
                },
            }
        )
    _write_jsonl(args.output_manifest, crop_rows)
    return {
        "detections": len(detections),
        "staff_crops": len(crops),
        "output_manifest": str(args.output_manifest),
    }


def _load_stage_b_state_dict(checkpoint_path: Path, device) -> Dict[str, object]:
    import torch

    payload = torch.load(str(checkpoint_path), map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    normalized: Dict[str, object] = {}
    for name, tensor in state_dict.items():
        key = str(name)
        for prefix in ("base_model.model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        normalized[key] = tensor
    return normalized


def _load_stage_b_crop_tensor(
    crop_path: Path,
    *,
    image_height: int,
    image_max_width: int,
    device,
):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(crop_path) as image_obj:
        gray = image_obj.convert("L")
        scale = float(image_height) / float(max(1, gray.height))
        new_width = max(1, min(image_max_width, int(round(gray.width * scale))))
        resized = gray.resize((new_width, image_height), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (image_max_width, image_height), color=255)
    canvas.paste(resized, (0, 0))
    image_array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


class _LazyLogitDict:
    """Dict-like wrapper that extracts tensor values on demand.

    Beam search only accesses ~20-50 grammar-valid tokens per step,
    but the full vocab has ~374 tokens.  This avoids ~320 unnecessary
    ``.item()`` calls per step (× beam_width × decode_steps).
    """

    __slots__ = ("_tensor", "_idx")

    def __init__(self, log_probs_cpu, token_to_idx: Dict[str, int]) -> None:
        self._tensor = log_probs_cpu
        self._idx = token_to_idx

    def __contains__(self, token: str) -> bool:
        return token in self._idx

    def __getitem__(self, token: str) -> float:
        return self._tensor[self._idx[token]].item()


def _resolve_stage_b_decode_model(obj):
    """Unwrap DoRA/PEFT wrappers to find the model with encode_staff/decode_tokens."""
    if hasattr(obj, "encode_staff") and hasattr(obj, "decode_tokens"):
        return obj
    base_model = getattr(obj, "base_model", None)
    nested = getattr(base_model, "model", None) if base_model is not None else None
    if nested is not None and hasattr(nested, "encode_staff") and hasattr(nested, "decode_tokens"):
        return nested
    nested_direct = getattr(obj, "model", None)
    if nested_direct is not None and hasattr(nested_direct, "encode_staff") and hasattr(nested_direct, "decode_tokens"):
        return nested_direct
    raise RuntimeError("Loaded Stage-B model does not expose encode_staff/decode_tokens for Stage-B decoding.")


def _quantize_decoder(decode_model, device):
    """Apply INT8 dynamic quantization to decoder nn.Linear layers.

    On CPU: uses built-in torch.quantization.quantize_dynamic (2-3x speedup).
    On GPU: tries torchao if installed, otherwise skips with warning.
    Returns the (possibly quantized) model and whether quantization was applied.
    """
    import sys
    import torch

    if device.type == "cpu":
        try:
            quantized = torch.quantization.quantize_dynamic(
                decode_model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            print("[inference] INT8 dynamic quantization applied (CPU)", file=sys.stderr)
            return quantized, True
        except Exception as exc:
            print(f"[inference] quantization failed, skipping: {exc}", file=sys.stderr)
            return decode_model, False

    # GPU: try torchao for device-agnostic INT8
    try:
        from torchao.quantization import int8_weight_only, quantize_
        quantize_(decode_model, int8_weight_only())
        print("[inference] INT8 weight-only quantization applied (CUDA via torchao)", file=sys.stderr)
        return decode_model, True
    except ImportError:
        print(
            "[inference] torchao not installed, GPU quantization skipped. "
            "Install with: pip install torchao",
            file=sys.stderr,
        )
        return decode_model, False
    except Exception as exc:
        print(f"[inference] GPU quantization failed, skipping: {exc}", file=sys.stderr)
        return decode_model, False


def _prepare_model_for_inference(model, device, *, use_fp16: bool = False, quantize: bool = False):
    """Prepare model for inference: unwrap, eval mode, optional FP16/quantization."""
    import torch

    decode_model = _resolve_stage_b_decode_model(model)
    if use_fp16 and device.type == "cuda":
        decode_model = decode_model.half()
    else:
        use_fp16 = False
    decode_model.eval()
    if quantize and not use_fp16:
        decode_model, _ = _quantize_decoder(decode_model, device)
    elif quantize and use_fp16:
        import sys
        print("[inference] --quantize and --fp16 are mutually exclusive, skipping quantization", file=sys.stderr)
    return decode_model, use_fp16


def _encode_staff_image(decode_model, pixel_values):
    """Run encoder and extract memory tensor."""
    import torch

    with torch.inference_mode():
        encoded = decode_model.encode_staff(pixel_values)
    if isinstance(encoded, tuple):
        if not encoded:
            raise RuntimeError("encode_staff returned an empty tuple.")
        return encoded[0]
    if isinstance(encoded, dict):
        if "memory" not in encoded:
            raise RuntimeError("encode_staff dict output is missing 'memory'.")
        return encoded["memory"]
    return encoded


def _decode_stage_b_tokens(
    *,
    model,
    pixel_values,
    vocabulary,
    beam_width: int,
    max_decode_steps: int,
    length_penalty_alpha: float = 0.6,
    use_kv_cache: bool = True,
    _precomputed=None,
) -> List[str]:
    """Decode tokens for a single staff crop.

    Args:
        _precomputed: optional dict with 'decode_model', 'memory', 'token_to_idx', 'use_fp16'
                      to skip redundant model preparation and encoding.
    """
    import torch

    from src.decoding.beam_search import BeamSearchConfig, constrained_beam_search_with_state

    if _precomputed is not None:
        decode_model = _precomputed["decode_model"]
        memory = _precomputed["memory"]
        _token_to_idx = _precomputed["token_to_idx"]
        use_fp16 = _precomputed["use_fp16"]
    else:
        decode_model, use_fp16 = _prepare_model_for_inference(model, pixel_values.device)
        if use_fp16:
            pixel_values = pixel_values.half()
        memory = _encode_staff_image(decode_model, pixel_values)
        _token_to_idx = {token: idx for idx, token in enumerate(vocabulary.tokens)}

    device = memory.device

    def _step_fn(
        prefix_tokens: Sequence[str],
        parent_cache: object | None,
    ) -> tuple[Dict[str, float], object | None]:
        if not use_kv_cache or parent_cache is None:
            token_ids = vocabulary.encode(prefix_tokens, strict=True)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            layer_cache = None
        else:
            if not prefix_tokens:
                raise ValueError("Prefix tokens cannot be empty during cached decoding.")
            token_ids = vocabulary.encode([prefix_tokens[-1]], strict=True)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            layer_cache = parent_cache

        with torch.inference_mode():
            logits, _, next_cache = decode_model.decode_tokens(
                input_ids,
                memory,
                past_key_values=layer_cache,
                use_cache=bool(use_kv_cache),
            )
        next_log_probs = torch.log_softmax(logits[0, -1], dim=-1).float().cpu()
        distribution = _LazyLogitDict(next_log_probs, _token_to_idx)
        return distribution, (next_cache if use_kv_cache else None)

    beams = constrained_beam_search_with_state(
        step_fn=_step_fn,
        vocabulary=vocabulary,
        config=BeamSearchConfig(beam_width=beam_width, max_steps=max_decode_steps, length_penalty_alpha=length_penalty_alpha),
        soft_penalty_fn=None,
        prefix_tokens=["<bos>", "<staff_start>"],
    )
    if not beams:
        return ["<bos>", "<staff_start>", "<staff_end>", "<eos>"]
    predicted = list(beams[0].tokens)
    if not predicted or predicted[-1] != "<eos>":
        predicted.append("<eos>")
    return predicted


def run_stage_b(args: argparse.Namespace) -> Dict[str, object]:
    import torch

    from src.tokenizer.vocab import build_default_vocabulary
    from src.train.model_factory import (
        ModelFactoryConfig,
        build_stage_b_components,
        model_factory_config_from_checkpoint_payload,
    )

    crop_rows = _read_jsonl(args.crops_manifest)
    if not crop_rows:
        raise ValueError(f"No crop rows found in {args.crops_manifest}")
    if args.checkpoint is None:
        raise ValueError("Stage-B checkpoint is required for inference.")

    device_name = str(args.device).strip() if args.device else ""
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = build_default_vocabulary()
    checkpoint_payload = torch.load(str(args.checkpoint), map_location=device)
    fallback_factory_cfg = ModelFactoryConfig(stage_b_vocab_size=vocab.size)
    factory_cfg = model_factory_config_from_checkpoint_payload(
        checkpoint_payload,
        vocab_size=vocab.size,
        fallback=fallback_factory_cfg,
    )
    components = build_stage_b_components(factory_cfg)
    model = components["model"].to(device)
    state_dict = _load_stage_b_state_dict(args.checkpoint, device)
    load_result = model.load_state_dict(state_dict, strict=False)
    loaded_keys = max(0, len(state_dict) - len(load_result.unexpected_keys))
    if loaded_keys == 0:
        raise RuntimeError(f"Checkpoint did not load any compatible Stage-B parameters: {args.checkpoint}")
    model.eval()

    # Prepare model once for all crops
    decode_model, use_fp16 = _prepare_model_for_inference(
        model, device,
        use_fp16=bool(getattr(args, "fp16", False)),
        quantize=bool(getattr(args, "quantize", False)),
    )
    _token_to_idx = {token: idx for idx, token in enumerate(vocab.tokens)}

    prediction_rows: List[Dict[str, object]] = []
    for row in crop_rows:
        crop_raw = row.get("crop_path")
        if not crop_raw:
            raise ValueError(f"Crop row missing crop_path: {row}")
        crop_path = Path(str(crop_raw))
        if not crop_path.is_absolute():
            crop_path = (args.project_root / crop_path).resolve()
        if not crop_path.exists():
            raise FileNotFoundError(f"Crop image not found: {crop_path}")

        pixel_values = _load_stage_b_crop_tensor(
            crop_path,
            image_height=max(32, int(args.image_height)),
            image_max_width=min(3000, max(256, int(args.image_max_width))),
            device=device,
        )
        if use_fp16:
            pixel_values = pixel_values.half()
        memory = _encode_staff_image(decode_model, pixel_values)

        tokens = _decode_stage_b_tokens(
            model=model,
            pixel_values=pixel_values,
            vocabulary=vocab,
            beam_width=max(1, int(args.beam_width)),
            max_decode_steps=max(8, int(args.max_decode_steps)),
            length_penalty_alpha=float(args.length_penalty_alpha),
            use_kv_cache=bool(getattr(args, "kv_cache", True)),
            _precomputed={
                "decode_model": decode_model,
                "memory": memory,
                "token_to_idx": _token_to_idx,
                "use_fp16": use_fp16,
            },
        )
        row_with_tokens = dict(row)
        row_with_tokens["tokens"] = tokens
        prediction_rows.append(row_with_tokens)

    _write_jsonl(args.output_predictions, prediction_rows)
    return {
        "input_crops": len(crop_rows),
        "predictions_written": len(prediction_rows),
        "output_predictions": str(args.output_predictions),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "missing_keys": len(load_result.missing_keys),
        "unexpected_keys": len(load_result.unexpected_keys),
    }


def _load_staff_results(staff_predictions_path: Path) -> List[StaffRecognitionResult]:
    rows = _read_jsonl(staff_predictions_path)
    results: List[StaffRecognitionResult] = []
    for row in rows:
        bbox = row.get("bbox", {})
        location = StaffLocation(
            page_index=int(row.get("page_index", 0)),
            y_top=float(bbox.get("y_min", row.get("y_top", 0.0))),
            y_bottom=float(bbox.get("y_max", row.get("y_bottom", 0.0))),
            x_left=float(bbox.get("x_min", row.get("x_left", 0.0))),
            x_right=float(bbox.get("x_max", row.get("x_right", 0.0))),
        )
        tokens = row.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError(f"Staff prediction row missing tokens list: {row}")
        results.append(
            StaffRecognitionResult(
                sample_id=str(row["sample_id"]),
                tokens=[str(token) for token in tokens],
                location=location,
                system_index_hint=(int(row["system_index"]) if "system_index" in row else None),
            )
        )
    return results


def run_assemble(args: argparse.Namespace) -> Dict[str, object]:
    staff_results = _load_staff_results(args.staff_predictions)
    assembled = assemble_score(staff_results)
    write_assembly_manifest(assembled, args.output_assembly)
    return {
        "systems": len(assembled.systems),
        "parts": len(assembled.part_order),
        "output_assembly": str(args.output_assembly),
    }


def run_export(args: argparse.Namespace) -> Dict[str, object]:
    assembled = load_assembled_score(args.assembly_manifest)
    validation = write_musicxml(assembled, args.output_musicxml)
    return validation


def _load_prediction_lookup(path: Path) -> Dict[str, List[str]]:
    rows = _read_jsonl(path)
    lookup: Dict[str, List[str]] = {}
    for row in rows:
        tokens = row.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError(f"Prediction row missing tokens list: {row}")
        if "crop_path" in row:
            key = Path(str(row["crop_path"])).name
        elif "sample_id" in row:
            key = str(row["sample_id"])
        else:
            raise ValueError(f"Prediction row missing crop_path/sample_id: {row}")
        lookup[key] = [str(token) for token in tokens]
    return lookup


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    work_dir: Path = args.work_dir
    crops_dir = work_dir / "crops"
    stage_a_manifest = work_dir / "stage_a_crops.jsonl"
    assembly_manifest = work_dir / "assembled_score.json"

    stage_a_result = run_stage_a(
        argparse.Namespace(
            image=args.image,
            weights=args.weights,
            confidence=args.confidence,
            iou=args.iou,
            output_crops_dir=crops_dir,
            output_manifest=stage_a_manifest,
            page_index=args.page_index,
        )
    )

    stage_b_result: Optional[Dict[str, object]] = None
    if args.staff_predictions is not None:
        prediction_source = args.staff_predictions
    else:
        if args.stage_b_checkpoint is None:
            raise ValueError("Provide either --staff-predictions or --stage-b-checkpoint for run mode.")
        prediction_source = work_dir / "stage_b_predictions.jsonl"
        stage_b_result = run_stage_b(
            argparse.Namespace(
                project_root=args.project_root,
                crops_manifest=stage_a_manifest,
                output_predictions=prediction_source,
                checkpoint=args.stage_b_checkpoint,
                beam_width=args.beam_width,
                max_decode_steps=args.max_decode_steps,
                length_penalty_alpha=args.length_penalty_alpha,
                image_height=args.image_height,
                image_max_width=args.image_max_width,
                device=args.stage_b_device,
                kv_cache=args.stage_b_kv_cache,
                fp16=getattr(args, "fp16", False),
                quantize=getattr(args, "quantize", False),
            )
        )

    prediction_lookup = _load_prediction_lookup(prediction_source)
    crop_rows = _read_jsonl(stage_a_manifest)
    assembled_rows = []
    for row in crop_rows:
        crop_name = Path(str(row["crop_path"])).name
        tokens = prediction_lookup.get(crop_name)
        if tokens is None:
            raise ValueError(f"No token prediction found for crop '{crop_name}'.")
        row_with_tokens = dict(row)
        row_with_tokens["tokens"] = tokens
        assembled_rows.append(row_with_tokens)

    merged_predictions = work_dir / "staff_predictions_merged.jsonl"
    _write_jsonl(merged_predictions, assembled_rows)
    assembly_result = run_assemble(
        argparse.Namespace(staff_predictions=merged_predictions, output_assembly=assembly_manifest)
    )
    export_result = run_export(
        argparse.Namespace(assembly_manifest=assembly_manifest, output_musicxml=args.output_musicxml)
    )

    return {
        "stage_a": stage_a_result,
        "stage_b": stage_b_result,
        "assembly": assembly_result,
        "export": export_result,
        "merged_staff_predictions": str(merged_predictions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OMR pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_a_parser = subparsers.add_parser("stage-a", help="Run Stage-A YOLO detection and staff cropping.")
    stage_a_parser.add_argument("--image", type=Path, required=True, help="Input page image path.")
    stage_a_parser.add_argument("--weights", type=Path, required=True, help="YOLO weights path.")
    stage_a_parser.add_argument("--output-crops-dir", type=Path, required=True, help="Output crop directory.")
    stage_a_parser.add_argument("--output-manifest", type=Path, required=True, help="Output crop manifest JSONL.")
    stage_a_parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    stage_a_parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    stage_a_parser.add_argument("--page-index", type=int, default=0, help="Page index for multi-page scores.")

    stage_b_parser = subparsers.add_parser("stage-b", help="Run Stage-B staff recognition on crop manifest.")
    stage_b_parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root for relative crop paths.")
    stage_b_parser.add_argument("--crops-manifest", type=Path, required=True, help="Stage-A crop manifest JSONL.")
    stage_b_parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-B model checkpoint path.")
    stage_b_parser.add_argument("--output-predictions", type=Path, required=True, help="Output staff predictions JSONL.")
    stage_b_parser.add_argument("--beam-width", type=int, default=8, help="Constrained beam width.")
    stage_b_parser.add_argument("--max-decode-steps", type=int, default=512, help="Max decode steps per crop.")
    stage_b_parser.add_argument("--length-penalty-alpha", type=float, default=0.6, help="Length normalization alpha (0=disabled).")
    stage_b_parser.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KV-cache decoding for speed (default: enabled).",
    )
    stage_b_parser.add_argument("--image-height", type=int, default=250, help="Input image height for Stage-B.")
    stage_b_parser.add_argument("--image-max-width", type=int, default=2500, help="Input image max width for Stage-B (max 3000).")
    stage_b_parser.add_argument("--device", type=str, default=None, help="Inference device (e.g. cuda, cpu).")
    stage_b_parser.add_argument("--fp16", action="store_true", help="Use FP16 inference on CUDA (trades precision for speed).")
    stage_b_parser.add_argument("--quantize", action="store_true", help="INT8 dynamic quantization on decoder (CPU: 2-3x faster, GPU: needs torchao).")

    assemble_parser = subparsers.add_parser("assemble", help="Assemble staff predictions into systems and parts.")
    assemble_parser.add_argument("--staff-predictions", type=Path, required=True, help="Staff predictions JSONL.")
    assemble_parser.add_argument("--output-assembly", type=Path, required=True, help="Output assembly JSON.")

    export_parser = subparsers.add_parser("export", help="Export assembled score JSON to MusicXML.")
    export_parser.add_argument("--assembly-manifest", type=Path, required=True, help="Assembly JSON path.")
    export_parser.add_argument("--output-musicxml", type=Path, required=True, help="Output MusicXML path.")

    run_parser = subparsers.add_parser("run", help="Run Stage-A + assembly + MusicXML export.")
    run_parser.add_argument("--image", type=Path, required=True, help="Input page image path.")
    run_parser.add_argument("--weights", type=Path, required=True, help="YOLO weights path.")
    run_parser.add_argument(
        "--staff-predictions",
        type=Path,
        required=False,
        help="JSONL with Stage-B token predictions per crop_path or sample_id.",
    )
    run_parser.add_argument(
        "--stage-b-checkpoint",
        type=Path,
        default=None,
        help="If staff predictions are not provided, run Stage-B inference with this checkpoint.",
    )
    run_parser.add_argument("--work-dir", type=Path, required=True, help="Working directory for intermediate files.")
    run_parser.add_argument("--output-musicxml", type=Path, required=True, help="Output MusicXML file path.")
    run_parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    run_parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    run_parser.add_argument("--page-index", type=int, default=0, help="Page index for multi-page scores.")
    run_parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root for relative crop paths.")
    run_parser.add_argument("--beam-width", type=int, default=8, help="Constrained beam width for Stage-B.")
    run_parser.add_argument("--max-decode-steps", type=int, default=512, help="Max decode steps per crop for Stage-B.")
    run_parser.add_argument("--length-penalty-alpha", type=float, default=0.6, help="Length normalization alpha (0=disabled).")
    run_parser.add_argument(
        "--stage-b-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KV-cache decoding for Stage-B in run mode (default: enabled).",
    )
    run_parser.add_argument("--image-height", type=int, default=250, help="Stage-B input height.")
    run_parser.add_argument("--image-max-width", type=int, default=2500, help="Stage-B input max width (max 3000).")
    run_parser.add_argument("--stage-b-device", type=str, default=None, help="Stage-B inference device.")
    run_parser.add_argument("--fp16", action="store_true", help="Use FP16 inference on CUDA (trades precision for speed).")
    run_parser.add_argument("--quantize", action="store_true", help="INT8 dynamic quantization on decoder (CPU: 2-3x faster, GPU: needs torchao).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "stage-a":
        result = run_stage_a(args)
    elif args.command == "stage-b":
        result = run_stage_b(args)
    elif args.command == "assemble":
        result = run_assemble(args)
    elif args.command == "export":
        result = run_export(args)
    elif args.command == "run":
        result = run_pipeline(args)
    else:
        raise ValueError(f"Unsupported command '{args.command}'")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
