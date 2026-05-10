"""Mode-agnostic Stage-B decoder runtime helpers.

Extracted from src/cli.py 2026-05-10 so the per-staff CLI can be archived
without breaking eval drivers (evaluate_stage_b_checkpoint, tune_penalties)
or any future per-system inference pipeline that needs the same primitives.

These helpers are agnostic to whether the input crop is per-staff or per-system;
both go through the same encode → beam-search decode loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence


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
    import torch  # noqa: F401  (kept so consumers see the import succeed)

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
    """Decode tokens for a single staff or system crop.

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
