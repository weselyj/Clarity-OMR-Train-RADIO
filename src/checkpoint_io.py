"""Shared Stage-B checkpoint loading utilities.

Single place that handles both plain and DoRA-trained checkpoints so that
``src/cli.py`` and ``src/eval/evaluate_stage_b_checkpoint.py`` use identical
load logic.

Public API
----------
load_stage_b_checkpoint(checkpoint_path, model, device, dora_config)
    Detect checkpoint format, optionally prepare model for DoRA, load state
    dict, enforce coverage threshold, and return a result dict.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SAMPLE_LIMIT = 5  # max keys to show in error/warning messages


def _strip_plain_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip ``base_model.model.`` and ``model.`` prefixes for plain checkpoints."""
    normalized: Dict[str, Any] = {}
    for name, tensor in state_dict.items():
        key = str(name)
        for prefix in ("base_model.model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = tensor
    return normalized


def load_stage_b_checkpoint(
    *,
    checkpoint_path: Path,
    model: "torch.nn.Module",
    device: "torch.device",
    dora_config: Optional[Dict[str, Any]],
    min_coverage: float = 0.50,
) -> Dict[str, Any]:
    """Load a Stage-B checkpoint into *model* with DoRA detection.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pt`` file.
    model:
        Freshly-built (unwrapped) Stage-B model.  Will be mutated in-place when
        DoRA is detected: ``_prepare_model_for_dora`` is called before
        ``load_state_dict``.
    device:
        Torch device to map tensors to.
    dora_config:
        DoRA configuration dict (from ``build_stage_b_components``).  When
        *None* and the checkpoint contains DoRA keys, loading will still be
        attempted but coverage will almost certainly fall below *min_coverage*,
        raising a ``RuntimeError``.
    min_coverage:
        Minimum fraction of checkpoint keys that must load successfully.
        Defaults to 0.50 (50%).

    Returns
    -------
    dict with keys:
        checkpoint_format   "plain" | "dora_peft"
        loaded_keys         int
        total_keys          int
        load_ratio          float
        missing_keys        int
        unexpected_keys     int
        missing_key_sample  list[str]  (up to _SAMPLE_LIMIT entries)
        unexpected_key_sample list[str]
    """
    import torch

    payload = torch.load(str(checkpoint_path), map_location=device)
    state_dict_raw: Dict[str, Any] = (
        payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    )
    if not isinstance(state_dict_raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    raw_keys = [str(k) for k in state_dict_raw.keys()]
    looks_like_dora = any("lora_" in k for k in raw_keys) or any(
        "modules_to_save" in k for k in raw_keys
    )
    checkpoint_format = "dora_peft" if looks_like_dora else "plain"

    if looks_like_dora:
        if dora_config is not None:
            from src.train.train import _prepare_model_for_dora
            model, _ = _prepare_model_for_dora(model, dora_config)
        # Use raw state dict (PEFT keys intact) for DoRA checkpoints
        state_dict = state_dict_raw
    else:
        state_dict = _strip_plain_prefix(state_dict_raw)

    model = model.to(device)
    load_result = model.load_state_dict(state_dict, strict=False)

    total_keys = len(state_dict)
    loaded_keys = max(0, total_keys - len(load_result.unexpected_keys))
    load_ratio = float(loaded_keys) / float(max(1, total_keys))

    missing_sample: List[str] = load_result.missing_keys[:_SAMPLE_LIMIT]
    unexpected_sample: List[str] = load_result.unexpected_keys[:_SAMPLE_LIMIT]

    if load_ratio < min_coverage:
        raise RuntimeError(
            f"Checkpoint load coverage is too low for reliable inference. "
            f"loaded={loaded_keys}/{total_keys} ({load_ratio:.1%}), "
            f"missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)}. "
            f"missing sample: {missing_sample}; "
            f"unexpected sample: {unexpected_sample}. "
            f"Hint: checkpoint format='{checkpoint_format}'; "
            f"{'pass dora_config to enable DoRA wrapping.' if looks_like_dora and dora_config is None else ''}"
        )

    # Emit CLI-visible samples to stderr (non-fatal, informational)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            f"[checkpoint_io] {checkpoint_format} checkpoint loaded "
            f"({load_ratio:.1%} coverage, "
            f"missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)}). "
            f"missing sample: {missing_sample}; "
            f"unexpected sample: {unexpected_sample}",
            file=sys.stderr,
        )

    return {
        "checkpoint_format": checkpoint_format,
        "loaded_keys": loaded_keys,
        "total_keys": total_keys,
        "load_ratio": load_ratio,
        "missing_keys": len(load_result.missing_keys),
        "unexpected_keys": len(load_result.unexpected_keys),
        "missing_key_sample": missing_sample,
        "unexpected_key_sample": unexpected_sample,
        # model is mutated in-place; also return it for callers that need the
        # possibly-wrapped reference (DoRA path replaces the model object).
        "_model": model,
    }
