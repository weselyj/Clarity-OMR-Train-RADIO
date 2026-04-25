#!/usr/bin/env python3
"""Factory utilities for Stage A and Stage B model construction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Optional

from src.models.davit_stage_b import (
    StageBModelConfig,
    build_dora_config,
    build_stage_b_model,
    list_dora_target_modules,
    run_stage_b_shape_smoke_test,
)
from src.models.yolo_stage_a import YoloStageA, YoloStageAConfig


@dataclass(frozen=True)
class ModelFactoryConfig:
    stage_a_weights_path: Optional[Path] = None
    stage_a_confidence_threshold: float = 0.25
    stage_a_iou_threshold: float = 0.45
    stage_b_vocab_size: int = 380
    stage_b_max_decode_length: int = 512
    stage_b_backbone: str = "davit_base.msft_in1k"
    stage_b_decoder_dim: int = 768
    stage_b_decoder_layers: int = 8
    stage_b_decoder_heads: int = 12
    stage_b_dora_rank: int = 16
    # Encoder selector: 'davit' uses the timm DaViT backbone;
    # 'radio_h' uses C-RADIOv4-H (~700M params).
    stage_b_encoder: str = "davit"


def build_stage_a_model(config: Optional[ModelFactoryConfig] = None) -> YoloStageA:
    cfg = config or ModelFactoryConfig()
    stage_a_config = YoloStageAConfig(
        weights_path=cfg.stage_a_weights_path,
        confidence_threshold=cfg.stage_a_confidence_threshold,
        iou_threshold=cfg.stage_a_iou_threshold,
    )
    return YoloStageA(stage_a_config)


def build_stage_b_components(config: Optional[ModelFactoryConfig] = None) -> Dict[str, object]:
    cfg = config or ModelFactoryConfig()

    encoder = str(cfg.stage_b_encoder).lower().strip()

    if encoder == "radio_h":
        from src.models.radio_stage_b import RadioStageBConfig, RadioStageB
        radio_config = RadioStageBConfig(
            decoder_dim=cfg.stage_b_decoder_dim,
            decoder_layers=cfg.stage_b_decoder_layers,
            decoder_heads=cfg.stage_b_decoder_heads,
            vocab_size=cfg.stage_b_vocab_size,
            max_decode_len=cfg.stage_b_max_decode_length,
        )
        model = RadioStageB(radio_config)
        # DoRA rank is used downstream by _prepare_model_for_dora.
        # stage_b_config is serialised into checkpoints so the architecture
        # can be reconstructed on resume.
        stage_b_config = radio_config
        dora_cfg = build_dora_config(cfg.stage_b_dora_rank)
        # RADIO's linear layers are enumerated at DoRA-wrap time; return an
        # empty list so the upstream helper builds its own target list.
        dora_target_modules: list = []
        return {
            "model": model,
            "stage_b_config": stage_b_config,
            "dora_config": dora_cfg,
            "dora_target_modules": dora_target_modules,
        }

    if encoder not in ("davit", ""):
        warnings.warn(
            f"build_stage_b_components: unknown encoder '{encoder}'; falling back to 'davit'.",
            stacklevel=2,
        )

    # --- DaViT path (default) ---
    stage_b_config = StageBModelConfig(
        vocab_size=cfg.stage_b_vocab_size,
        max_decode_length=cfg.stage_b_max_decode_length,
        pretrained_backbone=cfg.stage_b_backbone,
        decoder_dim=cfg.stage_b_decoder_dim,
        decoder_layers=cfg.stage_b_decoder_layers,
        decoder_heads=cfg.stage_b_decoder_heads,
        dora_rank=cfg.stage_b_dora_rank,
    )
    model = build_stage_b_model(stage_b_config)
    dora_config = build_dora_config(stage_b_config.dora_rank)
    dora_target_modules = list_dora_target_modules()
    return {
        "model": model,
        "stage_b_config": stage_b_config,
        "dora_config": dora_config,
        "dora_target_modules": dora_target_modules,
    }


def model_factory_config_from_checkpoint_payload(
    payload: object,
    *,
    vocab_size: int,
    fallback: Optional[ModelFactoryConfig] = None,
) -> ModelFactoryConfig:
    def _normalize_key(name: str) -> str:
        normalized = str(name)
        for prefix in ("base_model.model.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        # Strip PEFT/DoRA wrapper path segments so legacy checkpoints map to base names.
        for segment in (".original_module.", ".base_layer.", ".modules_to_save.default."):
            normalized = normalized.replace(segment, ".")
        while ".." in normalized:
            normalized = normalized.replace("..", ".")
        return normalized

    base = fallback or ModelFactoryConfig(stage_b_vocab_size=vocab_size)
    if not isinstance(payload, dict):
        return base
    raw_cfg = payload.get("stage_b_config")
    state_dict = payload.get("model_state_dict", payload)
    state_keys = [_normalize_key(key) for key in state_dict.keys()] if isinstance(state_dict, dict) else []
    normalized_state = (
        {_normalize_key(key): value for key, value in state_dict.items()}
        if isinstance(state_dict, dict)
        else {}
    )

    # Infer dora_rank from LoRA tensor shapes in the state dict.
    inferred_dora_rank = base.stage_b_dora_rank
    if isinstance(state_dict, dict):
        for key in state_dict:
            nk = _normalize_key(key)
            if "lora_A" in nk and hasattr(state_dict[key], "shape"):
                shape = state_dict[key].shape
                if len(shape) == 2:
                    inferred_dora_rank = int(shape[0])
                    break

    if isinstance(raw_cfg, dict):
        cfg_dora_rank = int(raw_cfg.get("dora_rank", inferred_dora_rank))
        return ModelFactoryConfig(
            stage_a_weights_path=base.stage_a_weights_path,
            stage_a_confidence_threshold=base.stage_a_confidence_threshold,
            stage_a_iou_threshold=base.stage_a_iou_threshold,
            stage_b_vocab_size=vocab_size,
            stage_b_max_decode_length=int(raw_cfg.get("max_decode_length", base.stage_b_max_decode_length)),
            stage_b_backbone=str(raw_cfg.get("pretrained_backbone", base.stage_b_backbone)),
            stage_b_decoder_dim=int(raw_cfg.get("decoder_dim", base.stage_b_decoder_dim)),
            stage_b_decoder_layers=int(raw_cfg.get("decoder_layers", base.stage_b_decoder_layers)),
            stage_b_decoder_heads=int(raw_cfg.get("decoder_heads", base.stage_b_decoder_heads)),
            stage_b_dora_rank=cfg_dora_rank,
            stage_b_encoder=str(raw_cfg.get("encoder", base.stage_b_encoder)),
        )

    # Legacy checkpoint fallback: infer architecture from tensor shapes when metadata is unavailable.
    inferred_decoder_dim = base.stage_b_decoder_dim
    inferred_decoder_layers = base.stage_b_decoder_layers
    inferred_decoder_heads = base.stage_b_decoder_heads

    if isinstance(state_dict, dict):
        embedding_weight = normalized_state.get("token_embedding.weight")
        if hasattr(embedding_weight, "shape") and len(embedding_weight.shape) == 2:
            inferred_decoder_dim = int(embedding_weight.shape[1])

        layer_pattern = re.compile(r"^decoder_blocks\.(\d+)\.q_proj\.weight$")
        layer_ids = []
        for key in state_keys:
            match = layer_pattern.match(key)
            if match:
                layer_ids.append(int(match.group(1)))
        if layer_ids:
            inferred_decoder_layers = max(layer_ids) + 1

        rope_key = "decoder_blocks.0.rope.inv_freq"
        rope_tensor = normalized_state.get(rope_key)
        if hasattr(rope_tensor, "numel"):
            head_dim = int(rope_tensor.numel()) * 2
            if head_dim > 0 and inferred_decoder_dim % head_dim == 0:
                inferred_decoder_heads = max(1, inferred_decoder_dim // head_dim)
        else:
            # Legacy PEFT checkpoints may omit RoPE buffers. Infer heads from deformable block,
            # whose output dimension is heads * 2.
            offset_weight = normalized_state.get("deformable_attention.offset_mlp.2.weight")
            offset_bias = normalized_state.get("deformable_attention.offset_mlp.2.bias")
            out_features = None
            if hasattr(offset_weight, "shape") and len(offset_weight.shape) == 2:
                out_features = int(offset_weight.shape[0])
            elif hasattr(offset_bias, "shape") and len(offset_bias.shape) == 1:
                out_features = int(offset_bias.shape[0])
            if out_features is not None and out_features >= 2:
                candidate_heads = max(1, out_features // 2)
                if inferred_decoder_dim % candidate_heads == 0:
                    inferred_decoder_heads = candidate_heads

    else:
        return base

    if inferred_decoder_dim % max(1, inferred_decoder_heads) != 0:
        for candidate in (16, 12, 10, 8, 6, 5, 4, 3, 2, 1):
            if inferred_decoder_dim % candidate == 0:
                inferred_decoder_heads = candidate
                break

    return ModelFactoryConfig(
        stage_a_weights_path=base.stage_a_weights_path,
        stage_a_confidence_threshold=base.stage_a_confidence_threshold,
        stage_a_iou_threshold=base.stage_a_iou_threshold,
        stage_b_vocab_size=vocab_size,
        stage_b_max_decode_length=base.stage_b_max_decode_length,
        stage_b_backbone=base.stage_b_backbone,
        stage_b_decoder_dim=max(64, int(inferred_decoder_dim)),
        stage_b_decoder_layers=max(1, int(inferred_decoder_layers)),
        stage_b_decoder_heads=max(1, int(inferred_decoder_heads)),
        stage_b_dora_rank=inferred_dora_rank,
        stage_b_encoder=base.stage_b_encoder,
    )


def run_stage_b_forward_smoke(
    config: Optional[ModelFactoryConfig] = None,
    batch_size: int = 2,
    token_length: int = 64,
    image_height: int = 192,
    image_width: int = 1600,
) -> Dict[str, object]:
    cfg = config or ModelFactoryConfig()
    return run_stage_b_shape_smoke_test(
        batch_size=batch_size,
        seq_len=token_length,
        image_height=image_height,
        image_width=image_width,
    )
