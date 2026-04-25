"""Stage B with C-RADIOv4-H encoder.

Replaces DaViT (86M params, hidden 768, stride-32 -> 6 x W/32 grid) with
C-RADIOv4-H (~700M, hidden 1280, patch-16 -> 12 x W/16 grid).

The positional bridge is reparameterised from 768->768 to 1280->768 by
constructing the upstream ``PositionalBridge`` with ``encoder_dim=1280``.
The decoder stack, deformable-attention block, and contour head are reused
unchanged from the DaViT module so the rest of the training pipeline does
not need to know which encoder is in use.

Per the omr-final-plan Stage B spec, the full 12 x (W/16) grid is kept for
the primary run (no pooling). RADIO produces ~4x more memory tokens than
DaViT did; the cross-attention has no length cap so this is safe.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.davit_stage_b import (
    DecoderBlock,
    DeformableContextBlock,
    PositionalBridge,
    RMSNorm,
)

RADIO_HUB_REPO = "NVlabs/RADIO"
RADIO_VERSION = "c-radio_v4-h"
RADIO_HIDDEN_DIM = 1280


class RadioEncoder(nn.Module):
    """C-RADIOv4-H wrapper that returns a (B, C, H, W) feature map.

    The wrapper:
      * loads the model via ``torch.hub`` once at construction time
      * snaps the input HxW to RADIO's nearest supported resolution
      * runs the forward in a bf16 autocast region (RADIO's expected path
        on the 5090) and casts the output back to the input dtype
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load(
            RADIO_HUB_REPO,
            "radio_model",
            version=RADIO_VERSION,
            progress=True,
            skip_validation=True,
            trust_repo=True,  # required for non-interactive SSH execution
        )
        self.hidden_dim = RADIO_HIDDEN_DIM
        if getattr(self.model, "embed_dim", None) != RADIO_HIDDEN_DIM:
            raise RuntimeError(
                f"Unexpected RADIO embed_dim: {getattr(self.model, 'embed_dim', None)}; "
                f"expected {RADIO_HIDDEN_DIM}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        nearest = self.model.get_nearest_supported_resolution(int(h), int(w))
        if nearest != (int(h), int(w)):
            x = F.interpolate(x, nearest, mode="bilinear", align_corners=False)
        in_dtype = x.dtype
        # RADIO expects pixel values in [0, 1]; the upstream pipeline already
        # clamps to that range, but be defensive in case a caller forgets.
        x = x.clamp(0.0, 1.0)
        if x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _summary, spatial_features = self.model(x, feature_fmt="NCHW")
        else:
            _summary, spatial_features = self.model(x, feature_fmt="NCHW")
        return spatial_features.to(in_dtype)


@dataclass
class RadioStageBConfig:
    """Configuration for the RADIO Stage B model."""

    decoder_dim: int = 768
    decoder_layers: int = 8
    decoder_heads: int = 12
    vocab_size: int = 487
    max_decode_len: int = 512
    # informational; actual dropout is hardcoded in DecoderBlock upstream
    dropout: float = 0.1
    contour_classes: int = 3


class RadioStageB(nn.Module):
    """Stage B with RADIO encoder and the DaViT-module decoder/bridge.

    Forward signature matches the test contract: ``model(image, tgt) -> dict``
    with at least the key ``"logits"`` of shape (B, T, vocab_size). A
    ``contour_logits`` key is also returned, mirroring the DaViT module's
    ``return_aux=True`` output.
    """

    def __init__(self, config: RadioStageBConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = RadioEncoder()
        encoder_dim = self.encoder.hidden_dim  # 1280
        self.deformable_attention = DeformableContextBlock(
            dim=encoder_dim, heads=config.decoder_heads
        )
        self.positional_bridge = PositionalBridge(
            encoder_dim=encoder_dim, decoder_dim=config.decoder_dim
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.decoder_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(config.decoder_dim, config.decoder_heads)
                for _ in range(config.decoder_layers)
            ]
        )
        self.decoder_norm = RMSNorm(config.decoder_dim)
        self.lm_head = nn.Linear(config.decoder_dim, config.vocab_size)
        self.contour_head = nn.Sequential(
            nn.Linear(config.decoder_dim, 128),
            nn.GELU(),
            nn.Linear(128, config.contour_classes),
        )
        self.max_decode_length = config.max_decode_len

    def encode_staff(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        feature_map = self.encoder(images)  # (B, 1280, H/16, W/16)
        batch, channels, height, width = feature_map.shape
        sequence = feature_map.flatten(2).transpose(1, 2)
        sequence = self.deformable_attention(sequence, height, width)
        sequence = sequence.transpose(1, 2).reshape(batch, channels, height, width)
        memory, _ = self.positional_bridge(sequence)
        contour_logits = self.contour_head(memory.mean(dim=1))
        return memory, contour_logits

    def decode_tokens(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    ]:
        hidden = self.token_embedding(decoder_input_ids)
        if past_key_values is not None and len(past_key_values) != len(self.decoder_blocks):
            raise ValueError(
                f"past_key_values length mismatch: expected {len(self.decoder_blocks)}, got {len(past_key_values)}."
            )
        next_past: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, block in enumerate(self.decoder_blocks):
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            hidden, layer_next_past = block(
                hidden,
                memory,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                if layer_next_past is None:
                    raise RuntimeError("Decoder block returned no cache while use_cache=True.")
                next_past.append(layer_next_past)
        hidden = self.decoder_norm(hidden)
        cache_tuple = tuple(next_past) if use_cache else None
        return self.lm_head(hidden), hidden, cache_tuple

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        tgt: Optional[torch.Tensor] = None,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **_: object,
    ) -> dict:
        if image is None:
            image = pixel_values
        if tgt is None:
            tgt = decoder_input_ids if decoder_input_ids is not None else input_ids
        if image is None or tgt is None:
            raise ValueError(
                "RadioStageB.forward requires an image tensor and a target/input token tensor."
            )
        memory, contour_logits = self.encode_staff(image)
        logits, _, _ = self.decode_tokens(tgt, memory)
        return {"logits": logits, "contour_logits": contour_logits}


def build_radio_stage_b(
    decoder_dim: int = 768,
    decoder_layers: int = 8,
    decoder_heads: int = 12,
    vocab_size: int = 487,
    max_decode_len: int = 512,
    dropout: float = 0.1,
    contour_classes: int = 3,
    **kwargs: object,
) -> RadioStageB:
    """Construct ``RadioStageB`` with the plan-aligned default dimensions.

    ``dropout`` and ``contour_classes`` are forwarded to ``RadioStageBConfig``.
    Unknown extra keyword arguments raise a warning rather than being silently
    swallowed, so YAML typos are surfaced at construction time.
    """
    if kwargs:
        warnings.warn(
            f"build_radio_stage_b received unknown keyword arguments and will ignore them: "
            f"{sorted(kwargs)}",
            stacklevel=2,
        )
    config = RadioStageBConfig(
        decoder_dim=decoder_dim,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        vocab_size=vocab_size,
        max_decode_len=max_decode_len,
        dropout=dropout,
        contour_classes=contour_classes,
    )
    return RadioStageB(config)
